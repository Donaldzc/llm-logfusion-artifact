"""MITRE ATT&CK attack knowledge graph with PageRank-optimised topology.

Nodes are the 12 attack stages, edge weights are the normalised co-occurrence
frequency of stage transitions observed in the LLM detection output, and node
importance is obtained with PageRank (alpha = 0.85, tolerance 1e-6).  Nodes with
PR(i) < 0.5/N are pruned as described in Section III-C-3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import config

SHORT = ["Init", "Exec", "Persist", "PrivEsc", "Evas", "Cred",
         "Disc", "LatMov", "Coll", "C2", "Exfil", "Impact"]


@dataclass
class StageGraph:
    n_stages: int
    counts: np.ndarray
    weights: np.ndarray
    pagerank: np.ndarray
    kept: List[int] = field(default_factory=list)

    @property
    def prune_threshold(self) -> float:
        return config.PR_PRUNE_FACTOR / self.n_stages

    def top_edges(self, k: int = 20) -> List[Tuple[int, int, float]]:
        edges = [(i, j, float(self.weights[i, j]))
                 for i in range(self.n_stages) for j in range(self.n_stages)
                 if self.weights[i, j] > 0 and i != j]
        edges.sort(key=lambda e: e[2], reverse=True)
        return edges[:k]


def build_stage_graph(sequences: Iterable[Sequence[int]],
                      n_stages: int = config.N_STAGES) -> StageGraph:
    counts = np.zeros((n_stages, n_stages), dtype=np.float64)
    for seq in sequences:
        stages = [int(s) for s in seq if 0 <= int(s) < n_stages]
        for a, b in zip(stages[:-1], stages[1:]):
            if a != b:
                counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    weights = np.divide(counts, row_sums, out=np.zeros_like(counts),
                        where=row_sums > 0)
    pr = pagerank(weights)
    kept = [i for i in range(n_stages) if pr[i] >= config.PR_PRUNE_FACTOR / n_stages]
    return StageGraph(n_stages, counts, weights, pr, kept)


def pagerank(W: np.ndarray, alpha: float = config.PR_ALPHA,
             tol: float = config.PR_TOL, max_iter: int = config.PR_MAX_ITER,
             ) -> np.ndarray:
    """PR(i) = (1-alpha)/N + alpha * sum_j w_ji PR(j)."""
    n = W.shape[0]
    dangling = W.sum(axis=1) == 0
    pr = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        contrib = W.T @ pr
        if dangling.any():
            contrib += pr[dangling].sum() / n
        nxt = (1 - alpha) / n + alpha * contrib
        if np.abs(nxt - pr).sum() < tol:
            pr = nxt
            break
        pr = nxt
    return pr / pr.sum()


def _is_subsequence(needle: Sequence[int], haystack: Sequence[int]) -> bool:
    it = iter(haystack)
    return all(any(x == y for y in it) for x in needle)


def chain_completeness(predicted: Dict[str, Sequence[int]],
                       campaigns: Dict[str, Sequence[int]]) -> float:
    """Fraction of ground-truth campaigns whose stage order is recovered."""
    if not campaigns:
        return 0.0
    ok = 0
    for campaign, stages in campaigns.items():
        platform = campaign.rsplit("-C", 1)[0]
        seq = [int(s) for s in predicted.get(platform, []) if s >= 0]
        if _is_subsequence([int(s) for s in stages], seq):
            ok += 1
    return ok / len(campaigns)


def attack_coverage(predicted: Dict[str, Sequence[int]],
                    ground_truth: Dict[str, Sequence[int]]) -> float:
    gt_stages = {int(s) for seq in ground_truth.values() for s in seq if s >= 0}
    if not gt_stages:
        return 0.0
    pred_stages = {int(s) for seq in predicted.values() for s in seq if s >= 0}
    return len(gt_stages & pred_stages) / len(gt_stages)


def plot_stage_graph(graph: StageGraph, path, title: str = "MITRE ATT&CK knowledge graph") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 5.6))
    positions = {}
    for i in range(graph.n_stages):
        row, col = divmod(i, 6)
        positions[i] = (col * 1.6, 1.0 - row * 1.25)

    cmap = plt.get_cmap("YlOrRd")
    vmax = max(1e-6, float(graph.pagerank.max()))
    for i in range(graph.n_stages):
        pr = float(graph.pagerank[i])
        kept = i in graph.kept
        color = cmap(0.25 + 0.75 * config.STAGE_SEVERITY[i]) if kept else "#e5e7eb"
        size = 600 + 5200 * (pr / vmax)
        ax.scatter(*positions[i], s=size, color=color, edgecolor="#374151",
                   linewidth=1.4, linestyle="-" if kept else "--", zorder=3)
        ax.text(*positions[i], f"{SHORT[i]}\nPR={pr:.3f}", ha="center",
                va="center", fontsize=7.5, zorder=4,
                color="#111827" if kept else "#6b7280")

    max_w = max([graph.weights[i, j] for i in range(graph.n_stages)
                 for j in range(graph.n_stages)] + [1e-6])
    for i in range(graph.n_stages):
        for j in range(graph.n_stages):
            w = graph.weights[i, j]
            if w <= 0 or i == j:
                continue
            if i not in graph.kept and j not in graph.kept:
                continue
            arrow = FancyArrowPatch(positions[i], positions[j],
                                    arrowstyle="-|>", mutation_scale=10 + 8 * w,
                                    linewidth=0.6 + 4.0 * (w / max_w),
                                    color=plt.get_cmap("Blues")(0.35 + 0.55 * w / max_w),
                                    alpha=0.75, shrinkA=22, shrinkB=22,
                                    connectionstyle="arc3,rad=0.12", zorder=2)
            ax.add_patch(arrow)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlim(-0.9, 8.9)
    ax.set_ylim(-0.9, 1.9)
    ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.9),
                      markersize=10, label="high severity / high PageRank"),
               Line2D([0], [0], linestyle="--", color="#9ca3af",
                      label="pruned (PR < 0.5/N)")]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
