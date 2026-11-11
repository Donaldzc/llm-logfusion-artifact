"""Isolation Forest implemented from the algorithmic description in III-C-2.

Random feature selection -> random split value -> recursive partitioning; the
anomaly score is 2^(-E[h(x)] / c(n)) so that shorter isolation paths (fewer
splits) receive higher anomaly scores.  The decision threshold is the
(1 - contamination) quantile of the training scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import config

_EULER = 0.5772156649015329


def _harmonic(i: int) -> float:
    return np.log(i) + _EULER if i > 0 else 0.0


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


@dataclass
class _Tree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf: np.ndarray
    depth: np.ndarray
    size: np.ndarray
    max_depth: int


class IsolationForest:
    def __init__(self, n_estimators: int = config.IF_N_TREES,
                 max_samples: int = config.IF_MAX_SAMPLES,
                 contamination: float = config.IF_CONTAMINATION,
                 random_state: int = config.IF_RANDOM_STATE,
                 block_sizes: tuple[int, ...] | None = None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.block_sizes = block_sizes
        self.trees_: List[_Tree] = []
        self.threshold_: float = 0.0
        self._norm = 1.0
        self._blocks: List[tuple[int, int]] | None = None
        self._n_features = 0

    # ------------------------------------------------------------------ build
    def _candidate_groups(self, rng: np.random.Generator):
        """Column order for a node: blocks first, then features inside a block.

        Without ``block_sizes`` this is the plain random permutation of all
        columns, which gives a d-dimensional block d / sum(d) of the splits.
        With ``block_sizes`` every block receives the same probability, so a
        384-d embedding cannot crowd out a 14-d structural block.
        """
        if not self._blocks:
            cols = np.arange(self._n_features)
            rng.shuffle(cols)
            return (cols,)
        groups = []
        for b in rng.permutation(len(self._blocks)):
            start, width = self._blocks[int(b)]
            groups.append(rng.permutation(width) + start)
        return groups

    def _build_tree(self, X: np.ndarray, rng: np.random.Generator) -> _Tree:
        height_limit = int(np.ceil(np.log2(max(2, self.max_samples))))
        feat, thr, left, right, leaf, depth, size = [], [], [], [], [], [], []

        def new_node() -> int:
            feat.append(-1)
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            leaf.append(True)
            depth.append(0)
            size.append(0)
            return len(feat) - 1

        root = new_node()
        stack = [(root, np.arange(X.shape[0]), 0)]
        while stack:
            node, idx, d = stack.pop()
            depth[node] = d
            size[node] = len(idx)
            if d >= height_limit or len(idx) <= 1:
                continue
            groups = self._candidate_groups(rng)
            split_done = False
            for cols in groups:
                for f in cols:
                    col = X[idx, f]
                    lo, hi = float(col.min()), float(col.max())
                    if hi - lo < 1e-12:
                        continue
                    t = float(rng.uniform(lo, hi))
                    mask = col <= t
                    if mask.all() or not mask.any():
                        continue
                    feat[node] = int(f)
                    thr[node] = t
                    leaf[node] = False
                    l_id, r_id = new_node(), new_node()
                    left[node], right[node] = l_id, r_id
                    stack.append((l_id, idx[mask], d + 1))
                    stack.append((r_id, idx[~mask], d + 1))
                    split_done = True
                    break
                if split_done:
                    break
            if not split_done:
                continue
        return _Tree(np.asarray(feat), np.asarray(thr, float), np.asarray(left),
                     np.asarray(right), np.asarray(leaf, bool), np.asarray(depth),
                     np.asarray(size), height_limit)

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=np.float32)
        self._n_features = X.shape[1]
        if self.block_sizes and sum(self.block_sizes) == self._n_features:
            starts = np.cumsum([0] + list(self.block_sizes[:-1]))
            self._blocks = [(int(s), int(w))
                            for s, w in zip(starts, self.block_sizes)]
        else:
            self._blocks = None
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        sub = min(self.max_samples, n)
        self._norm = _c(sub) or 1.0
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=sub, replace=False) if sub < n else np.arange(n)
            self.trees_.append(self._build_tree(X[idx], rng))
        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    # ----------------------------------------------------------------- score
    def _tree_path(self, tree: _Tree, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        rows = np.arange(n)
        node = np.zeros(n, dtype=np.int64)
        for _ in range(tree.max_depth + 2):
            if tree.leaf[node].all():
                break
            f = tree.feature[node]
            f_safe = np.where(f < 0, 0, f)
            go_left = X[rows, f_safe] <= tree.threshold[node]
            nxt = np.where(go_left, tree.left[node], tree.right[node])
            node = np.where(tree.leaf[node], node, nxt)
        h = tree.depth[node].astype(np.float64)
        leaf_size = tree.size[node]
        cmap = {int(s): _c(int(s)) for s in np.unique(leaf_size)}
        h = h + np.array([cmap[int(s)] for s in leaf_size])
        return h

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.trees_:
            raise RuntimeError("IsolationForest is not fitted")
        h = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees_:
            h += self._tree_path(tree, X)
        h /= len(self.trees_)
        return np.power(2.0, -h / self._norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Boolean array, True = anomaly."""
        return self.score_samples(X) >= self.threshold_
