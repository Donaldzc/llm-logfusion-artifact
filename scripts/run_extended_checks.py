"""Extended checks used to rewrite the paper with reproduced numbers only.

Covers the questions that the earlier revision left open and that *can* be
answered with the reproduction pipeline:

``lowrate``   anomaly ratio 0.5 % / 0.1 % with contamination recalibrated on an
              inner validation split (5 seeds).
``threshold`` confidence-threshold calibration curve of the second stage.
``leakage``   template- and campaign-level overlap between train and test splits.
``chain``     chain-reconstruction metrics (LCS-F1, edit similarity, edge F1).
``repeats``   five-seed repeats with standard deviation for the transfer and
              stage tables.
``reliability`` parse-failure count and a stage-hallucination proxy.

Run ``python scripts/run_extended_checks.py --all`` after
``scripts/run_reproduction.py`` has produced the cached embeddings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / ".pylibs", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import run_reproduction as rr                                    # noqa: E402
from llm_logfusion import config                                  # noqa: E402
from llm_logfusion.evaluation import (classification_metrics, mean_ci,  # noqa: E402
                                      stage_metrics)
from llm_logfusion.embedding import SemanticEncoder               # noqa: E402
from llm_logfusion.features import StructuralFeatureExtractor     # noqa: E402
from llm_logfusion.pipeline import LLMLogFusion, PipelineConfig   # noqa: E402
from llm_logfusion.projection import AttackStageProjection        # noqa: E402
from llm_logfusion.attack_graph import build_stage_graph          # noqa: E402

TABLES = config.TABLE_DIR
SEEDS = config.SEEDS


def _split(df: pd.DataFrame, seed: int):
    return train_test_split(np.arange(len(df)), test_size=config.TEST_FRACTION,
                            stratify=df["label"], random_state=seed)


# ---------------------------------------------------------------------------
# 1. low anomaly ratio with contamination recalibration
# ---------------------------------------------------------------------------
def downsample_ratio(df: pd.DataFrame, rate: float, seed: int):
    """Keep every benign log, subsample anomalies to reach ``rate``.

    The corpus holds far fewer benign logs than a 0.5 %/0.1 % ratio would
    require, so the ratio is realised by thinning anomalies instead of
    inflating benign traffic.
    """
    rng = np.random.default_rng(seed)
    ano_idx = np.flatnonzero(df["label"].to_numpy() == 1)
    ben_idx = np.flatnonzero(df["label"].to_numpy() == 0)
    n_ano = int(round(rate * len(ben_idx) / (1 - rate)))
    n_ano = max(10, min(n_ano, len(ano_idx)))
    kept_ano = rng.choice(ano_idx, size=n_ano, replace=False)
    picked = np.concatenate([ben_idx, kept_ano])
    picked.sort()
    sub = df.iloc[picked].reset_index(drop=True)
    return sub, picked


def exp_lowrate(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    rows = []
    for rate in (0.005, 0.001):
        for seed in SEEDS:
            sub, picked = downsample_ratio(df, rate, seed)
            sub_emb = emb[picked]
            tr, te = _split(sub, seed)
            tr2, va = train_test_split(tr, test_size=0.2,
                                       stratify=sub["label"].to_numpy()[tr],
                                       random_state=seed)
            grid = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
            best_c, best_val = None, -1.0
            for c in grid:
                m = rr.evaluate_run(sub, sub_emb, tr2, va,
                                    PipelineConfig(seed=seed, contamination=c))["metrics"]
                if m["f1"] > best_val:
                    best_c, best_val = c, m["f1"]
            m = rr.evaluate_run(sub, sub_emb, tr, te,
                                PipelineConfig(seed=seed, contamination=best_c))["metrics"]
            m.update({"target_rate": rate, "seed": seed, "chosen_contamination": best_c,
                      "val_f1": best_val})
            rows.append(m)
            print(f"[lowrate] rate={rate:.3%} seed={seed} c={best_c:.4f} "
                  f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}",
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "extra_lowrate.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 2. confidence-threshold calibration
# ---------------------------------------------------------------------------
def exp_threshold(df: pd.DataFrame, emb: np.ndarray, seed: int = 0) -> pd.DataFrame:
    tr, te = _split(df, seed)
    out = rr.evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed, contamination=0.10))
    res, td = out["res"], out["test_df"]
    conf, sp = res.confidence, res.stage_pred
    true_stage = td["stage"].to_numpy()
    cand = res.candidates
    rows = []
    for t in np.round(np.arange(0.50, 0.991, 0.02), 2):
        sel = cand & (conf >= t)
        n = int(sel.sum())
        if n == 0:
            rows.append({"threshold": float(t), "n": 0, "coverage": 0.0,
                         "binary_f1": np.nan, "stage_acc": np.nan})
            continue
        y_true = td["label"].to_numpy()[sel]
        y_pred = res.y_pred[sel]
        bm = classification_metrics(y_true, y_pred)
        anom = sel & (td["label"].to_numpy() == 1)
        acc = float((sp[anom] == true_stage[anom]).mean()) if anom.sum() else np.nan
        rows.append({"threshold": float(t), "n": n,
                     "coverage": n / max(int(cand.sum()), 1),
                     "binary_f1": bm["f1"], "stage_acc": acc})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "extra_threshold.csv", index=False)
    print("[threshold] written", flush=True)
    return table


# ---------------------------------------------------------------------------
# 3. template / campaign leakage between splits
# ---------------------------------------------------------------------------
_DIGITS = re.compile(r"\b[0-9a-fA-F]{2,}\b")
_HEX = re.compile(r"[0-9a-fA-F]{8,}")


def _signature(text: str) -> str:
    s = _HEX.sub("<hex>", str(text))
    s = _DIGITS.sub("<num>", s)
    s = re.sub(r"/[^\s]+", "<path>", s)
    return s[:160]


def exp_leakage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        tr, te = _split(df, seed)
        tr_sig = set(_signature(t) for t in df["raw"].to_numpy()[tr])
        te_sig = [_signature(t) for t in df["raw"].to_numpy()[te]]
        hit = np.array([s in tr_sig for s in te_sig])
        lab = df["label"].to_numpy()[te]
        camp_tr = set(df["campaign"].to_numpy()[tr])
        camp_te = df["campaign"].to_numpy()[te]
        camp_te = camp_te[pd.notna(camp_te)]
        rows.append({"seed": seed,
                     "test_sig_overlap": float(hit.mean()),
                     "test_anomaly_sig_overlap": float(hit[lab == 1].mean()),
                     "test_benign_sig_overlap": float(hit[lab == 0].mean()),
                     "test_campaigns": int(len(set(camp_te))),
                     "campaign_overlap": float(np.mean([c in camp_tr for c in camp_te]))
                     if len(camp_te) else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "extra_leakage.csv", index=False)
    print("[leakage]\n" + table.round(3).to_string(index=False), flush=True)
    return table


# ---------------------------------------------------------------------------
# 4. chain-level metrics
# ---------------------------------------------------------------------------
def _lcs(a: Sequence[int], b: Sequence[int]) -> int:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[m, n])


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


def chain_scores(pred: Sequence[int], truth: Sequence[int]) -> Dict[str, float]:
    if not pred or not truth:
        return {"lcs_f1": 0.0, "edit_sim": 0.0, "edge_f1": 0.0}
    l = _lcs(pred, truth)
    p, r = l / len(pred), l / len(truth)
    lcs_f1 = 2 * p * r / (p + r) if p + r else 0.0
    dist = _levenshtein(pred, truth)
    edit_sim = 1 - dist / max(len(pred), len(truth))
    ep = {(pred[i], pred[i + 1]) for i in range(len(pred) - 1)}
    eg = {(truth[i], truth[i + 1]) for i in range(len(truth) - 1)}
    edge_f1 = 2 * len(ep & eg) / (len(ep) + len(eg)) if (ep or eg) else 0.0
    return {"lcs_f1": lcs_f1, "edit_sim": edit_sim, "edge_f1": edge_f1}


def exp_chain(df: pd.DataFrame, emb: np.ndarray, seed: int = 0) -> pd.DataFrame:
    tr, te = _split(df, seed)
    out = rr.evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed, contamination=0.10))
    td = out["test_df"].copy()
    td["pred"] = out["res"].y_pred
    td["sp"] = out["res"].stage_pred
    td["_dt"] = pd.to_datetime(td["timestamp"], errors="coerce", utc=True, format="mixed")
    rows = []
    for cid, g in td[td["campaign"].notna()].groupby("campaign"):
        g = g.sort_values("_dt")
        truth: List[int] = []
        for s in g["stage"].tolist():
            if int(s) >= 0 and (not truth or truth[-1] != int(s)):
                truth.append(int(s))
        if not truth:
            continue
        t0 = g["_dt"].min() - pd.Timedelta(minutes=30)
        t1 = g["_dt"].max() + pd.Timedelta(minutes=30)
        plats = set(g["platform"])
        win = td[(td["platform"].isin(plats)) & (td["_dt"] >= t0) & (td["_dt"] <= t1)
                 & (td["pred"] == 1) & (td["sp"] >= 0)].sort_values("_dt")
        pred: List[int] = []
        for s in win["sp"].tolist():
            if not pred or pred[-1] != int(s):
                pred.append(int(s))
        sc = chain_scores(pred, truth)
        sc.update({"campaign": str(cid), "pred_len": len(pred), "true_len": len(truth)})
        rows.append(sc)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "extra_chain.csv", index=False)
    summary = table[["lcs_f1", "edit_sim", "edge_f1"]].mean().round(4).to_dict()
    summary["n_campaigns"] = int(len(table))
    (TABLES / "extra_chain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[chain] {summary}", flush=True)
    return table


# ---------------------------------------------------------------------------
# 5. repeated runs (mean ± std) for transfer and stage tables
# ---------------------------------------------------------------------------
def exp_transfer_repeats(df: pd.DataFrame, emb: np.ndarray) -> Dict[str, Dict]:
    rows = []
    for seed in SEEDS:
        tr, te = _split(df, seed)
        s_tr, s_te = rr.struct_features(df, tr, te)
        X_tr = rr.fusion_block(emb, tr, s_tr)
        X_te = rr.fusion_block(emb, te, s_te)
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        model = LLMLogFusion(PipelineConfig(seed=seed, transfer="full")).fit(train_df, X_tr, s_tr)
        y = test_df["label"].to_numpy()
        for mode in ("none", "score", "features", "full"):
            model.cfg.transfer = mode
            res = model.predict(test_df, X_te, s_te)
            m = classification_metrics(y, res.y_pred)
            m.update({"seed": seed, "transfer": mode, "n_llm_calls": res.n_llm_calls})
            rows.append(m)
        print(f"[transfer+]{seed} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "extra_transfer_repeats.csv", index=False)
    summary = (table.groupby("transfer")[["f1", "precision", "recall", "fpr"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(TABLES / "extra_transfer_repeats_summary.csv")
    print("[transfer+]\n" + summary.to_string(), flush=True)
    return summary


def exp_stage_repeats() -> pd.DataFrame:
    _, darpa = rr.build_datasets()
    pairs = pd.read_csv(darpa)
    rows, per_stage = [], []
    for seed in SEEDS:
        idx = np.random.default_rng(seed).permutation(len(pairs))
        tr = pairs.iloc[idx[:config.DARPA_TRAIN_PAIRS]].reset_index(drop=True)
        te = pairs.iloc[idx[config.DARPA_TRAIN_PAIRS:]].reset_index(drop=True)
        enc = SemanticEncoder().fit(pairs["raw"].tolist())
        e = enc.encode_cached(pairs["raw"].tolist(), tag="darpa")
        ext = StructuralFeatureExtractor().fit(pairs)
        s = ext.transform(pairs)
        X = np.hstack([s, e]).astype(np.float32)
        proj = AttackStageProjection(seed=seed).fit(X[idx[:config.DARPA_TRAIN_PAIRS]],
                                                    tr["stage"].to_numpy())
        pred = proj.predict(X[idx[config.DARPA_TRAIN_PAIRS:]])
        sm = stage_metrics(te["stage"].to_numpy(), pred)
        rows.append({"seed": seed, "macro_f1": sm["macro"]["f1"],
                     "macro_p": sm["macro"]["precision"], "macro_r": sm["macro"]["recall"],
                     "weighted_f1": sm["weighted"]["f1"],
                     "weighted_p": sm["weighted"]["precision"],
                     "weighted_r": sm["weighted"]["recall"],
                     "accuracy": sm["accuracy"]})
        for r in sm["rows"]:
            per_stage.append({"seed": seed, **r})
        print(f"[stage+] seed={seed} wF1={sm['weighted']['f1']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "extra_stage_repeats.csv", index=False)
    ps = pd.DataFrame(per_stage)
    ps.to_csv(TABLES / "extra_stage_per_stage_repeats.csv", index=False)
    (TABLES / "extra_stage_repeats_summary.json").write_text(json.dumps({
        k: {"mean": float(table[k].mean()), "std": float(table[k].std(ddof=1))}
        for k in ("macro_f1", "weighted_f1", "accuracy")}, indent=2), encoding="utf-8")
    print(table.round(4).to_string(index=False), flush=True)
    return table


# ---------------------------------------------------------------------------
# 6. surrogate reliability: parse failures and stage "hallucination"
# ---------------------------------------------------------------------------
def exp_reliability(df: pd.DataFrame, emb: np.ndarray, seed: int = 0) -> Dict:
    tr, te = _split(df, seed)
    out = rr.evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed, contamination=0.10))
    td, res = out["test_df"].copy(), out["res"]
    td["pred"], td["sp"] = res.y_pred, res.stage_pred
    det = td[(td["pred"] == 1) & (td["label"] == 1) & (td["campaign"].notna())]
    true_sets = td[td["campaign"].notna()].groupby("campaign")["stage"].apply(
        lambda s: set(int(v) for v in s if int(v) >= 0))
    hall = [int(r["sp"]) not in true_sets.get(r["campaign"], set()) for _, r in det.iterrows()]
    stage_acc = float((det["sp"] == det["stage"]).mean()) if len(det) else float("nan")
    summary = {
        "backend": res.source,
        "n_candidates": int(res.candidates.sum()),
        "n_llm_calls": int(res.n_llm_calls),
        "parse_failures": 0,
        "parse_failure_note": "离线代理不产生自由文本，无 JSON 解析环节；API 后端需重跑统计",
        "detected_true_anomalies": int(len(det)),
        "stage_accuracy_on_detected": stage_acc,
        "stage_hallucination_rate": float(np.mean(hall)) if hall else float("nan"),
        "hallucination_definition": "预测阶段不在该样本所属攻击链的真实阶段集合中",
    }
    (TABLES / "extra_reliability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[reliability]", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    todo = {"lowrate", "threshold", "leakage", "chain", "repeats", "reliability"}
    if args.only:
        todo = set(args.only)

    t0 = time.perf_counter()
    df = rr.load_corpus()
    emb = rr.build_embeddings(df)

    if "leakage" in todo:
        exp_leakage(df)
    if "lowrate" in todo:
        exp_lowrate(df, emb)
    if "threshold" in todo:
        exp_threshold(df, emb)
    if "chain" in todo:
        exp_chain(df, emb)
    if "repeats" in todo:
        exp_transfer_repeats(df, emb)
        exp_stage_repeats()
    if "reliability" in todo:
        exp_reliability(df, emb)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
