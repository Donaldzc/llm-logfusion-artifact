"""Why does the campaign-isolated split score 0.902 while the random split
scores 0.886?

The reviewer asked to check whether the gap is an artefact of a different test
composition (class/platform mix, template overlap) before offering any
mechanism.  This script computes, for the same five seeds:

  * test-set size, anomaly count and anomaly rate of both splits;
  * per-platform test-anomaly counts and per-platform F1 (default config);
  * template (signature) overlap between train and test, overall and for
    anomalies;
  * a reweighted counterfactual: pool the per-platform F1 of one split with the
    platform weights of the *other* split.

Outputs
-------
results/tables/chain_split_analysis.csv   per seed x split
results/tables/chain_split_summary.json   summary + counterfactual
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                                   # noqa: E402
from llm_logfusion.embedding import SemanticEncoder                # noqa: E402
from llm_logfusion.evaluation import classification_metrics        # noqa: E402
from llm_logfusion.pipeline import PipelineConfig                  # noqa: E402
from scripts.run_extended_checks import _signature                 # noqa: E402
from scripts.run_reproduction import evaluate_run                  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT_CSV = config.TABLE_DIR / "chain_split_analysis.csv"
OUT_JSON = config.TABLE_DIR / "chain_split_summary.json"


def random_split(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    tr, te = train_test_split(np.arange(len(df)), test_size=config.TEST_FRACTION,
                              stratify=df["label"], random_state=seed)
    return tr, te


def campaign_split(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same construction as scripts/run_review_checks.py."""
    rng = np.random.default_rng(seed)
    camp = df["campaign"].fillna("").astype(str).to_numpy()
    test_mask = np.zeros(len(df), dtype=bool)
    camps = np.array([c for c in pd.unique(camp) if c != ""])
    rng.shuffle(camps)
    test_camps = set(camps[: len(camps) // 2])
    anom = df["label"].to_numpy() == 1
    test_mask |= anom & np.isin(camp, list(test_camps))
    norm_idx = np.flatnonzero(~anom)
    test_mask[rng.choice(norm_idx, size=len(norm_idx) // 2, replace=False)] = True
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def describe(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
             split: str, seed: int, emb: np.ndarray) -> dict:
    lab = df["label"].to_numpy()
    plat = df["platform"].to_numpy()
    tr_sig = set(_signature(t) for t in df["raw"].to_numpy()[tr])
    te_sig = [_signature(t) for t in df["raw"].to_numpy()[te]]

    out = evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed))
    y, pred = out["y"], out["res"].y_pred

    row = {"split": split, "seed": seed, "n_test": int(len(te)),
           "n_test_anomalies": int(lab[te].sum()),
           "test_anomaly_rate": float(lab[te].mean()),
           "test_sig_overlap": float(np.mean([s in tr_sig for s in te_sig])),
           "test_anomaly_sig_overlap": float(np.mean(
               [s in tr_sig for s, l in zip(te_sig, lab[te]) if l == 1])),
           "f1_overall": float(classification_metrics(y, pred)["f1"])}
    for p in config.PLATFORMS:
        m = plat[te] == p
        row[f"anom_{p}"] = int(lab[te][m].sum())
        row[f"f1_{p}"] = (float(classification_metrics(y[m], pred[m])["f1"])
                          if m.any() and lab[te][m].sum() else np.nan)
    return row


def reweight(rows: pd.DataFrame, weights_from: pd.DataFrame) -> float:
    """Pool per-platform F1 of ``rows`` with anomaly weights of ``weights_from``."""
    num = den = 0.0
    for p in config.PLATFORMS:
        w = float(weights_from[f"anom_{p}"])
        f = float(rows[f"f1_{p}"])
        if np.isfinite(f) and w > 0:
            num += w * f
            den += w
    return num / den if den else float("nan")


def main() -> None:
    df = pd.read_csv(config.DATA_DIR / "extended_logs.csv")
    emb = SemanticEncoder(backend="auto", attention="nystrom").fit(
        df["raw"].tolist()).encode_cached(df["raw"].tolist(), tag="repro")

    rows = []
    for seed in SEEDS:
        for split, fn in (("random", random_split), ("campaign", campaign_split)):
            tr, te = fn(df, seed)
            r = describe(df, tr, te, split, seed, emb)
            rows.append(r)
            print(f"[{split}] seed={seed} F1={r['f1_overall']:.3f} "
                  f"test_anom={r['n_test_anomalies']} "
                  f"rate={r['test_anomaly_rate']:.3f} "
                  f"overlap={r['test_sig_overlap']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    rand = table[table.split == "random"].reset_index(drop=True)
    camp = table[table.split == "campaign"].reset_index(drop=True)
    summary = {
        "random_f1_mean": float(rand["f1_overall"].mean()),
        "random_f1_std": float(rand["f1_overall"].std(ddof=1)),
        "campaign_f1_mean": float(camp["f1_overall"].mean()),
        "campaign_f1_std": float(camp["f1_overall"].std(ddof=1)),
        "delta": float(camp["f1_overall"].mean() - rand["f1_overall"].mean()),
        "random_test_anomalies": rand["n_test_anomalies"].tolist(),
        "campaign_test_anomalies": camp["n_test_anomalies"].tolist(),
        "random_test_rate": float(rand["test_anomaly_rate"].mean()),
        "campaign_test_rate": float(camp["test_anomaly_rate"].mean()),
        "random_overlap": float(rand["test_sig_overlap"].mean()),
        "campaign_overlap": float(camp["test_sig_overlap"].mean()),
        "random_anomaly_overlap": float(rand["test_anomaly_sig_overlap"].mean()),
        "campaign_anomaly_overlap": float(camp["test_anomaly_sig_overlap"].mean()),
        "counterfactual": {
            "campaign_f1_with_random_weights": [
                reweight(camp.iloc[i], rand.iloc[i]) for i in range(len(camp))],
            "random_f1_with_campaign_weights": [
                reweight(rand.iloc[i], camp.iloc[i]) for i in range(len(rand))],
        },
    }
    summary["counterfactual"]["campaign_with_random_weights_mean"] = float(
        np.mean(summary["counterfactual"]["campaign_f1_with_random_weights"]))
    summary["counterfactual"]["random_with_campaign_weights_mean"] = float(
        np.mean(summary["counterfactual"]["random_f1_with_campaign_weights"]))

    # per-platform anomaly composition (mean over seeds)
    comp = pd.DataFrame({
        "platform": config.PLATFORMS,
        "random_anomalies": [rand[f"anom_{p}"].mean() for p in config.PLATFORMS],
        "campaign_anomalies": [camp[f"anom_{p}"].mean() for p in config.PLATFORMS],
        "random_f1": [rand[f"f1_{p}"].mean() for p in config.PLATFORMS],
        "campaign_f1": [camp[f"f1_{p}"].mean() for p in config.PLATFORMS],
    })
    comp["random_share"] = comp["random_anomalies"] / comp["random_anomalies"].sum()
    comp["campaign_share"] = (comp["campaign_anomalies"]
                              / comp["campaign_anomalies"].sum())
    comp.round(4).to_csv(config.TABLE_DIR / "chain_split_composition.csv", index=False)
    summary["per_platform"] = comp.round(4).to_dict(orient="records")

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_platform"},
                     indent=2, ensure_ascii=False), flush=True)
    print(comp.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
