"""Round-1 review experiments: bootstrap CI, McNemar, chain-isolated split,
second-layer error injection, Nystrom latency.

Writes results/tables/review_round1.json (and prints a readable summary).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                                   # noqa: E402
from llm_logfusion.embedding import SemanticEncoder                # noqa: E402
from llm_logfusion.evaluation import classification_metrics        # noqa: E402
from llm_logfusion.nystrom import benchmark_attention              # noqa: E402
from llm_logfusion.pipeline import PipelineConfig                  # noqa: E402
from scripts.run_reproduction import evaluate_run                  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT = config.TABLE_DIR / "review_round1.json"


def bootstrap_ci(y, pred, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s, ps, rs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = classification_metrics(y[idx], pred[idx])
        f1s.append(m["f1"]); ps.append(m["precision"]); rs.append(m["recall"])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"f1": q(f1s), "precision": q(ps), "recall": q(rs)}


def mcnemar(y, a, b):
    """Exact McNemar test between two paired binary predictions."""
    a_ok = (a == y)
    b_ok = (b == y)
    n01 = int(np.sum(a_ok & ~b_ok))   # a right, b wrong
    n10 = int(np.sum(~a_ok & b_ok))   # a wrong, b right
    if n01 + n10 == 0:
        return {"n_a_only": n01, "n_b_only": n10, "p": 1.0}
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n_a_only": n01, "n_b_only": n10, "p": float(p)}


def main() -> None:
    df = pd.read_csv(config.DATA_DIR / "extended_logs.csv")
    print(f"[data] {len(df):,} logs, anomalies={int(df.label.sum())}, "
          f"campaigns={df.campaign.nunique()}", flush=True)
    emb = SemanticEncoder(backend="auto", attention="nystrom").fit(df["raw"].tolist()).encode_cached(
        df["raw"].tolist(), tag="repro")
    print(f"[embed] {emb.shape}", flush=True)

    res = {"bootstrap": [], "mcnemar": [], "campaign_split": [], "error_injection": {},
           "nystrom": []}

    # ---- 1. stratified split: default vs struct-only, bootstrap + McNemar
    pair_f1 = {"default": [], "struct": []}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=config.TEST_FRACTION,
                                  stratify=df["label"], random_state=seed)
        d = evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed))
        s = evaluate_run(df, emb, tr, te,
                         PipelineConfig(seed=seed, filter_features="struct"))
        pair_f1["default"].append(d["metrics"]["f1"])
        pair_f1["struct"].append(s["metrics"]["f1"])
        ci = bootstrap_ci(d["y"], d["res"].y_pred)
        res["bootstrap"].append({"seed": seed, "f1": d["metrics"]["f1"],
                                 "ci": ci})
        mm = mcnemar(d["y"], d["res"].y_pred, s["res"].y_pred)
        mm.update({"seed": seed, "f1_default": d["metrics"]["f1"],
                   "f1_struct": s["metrics"]["f1"]})
        res["mcnemar"].append(mm)
        # ---- 3. second-layer error injection (on default run)
        y, pred, cand = d["y"], d["res"].y_pred.copy(), d["res"].candidates
        rows = []
        rng = np.random.default_rng(100 + seed)
        for eps in (0.0, 0.05, 0.10, 0.20):
            f1s = []
            for _ in range(100):
                p = pred.copy()
                flip = rng.random(len(p)) < eps
                p[cand & flip] = 1 - p[cand & flip]
                f1s.append(classification_metrics(y, p)["f1"])
            rows.append({"eps": eps, "f1_mean": float(np.mean(f1s)),
                         "f1_p05": float(np.percentile(f1s, 5)),
                         "f1_p95": float(np.percentile(f1s, 95))})
        res["error_injection"][str(seed)] = rows
        print(f"[split] seed={seed} default={d['metrics']['f1']:.3f} "
              f"struct={s['metrics']['f1']:.3f} mcnemar_p={mm['p']:.2e}", flush=True)

    # ---- 2. campaign-isolated split (whole attack chain on one side)
    camp_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        camp = df["campaign"].fillna("").astype(str).to_numpy()
        test_mask = np.zeros(len(df), dtype=bool)
        # anomalies: split campaigns 50/50
        camps = np.array([c for c in pd.unique(camp) if c != ""])
        rng.shuffle(camps)
        test_camps = set(camps[: len(camps) // 2])
        anom = df["label"].to_numpy() == 1
        test_mask |= anom & np.isin(camp, list(test_camps))
        # normals: random 50/50
        norm_idx = np.flatnonzero(~anom)
        test_norm = rng.choice(norm_idx, size=len(norm_idx) // 2, replace=False)
        test_mask[test_norm] = True
        te = np.flatnonzero(test_mask)
        tr = np.flatnonzero(~test_mask)
        d = evaluate_run(df, emb, tr, te, PipelineConfig(seed=seed))
        s = evaluate_run(df, emb, tr, te,
                         PipelineConfig(seed=seed, filter_features="struct"))
        camp_rows.append({"seed": seed,
                          "default": d["metrics"], "struct": s["metrics"],
                          "n_train": int(len(tr)), "n_test": int(len(te)),
                          "test_anomalies": int(df.label.to_numpy()[te].sum())})
        print(f"[chain] seed={seed} default={d['metrics']['f1']:.3f} "
              f"struct={s['metrics']['f1']:.3f} "
              f"call={d['metrics']['call_rate']:.3f}", flush=True)
    res["campaign_split"] = camp_rows

    # ---- 4. Nystrom latency at several sequence lengths
    for n in (32, 64, 128, 256):
        ny, va = benchmark_attention(n_tokens=n, d=64, repeats=100)
        res["nystrom"].append({"n_tokens": n, "nystrom_ms": ny, "vanilla_ms": va,
                               "speedup": va / ny})
        print(f"[nystrom] n={n} nystrom={ny:.3f}ms vanilla={va:.3f}ms "
              f"speedup={va / ny:.2f}x", flush=True)

    # ---- 5. encoder end-to-end latency on a 2000-log subset
    sub_texts = df["raw"].tolist()[:2000]
    enc_times = {}
    for att in ("nystrom", "vanilla"):
        enc = SemanticEncoder(backend="lsa", attention=att, seed=0).fit(df["raw"].tolist())
        t0 = time.perf_counter()
        enc.encode(sub_texts)
        enc_times[att] = time.perf_counter() - t0
        print(f"[encoder] {att}: {enc_times[att]:.2f}s for 2000 logs", flush=True)
    res["encoder_latency"] = enc_times

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
