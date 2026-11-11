"""Aggregate the real-API runs of seeds 0-3 and compare them with the surrogate.

Reads results/tables/real_llm_summary_seed{0,1,2,3}.json, writes
  results/tables/table_real_four_seeds.csv     (per-seed metrics)
  results/tables/table_real_vs_proxy.csv       (four-seed mean +/- std)
  results/tables/real_llm_four_seeds_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
SEEDS = [0, 1, 2, 3]

rows, raw = [], []
for s in SEEDS:
    p = TAB / f"real_llm_summary_seed{s}.json"
    if not p.exists():
        print(f"[skip] missing {p.name}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    raw.append(d)
    rows.append({
        "seed": s,
        "n_candidates": d["n_candidates_called"],
        "call_rate": d["call_rate"],
        "second_layer_f1_real": d["candidate_second_layer_real"]["f1"],
        "second_layer_f1_proxy": d["candidate_second_layer_proxy"]["f1"],
        "second_layer_p_real": d["candidate_second_layer_real"]["precision"],
        "second_layer_r_real": d["candidate_second_layer_real"]["recall"],
        "second_layer_fpr_real": d["candidate_second_layer_real"]["fpr"],
        "second_layer_p_proxy": d["candidate_second_layer_proxy"]["precision"],
        "second_layer_r_proxy": d["candidate_second_layer_proxy"]["recall"],
        "second_layer_fpr_proxy": d["candidate_second_layer_proxy"]["fpr"],
        "cascade_f1_real": d["cascade_full_real"]["f1"],
        "cascade_f1_proxy": d["cascade_full_proxy"]["f1"],
        "cascade_p_real": d["cascade_full_real"]["precision"],
        "cascade_r_real": d["cascade_full_real"]["recall"],
        "cascade_fpr_real": d["cascade_full_real"]["fpr"],
        "cascade_p_proxy": d["cascade_full_proxy"]["precision"],
        "cascade_r_proxy": d["cascade_full_proxy"]["recall"],
        "cascade_fpr_proxy": d["cascade_full_proxy"]["fpr"],
        "stage_weighted_f1_real": d["stage_weighted_f1_real"],
        "stage_weighted_f1_proxy": d["stage_weighted_f1_proxy"],
        "stage_accuracy_real": d["stage_accuracy_real"],
        "stage_mismatch_among_outputs": d["stage_mismatch_rate_among_outputs"],
        "stage_mismatch_true_anom": d["stage_mismatch_rate_among_true_anomalies"],
        "parse_fail_rate": d["parse_failure_rate_among_called"],
        "n_parse_fail": d["n_parse_fail"],
        "n_request_fail": d["n_request_fail"],
        "n_truncated": d["n_truncated"],
        "n_invalid_stage": d["n_invalid_stage_outputs"],
        "latency_mean_s": d["latency_ms"]["mean"] / 1000.0,
        "latency_p50_s": d["latency_ms"]["p50"] / 1000.0,
        "latency_p95_s": d["latency_ms"]["p95"] / 1000.0,
        "wall_s": d["wall_seconds"],
        "prompt_tokens": d["tokens"]["prompt"],
        "completion_tokens": d["tokens"]["completion"],
        "reasoning_tokens": d["tokens"]["reasoning"],
        "total_tokens": d["tokens"]["total"],
        "cost_cny": d["balance_cny"]["delta"],
        "cost_per_1000": d["balance_cny"]["per_1000_calls"],
        "agreement": d["agreement_real_proxy"],
        "kappa": d["cohen_kappa_real_proxy"],
        "mcnemar_b": d["mcnemar_b_real_only"],
        "mcnemar_c": d["mcnemar_c_proxy_only"],
        "mcnemar_p": d["mcnemar_p"],
    })

df = pd.DataFrame(rows)
df.to_csv(TAB / "table_real_four_seeds.csv", index=False, encoding="utf-8-sig")

num = df.select_dtypes(include=[np.number]).drop(columns=["seed"])
agg = pd.DataFrame({
    "mean": num.mean(),
    "std": num.std(ddof=1),
    "n": len(df),
})
agg.to_csv(TAB / "table_real_vs_proxy.csv", encoding="utf-8-sig")

summary = {
    "seeds": [int(s) for s in df["seed"]],
    "n_seeds": len(df),
    "per_seed": df.to_dict(orient="records"),
    "mean_std": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                 for k, v in agg.iterrows()},
    "total_cost_cny": float(df["cost_cny"].sum()),
    "total_tokens": int(df["total_tokens"].sum()),
    "total_calls": int(df["n_candidates"].sum()),
}

pairs = {
    "cascade_f1": ("cascade_f1_real", "cascade_f1_proxy"),
    "second_layer_f1": ("second_layer_f1_real", "second_layer_f1_proxy"),
    "stage_weighted_f1": ("stage_weighted_f1_real", "stage_weighted_f1_proxy"),
}
for name, (a, b) in pairs.items():
    diff = df[a] - df[b]
    t = stats.ttest_rel(df[a], df[b])
    summary[f"paired_{name}"] = {
        "real_mean": float(df[a].mean()), "real_std": float(df[a].std(ddof=1)),
        "proxy_mean": float(df[b].mean()), "proxy_std": float(df[b].std(ddof=1)),
        "mean_diff": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "relative_drop": float(diff.mean() / df[b].mean()),
        "t": float(t.statistic), "p": float(t.pvalue), "n": len(df),
    }
    if len(df) >= 3:
        try:
            w = stats.wilcoxon(df[a], df[b])
            summary[f"paired_{name}"]["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            pass

(TAB / "real_llm_four_seeds_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

pd.set_option("display.width", 200)
print(df[["seed", "n_candidates", "second_layer_f1_real", "second_layer_f1_proxy",
          "cascade_f1_real", "cascade_f1_proxy", "stage_weighted_f1_real",
          "parse_fail_rate", "latency_p50_s", "total_tokens", "cost_cny"]].to_string(index=False))
print()
for name in pairs:
    p = summary[f"paired_{name}"]
    print(f"{name}: real {p['real_mean']:.4f}±{p['real_std']:.4f} | "
          f"proxy {p['proxy_mean']:.4f}±{p['proxy_std']:.4f} | "
          f"diff {p['mean_diff']:.4f}±{p['std_diff']:.4f} | t={p['t']:.3f} p={p['p']:.4g}")
print(f"total cost CNY {summary['total_cost_cny']:.2f}; tokens {summary['total_tokens']:,}; calls {summary['total_calls']:,}")
print("wrote table_real_four_seeds.csv, table_real_vs_proxy.csv, real_llm_four_seeds_summary.json")
