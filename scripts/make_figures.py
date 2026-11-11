"""Regenerate all Chinese figures from the reproduction result tables.

Every value is read from ``results/tables/*.csv``; nothing is hard-coded.
Run after ``scripts/run_reproduction.py`` and ``scripts/run_extended_checks.py``.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "tables"
F = ROOT / "figures"

ds = pd.read_csv(T / "table_dataset.csv")
per = pd.read_csv(T / "table_per_platform.csv")
lopo = pd.read_csv(T / "table_lopo.csv")
filt = pd.read_csv(T / "table_filters.csv")
sens = pd.read_csv(T / "table_sensitivity.csv")
trans = pd.read_csv(T / "table_transfer.csv")
base_sum_path = T / "table_baselines_summary.csv"
base = pd.read_csv(base_sum_path if base_sum_path.exists() else T / "table_baselines.csv")
main = pd.read_csv(T / "table_main.csv")

PLAT_ORDER = ["HDFS", "BGL", "OpenStack", "Windows", "Apache",
              "Firewall", "Database", "SSH", "DNS"]

# ---------------------------------------------------------------- 图2 数据集
d = ds.set_index("platform").reindex(PLAT_ORDER).reset_index()
counts = d["logs"].tolist()
n_ano, n_nor = int(d["anomalies"].sum()), int(d["logs"].sum() - d["anomalies"].sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=600,
                               gridspec_kw={"width_ratios": [1.75, 1]})
x = np.arange(len(d))
bars = ax1.bar(x, counts, width=0.62, color="#4C72B0", edgecolor="black", linewidth=0.4)
for b, v in zip(bars, counts):
    ax1.text(b.get_x() + b.get_width() / 2, v + 80, f"{v:,}", ha="center", fontsize=5.8)
ax1.set_ylim(0, max(counts) * 1.18)
ax1.set_ylabel("日志数量/条", fontsize=7.5)
ax1.set_xticks(x)
ax1.set_xticklabels(d["platform"], rotation=35, ha="right", fontsize=6.5)
ax1.tick_params(axis="y", labelsize=6.5)
ax1.set_title("(a) 各平台日志数量", fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.pie([n_nor, n_ano], labels=[f"正常\n{n_nor:,}", f"异常\n{n_ano:,}"],
        colors=["#B8C9DC", "#C44E52"], autopct="%1.2f%%", startangle=90,
        textprops={"fontsize": 6.2}, pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 0.6})
ax2.set_title("(b) 总体类别分布", fontsize=8)
plt.tight_layout(pad=0.3)
plt.savefig(F / "fig_dataset_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图3 逐平台性能
g = per.groupby("platform")[["precision", "recall", "f1"]].mean().reindex(PLAT_ORDER)
fig, ax = plt.subplots(figsize=(3.35, 2.45), dpi=600)
x = np.arange(len(g))
w = 0.26
for k, (col, color, label) in enumerate([("precision", "#4C72B0", "精确率"),
                                         ("recall", "#DD8452", "召回率"),
                                         ("f1", "#55A868", "F1")]):
    ax.bar(x + (k - 1) * w, g[col].values, width=w, color=color, edgecolor="black",
           linewidth=0.35, label=label)
ax.axhline(main["f1"].mean(), color="#C44E52", linestyle="--", linewidth=0.8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("指标值", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(g.index, rotation=32, ha="right", fontsize=6)
ax.tick_params(axis="y", labelsize=6.5)
ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(F / "fig_per_platform_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图4 留一平台
lp = lopo.set_index("held_out").reindex(PLAT_ORDER)
fig, ax = plt.subplots(figsize=(3.35, 2.45), dpi=600)
x = np.arange(len(lp))
b1 = ax.bar(x - 0.2, [main["f1"].mean()] * len(lp), width=0.38, color="#4C72B0",
            edgecolor="black", linewidth=0.35,
            label=f"平台内(均值{main['f1'].mean():.2f})")
b2 = ax.bar(x + 0.2, lp["f1"].values, width=0.38, color="#DD8452",
            edgecolor="black", linewidth=0.35,
            label=f"留一平台评估(均值{lp['f1'].mean():.2f})")
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1值", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(lp.index, rotation=32, ha="right", fontsize=6)
ax.tick_params(axis="y", labelsize=6.5)
ax.legend(fontsize=5.2, frameon=True, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(F / "fig_lopo_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图5 第一层过滤
NAME = {"Isolation Forest": "孤立森林", "HBOS": "HBOS", "LOF": "LOF",
        "One-Class SVM": "One-Class SVM", "AutoEncoder": "自编码器",
        "DBSCAN": "DBSCAN"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), dpi=600,
                               gridspec_kw={"width_ratios": [1.35, 1]})
labels = [NAME.get(v, v) for v in filt["filter"]]
x = np.arange(len(filt))
w = 0.26
for k, (col, color, lab) in enumerate([("f1", "#4C72B0", "F1"),
                                       ("precision", "#DD8452", "精确率"),
                                       ("recall", "#55A868", "召回率")]):
    ax1.bar(x + (k - 1) * w, filt[col], width=w, color=color, edgecolor="black",
            linewidth=0.35, label=lab)
ax1.set_ylim(0, 1.32)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_ylabel("指标值", fontsize=7)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=6)
ax1.tick_params(axis="y", labelsize=6.5)
ax1.legend(fontsize=5.4, frameon=False, ncol=3, loc="upper center")
ax1.set_title("(a) 检测性能", fontsize=7.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
bb = ax2.bar(x, filt["throughput"], width=0.55, color="#8FBFE8", edgecolor="black",
             linewidth=0.35)
for b, v in zip(bb, filt["throughput"]):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.0f}", ha="center", fontsize=5.2)
ax2.set_yscale("log")
ax2.set_ylim(1.0e4, 2.4e5)
ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2e4, 5e4, 1e5, 2e5]))
ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: {2e4: r"$2\times10^4$", 5e4: r"$5\times10^4$",
                  1e5: r"$10^5$", 2e5: r"$2\times10^5$"}.get(v, "")))
ax2.set_ylabel("打分吞吐量/(条·s$^{-1}$)", fontsize=7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=6)
ax2.tick_params(axis="y", labelsize=6.5)
ax2.set_title("(b) 打分速度(对数轴)", fontsize=7.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(F / "fig_filters_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图6 敏感性
s = sens.sort_values("contamination")
call = s["call_rate"] * 100
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 3.0), dpi=600,
                               gridspec_kw={"height_ratios": [1.15, 1]})
ax1.plot(call, s["f1"], marker="o", markersize=3.5, linewidth=1.2, label="F1值")
ax1.plot(call, s["precision"], marker="s", markersize=3.2, linewidth=1.1,
         linestyle="--", label="精确率")
ax1.plot(call, s["recall"], marker="^", markersize=3.2, linewidth=1.1,
         linestyle="-.", label="召回率")
ax1.set_ylim(0, 1.05)
ax1.set_xlim(call.min() - 1.2, call.max() + 1.2)
ax1.set_ylabel("指标值", fontsize=6.5)
ax1.tick_params(labelsize=5.8)
ax1.legend(fontsize=5.3, loc="center right", frameon=True)
ax1.set_title("检测指标随LLM调用比例的变化", fontsize=7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
times = s["estimated_seconds"]
bars = ax2.bar(range(len(s)), times, width=0.55, color="#8FBFE8", edgecolor="black",
               linewidth=0.35)
for b, v in zip(bars, times):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.0f}", ha="center", fontsize=5.2)
ax2.set_xticks(range(len(s)))
ax2.set_xticklabels([f"{v:.1f}%" for v in call], fontsize=5.8)
ax2.set_ylim(0, max(times) * 1.2)
ax2.set_xlabel("LLM调用比例/%", fontsize=6.5)
ax2.set_ylabel("估算时间/s", fontsize=6.5)
ax2.tick_params(axis="y", labelsize=5.8)
ax2.set_title("估算端到端开销(含API延迟摊销)", fontsize=7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout(pad=0.3, h_pad=0.6)
plt.savefig(F / "fig_sensitivity_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图7 跨层传递
MODES = [("none", "A 无传递"), ("score", "B 仅分数"),
         ("features", "C 仅特征"), ("full", "D 完整传递")]
summ_path = T / "extra_transfer_repeats_summary.csv"
if summ_path.exists():
    sm = pd.read_csv(summ_path, header=[0, 1], index_col=0)
    f1 = [float(sm.loc[m, ("f1", "mean")]) for m, _ in MODES]
    err = [float(sm.loc[m, ("f1", "std")]) for m, _ in MODES]
    fpr = [float(sm.loc[m, ("fpr", "mean")]) * 100 for m, _ in MODES]
else:
    tm = trans.set_index("transfer")
    f1 = [float(tm.loc[m, "f1"]) for m, _ in MODES]
    err = [0.0] * 4
    fpr = [float(tm.loc[m, "fpr"]) * 100 for m, _ in MODES]
fig, ax1 = plt.subplots(figsize=(3.35, 2.35), dpi=600)
x = np.arange(len(MODES))
bars = ax1.bar(x, f1, width=0.52, color="#4C72B0", edgecolor="black", linewidth=0.5,
               yerr=err, capsize=2, error_kw={"linewidth": 0.6})
ax1.set_ylim(max(0.0, min(f1) - 0.15), max(f1) + 0.20)
ax1.set_ylabel("F1值", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels([n for _, n in MODES], fontsize=6.4)
ax1.tick_params(axis="y", labelsize=7)
ax1.set_title("层间信息传递配置对比", fontsize=8.5)
for b, v, e in zip(bars, f1, err):
    ax1.text(b.get_x() + b.get_width() / 2, v - 0.012, f"{v:.3f}",
             ha="center", va="top", fontsize=6.0, color="white")
ax1.spines["top"].set_visible(False)
ax2 = ax1.twinx()
ax2.plot(x, fpr, color="#DD8452", marker="s", markersize=6, linewidth=1.6,
         linestyle="--", label="假正率(%)")
ax2.set_ylim(0, max(fpr) * 1.35 if max(fpr) > 0 else 1)
ax2.set_ylabel("假正率(%)", fontsize=8, color="#B85C1E")
ax2.tick_params(axis="y", colors="#B85C1E", labelsize=7)
ax2.spines["top"].set_visible(False)
ax2.legend(loc="upper left", fontsize=6.5, frameon=True)
plt.tight_layout(pad=0.5)
plt.savefig(F / "fig_transfer_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------ 图8 基线
NAME_B = {"Isolation Forest only": "仅孤立森林",
          "DeepLog-style log-key": "DeepLog类",
          "Text-only LLM (LogPrompt-like)": "纯文本LLM",
          "Text+struct LLM (LLM-LADE-like)": "文本+结构LLM",
          "398-d logistic classifier": "398维逻辑回归",
          "LLM-LogFusion (cascade)": "LLM-LogFusion",
          "Cascade, struct-only 1st layer": "结构第一层级联"}
bd = base.copy()
bd["label"] = bd["baseline"].map(lambda v: NAME_B.get(v, v))
bd = bd.sort_values("f1")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=600,
                               gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(bd))
colors = ["#55A868" if "LLM-LogFusion" in v else
          ("#DD8452" if "结构第一层" in v else "#7FA6D9") for v in bd["label"]]
err = bd["f1_sd"] if "f1_sd" in bd else None
bars = ax1.bar(x, bd["f1"], width=0.64, color=colors, edgecolor="black", linewidth=0.4,
               yerr=err, capsize=2, error_kw={"linewidth": 0.6})
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("F1值", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(bd["label"], rotation=28, ha="right", fontsize=6.2)
ax1.tick_params(axis="y", labelsize=7)
ax1.set_title("F1值对比", fontsize=8.5)
for b, v in zip(bars, bd["f1"]):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
             ha="center", va="bottom", fontsize=6.0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
NUDGE = {"纯文本LLM": (-34, -20), "文本+结构LLM": (-46, 16), "仅孤立森林": (5, -12),
         "LLM-LogFusion": (-40, -18), "398维逻辑回归": (-72, 8),
         "结构第一层级联": (-64, 14), "DeepLog类": (-46, -10)}
for _, r in bd.iterrows():
    ax2.scatter(r["estimated_throughput"], r["f1"], s=42,
                color="#55A868" if "LLM-LogFusion" in r["label"] else "#7FA6D9",
                marker="o", edgecolor="black", linewidth=0.4)
    ax2.annotate(r["label"], (r["estimated_throughput"], r["f1"]),
                 textcoords="offset points", xytext=NUDGE.get(r["label"], (4, 4)),
                 fontsize=5.4)
ax2.set_xscale("log")
ax2.set_xlim(8, 1.2e6)
ax2.set_ylim(0, 1.16)
ax2.set_xlabel("估算吞吐量/(条/秒，对数轴)", fontsize=7.5)
ax2.set_ylabel("F1值", fontsize=8)
ax2.set_title("F1值 vs. 估算吞吐量", fontsize=8.5)
ax2.tick_params(labelsize=6.5)
ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout(pad=0.6)
plt.savefig(F / "fig_baselines_cn.png", bbox_inches="tight", facecolor="white")
plt.close()

print("figures written")
