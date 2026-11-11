# LLM-LogFusion: Hierarchical Multi-Source Heterogeneous Log Anomaly Detection with Real-API Validation — artifact

Repository: <https://github.com/Donaldzc/llm-logfusion-artifact> (branch `main`).

This package contains the code, the synthetic-corpus generator, the public real-log
pilot data and every result table/figure reported in the manuscript
"LLM-LogFusion: Hierarchical Multi-Source Heterogeneous Log Anomaly Detection
with Real-API Validation".

The five-seed tables use an offline surrogate as the second layer (TF-IDF text
evidence + structural evidence + first-layer score, combined by a stacked
logistic regression), because the surrogate is deterministic and cheap to
re-run. The artifact additionally contains `scripts/run_real_llm.py`, which
replays the first-layer candidate set of a given seed against a real
chat-completions API with the same prompt and verdict parser, plus the per-call
records of our four runs (seeds 0-3). A real API key is **not** bundled; the
script reads it from `DEEPSEEK_API_KEY` and is optional for reviewing the
offline results.

---

## 1. Contents

```
llm_logfusion/            framework package (features, filters, embedding,
                          projection, offline surrogate, pipeline, evaluation)
scripts/
  run_reproduction.py     main experiment: Tables 1-6 inputs
  run_extended_checks.py  low anomaly rate, threshold sweep, template leakage,
                          chain reconstruction, repeated runs
  run_review_checks.py    bootstrap CIs, McNemar test, campaign-isolated split,
                          second-layer error injection, Nyström latency
  recheck_filters_full.py full-test-set re-check of IF / HBOS / autoencoder
  review_lopo_recal.py    LOPO target-domain threshold recalibration
  analyze_chain_split.py  composition analysis of random vs campaign split
  run_deeplog_baseline.py DeepLog-style log-key baseline (frequency + transition)
  run_loghub_pilot.py     external-validity pilot on the public BGL_2k sample
  run_real_llm.py         optional real-API replay of the seed-0 candidate set
  make_figures.py         redraws Figures 2-8 from results/tables
data/
  extended_logs.csv       synthetic corpus (42,940 logs, 2,395 anomalies)
  external/               LogHub BGL_2k and HDFS_2k samples (see Section 4)
results/tables/           all CSV/JSON result files referenced by the scripts
figures/                  Figures 2-8 as PNG
docs/                     experiment-to-claim mapping and fusion diagnosis
requirements.txt          exact package versions
```

## 2. Environment

Windows 11 (or any OS with the same package versions), Python 3.12.10, CPU only:

```
pip install -r requirements.txt
```

All experiments in the paper were run on an Intel Core i9-14900HX (24C/32T,
32 GB RAM) without a GPU. The semantic encoder falls back to TF-IDF + SVD (LSA)
because no HuggingFace weights are bundled; this is the configuration reported in
the paper.

## 3. Reproduction commands

Run from the package root. Expected wall-clock times are for the reference
machine described above.

| Command | Produces | Time |
|---|---|---|
| `python scripts/run_reproduction.py` | `results/tables/*.csv` for Tables 1-6 | ~350 s |
| `python scripts/run_extended_checks.py` | low-rate, threshold, leakage, chain metrics | ~420 s |
| `python scripts/run_review_checks.py` | `review_round1.json` (bootstrap, McNemar, campaign split, error injection, Nyström latency) | ~300 s |
| `python scripts/recheck_filters_full.py` | `review_filters_full.csv` | ~120 s |
| `python scripts/review_lopo_recal.py` | `review_lopo_recalibration.csv`, `review_lopo_recal_prior.csv` | ~90 s |
| `python scripts/analyze_chain_split.py` | `chain_split_analysis.csv`, `chain_split_summary.json`, `chain_split_composition.csv` | ~180 s |
| `python scripts/run_deeplog_baseline.py` | `table_deeplog.csv`, `deeplog_summary.json` | ~60 s |
| `python scripts/run_loghub_pilot.py` | `table_loghub_pilot.csv`, `loghub_pilot.json` | ~40 s |
| `python scripts/run_real_llm.py` (needs `DEEPSEEK_API_KEY`) | `real_llm_calls_seed0.csv`, `real_llm_summary_seed0.json`, `table_real_vs_proxy_seed0.csv` | ~20 min |
| `python scripts/make_figures.py` | `figures/*.png` | ~60 s |

`docs/reproduction-data-mapping.md` maps every number in the paper to the script
and result file that produces it.

## 4. Real-log data (external validity pilot)

`data/external/` contains the 2 000-line public samples distributed with LogHub:

* `BGL_BGL_2k.log`, `BGL_BGL_2k.log_structured.csv` — 2 000 lines, 143 anomalies
  (7.15%), labelled in the `Label` column (`-` = normal). Downloaded from
  `https://github.com/logpai/loghub` (path `BGL/BGL_2k.log*`).
* `HDFS_HDFS_2k.log`, `HDFS_HDFS_2k.log_structured.csv` — 2 000 lines **without
  labels**, therefore not usable for the detection protocol. Included only for
  completeness.

LogHub is released for research use; cite He et al., "Loghub: A Large Collection
of System Logs for Data Mining and System Management" (IEEE BigData 2017
Workshop) and the BGL source (Oliner & Stearley, DSN 2007) when reusing them.
`run_loghub_pilot.py` reconstructs the pipeline's column contract from these
files: the original log line is the text input, `Time` is normalised to the
platform timestamp format, `platform` is `BGL`, and `stage` is left empty
(-1) because BGL carries no ATT&CK annotations. Only binary detection is
therefore evaluated. The 2 000-line sample is a pilot, not a full BGL run.

## 5. Real-API validation (seeds 0-3)

`real_llm_calls_seed{0,1,2,3}.csv` record one row per API call made with the
pipeline's own prompt and verdict parser: every first-layer candidate of seeds
0-3 (2 233 + 2 196 + 2 263 + 2 137 = 8 829 calls, no sampling, no surrogate
fallback).
`real_llm_summary_seed{0,1,2,3}.json` aggregate metrics, bootstrap intervals,
McNemar test and cost per seed; `aggregate_real_llm_seeds.py` merges them into
`table_real_four_seeds.csv` (per seed), `table_real_vs_proxy.csv` (mean ± std)
and `real_llm_four_seeds_summary.json` (paired tests).

Headline numbers (seeds 0-3, `deepseek-flash`, temperature 0, `max_tokens` 4096,
mean ± std):

| Quantity | Real API | Offline proxy (same seeds) |
|---|---|---|
| Second layer F1 (8 829 candidates) | 0.792 ± 0.012 | 0.979 ± 0.007 |
| Cascade F1 (4 × 21 470 test logs) | 0.720 ± 0.025 | 0.884 ± 0.023 |
| Cascade precision / recall | 0.726 ± 0.015 / 0.714 ± 0.035 | 0.962 ± 0.014 / 0.818 ± 0.035 |
| Stage-weighted F1 | 0.532 ± 0.010 | 0.861 ± 0.006 |

Paired differences (real − proxy) are −0.187 ± 0.011 (second layer) and
−0.164 ± 0.007 (cascade), paired t-test p < 1e-4.

Operational statistics over the 8 829 calls: parse failures 29 (0.33%), request
failures 0, invalid stage labels 0; stage mismatch against gold 45.1% of outputs
/ 52.2% of true anomalies; latency mean 4.87 s, p50 3.57-3.88 s, p95 11.76 s;
9.58 M tokens (91.9% reasoning tokens); CNY 31.94 in total (CNY 3.62 per 1 000
calls; seed 0 cost CNY 12.33 on a cold cache, seeds 1-3 cost CNY 5.93-7.26);
agreement 0.800, Cohen's kappa 0.602.

Reproducing these tables requires a key and incurs real cost; the committed
per-call records allow every number to be checked offline.

## 6. What is intentionally absent

* **No API key.** The main pipeline and all five-seed tables are offline. The
  optional real-LLM script reads `DEEPSEEK_API_KEY` from the environment;
  `.env.example` is a placeholder with an empty value. Our per-call records are
  included so the real-API results can be inspected without a key. The key was
  never written to any script, table, figure or document.
* **No HuggingFace model weights.** The encoder therefore uses the LSA fallback
  reported in the paper.
* **No GPU code path.** Everything runs on CPU.

**Security and privacy.** `data/extended_logs.csv` is synthetic; Windows event
payloads contain generic strings such as `C:\Windows\...` and
`C:\Users\Public\...` because they imitate Windows log templates. They are not
paths or identifiers from the authors' machine. The real-API validation sends
the candidate log text to the model provider; do not run it on production logs
without a data-processing review.

## 7. Repository and archival

The artifact is published at
<https://github.com/Donaldzc/llm-logfusion-artifact> (branch `main`). Clone and
run:

```
git clone https://github.com/Donaldzc/llm-logfusion-artifact.git
cd llm-logfusion-artifact
pip install -r requirements.txt
python scripts/run_reproduction.py
```

The repository mirrors this directory; no file is added or removed during
publication. For an archival DOI, the same content can be uploaded to Zenodo or
Figshare:

1. **Zenodo (gives a DOI).** Sign in at `https://zenodo.org` → *New upload* →
   upload `llm-logfusion-artifact.zip` → fill Title/Authors/Description (copy
   the abstract) → choose licence (MIT for code; keep the LogHub citation note)
   → *Publish*.
2. **Figshare.** `https://figshare.com` → *New item* → upload the zip → publish.
3. **Institutional repository (optional).** If the funding agreement requires a
   domestic repository, upload the same zip there; no file changes are needed.

## 8. Contact

Questions and issues: open an issue at
<https://github.com/Donaldzc/llm-logfusion-artifact/issues>. Author e-mail
addresses are not duplicated in this package; they are listed in the
manuscript and in `CITATION.cff` by name only.
