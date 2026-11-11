# Anonymous artifact — LLM-LogFusion (offline-proxy and real-LLM evaluation)

This package contains the code, the synthetic-corpus generator, the public real-log
pilot data and every result table/figure reported in the manuscript
"LLM-LogFusion: A Hierarchical Multi-Source Heterogeneous Log Anomaly Detection
Method with Offline-Proxy and Real-LLM Evaluation".

The five-seed tables use an offline surrogate as the second layer (TF-IDF text
evidence + structural evidence + first-layer score, combined by a stacked
logistic regression), because the surrogate is deterministic and cheap to
re-run. The artifact additionally contains `scripts/run_real_llm.py`, which
replays the seed-0 candidate set against a real chat-completions API with the
same prompt and verdict parser, plus the per-call records of our run. A real API
key is **not** bundled; the script reads it from `DEEPSEEK_API_KEY` and is
optional for reviewing the offline results.

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

## 5. Real-LLM validation (seed 0)

`real_llm_calls_seed0.csv` records one row per API call made with the pipeline's
own prompt and verdict parser: 2 233 first-layer candidates from the seed-0 test
split (10.40% of 21 470 logs), no surrogate fallback. `real_llm_summary_seed0.json`
aggregates metrics, bootstrap intervals, McNemar test and cost;
`table_real_vs_proxy_seed0.csv` is the two-backend comparison used in the paper.

Headline numbers (one seed, `deepseek-flash`, temperature 0, `max_tokens` 4096):

| Quantity | Real API | Offline proxy (same seed) |
|---|---|---|
| Second layer F1 (2 233 candidates) | 0.8011 | 0.9829 |
| Cascade F1 (21 470 test logs) | 0.7487 | 0.9144 |
| Cascade precision / recall | 0.7422 / 0.7552 | 0.9691 / 0.8655 |
| Stage-weighted F1 | 0.5172 | 0.8594 |

Operational statistics: parse failures 7/2 233 (0.31%), request failures 0,
truncated 7, invalid stage labels 0; stage mismatch against gold 46.6% of
outputs / 53.5% of true anomalies; latency p50 3.57 s, p95 11.1 s; 2.41 M tokens
(59.7% reasoning tokens); CNY 12.33 for the run (~CNY 5.52 per 1 000 calls).
Bootstrap 95% CI: cascade real [0.7307, 0.7672], proxy [0.9027, 0.9262];
agreement 0.797, Cohen's kappa 0.596, McNemar p = 2.3e-12.

Reproducing this table requires a key and incurs real cost; the committed
per-call records allow the numbers to be checked offline.

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

## 7. Publishing this artifact (steps and account requirements)

Steps 1-3 need an author account; step 4 is optional. None of them changes the
content of this directory.

1. **Zenodo (recommended, gives a DOI).** Sign in at `https://zenodo.org`
   (author account) → *New upload* → upload
   `llm-logfusion-anonymous.zip` → fill Title/Authors/Description (copy the
   abstract) → choose licence (e.g. MIT for code; keep the LogHub citation note)
   → *Publish*. Zenodo then shows a DOI and an anonymised share link.
   Account required: yes (ORCID login or e-mail).
2. **Anonymous GitHub mirror.** Create a repository at
   `https://anonymous.4open.science/` (it mirrors a GitHub repo and strips
   author identity; a GitHub account is required to create the mirror). Upload
   the same zip content as a repository first.
3. **Figshare.** `https://figshare.com` → *New item* → upload the zip → publish
   (author account required). Use this if the venue asks for a non-Zenodo DOI.
4. **Institutional repository (optional).** If the funding agreement requires a
   domestic repository, upload the same zip there; no file changes are needed.
5. **GitHub (SSH push).** Create an empty repository (no README/LICENCE
   initialisation) named `llm-logfusion-artifact`, then from the repository root:
   ```
   git init -b main
   git add .
   git commit -m "LLM-LogFusion anonymous artifact"
   git remote add origin git@github.com:<account>/llm-logfusion-artifact.git
   git push -u origin main
   ```
   If port 22 is blocked, add this to `~/.ssh/config` and keep the usual
   `git@github.com:` remote:
   ```
   Host github.com
       HostName ssh.github.com
       Port 443
       User git
   ```

After publication, insert the DOI/link into the "Data and code availability"
statement of the manuscript (currently it says the repository is not yet public
and that readers may request the files from the corresponding author).

## 7. Contact

During anonymous review, please use the venue's discussion channel. The
corresponding author's e-mail is withheld in this package.
