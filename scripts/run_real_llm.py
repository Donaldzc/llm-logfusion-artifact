"""Run the second stage against the real DeepSeek API on the cascade's candidate set.

The script reproduces the exact feature construction, split (seed 0) and first
layer of ``scripts/run_reproduction.py``.  It then sends every log that the
Isolation Forest forwards to the second stage to the chat-completions API with
the same prompt template and the same verdict parser used by the offline
surrogate (``llm_logfusion.llm_layer``).  Per-call records are written
incrementally so an interrupted run can be inspected.

The API key is read from ``DEEPSEEK_API_KEY`` only; it is never written to any
result file, log, table or figure.

Usage
-----
    $env:DEEPSEEK_API_KEY = "<your-key>"  # never commit the key
    python scripts/run_real_llm.py --model deepseek-flash --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                                   # noqa: E402
from llm_logfusion.embedding import SemanticEncoder                # noqa: E402
from llm_logfusion.evaluation import classification_metrics, stage_metrics  # noqa: E402
from llm_logfusion.features import StructuralFeatureExtractor      # noqa: E402
from llm_logfusion.llm_layer import build_prompt, _parse_verdict   # noqa: E402
from llm_logfusion.pipeline import LLMLogFusion, PipelineConfig    # noqa: E402

TABLE_DIR = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# environment / HTTP helpers
# ---------------------------------------------------------------------------
def _api_key() -> str:
    key = os.environ.get(config.LLM_API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"{config.LLM_API_KEY_ENV} is not set")
    return key


def _balance(base_url: str, key: str, timeout: int = 30) -> Optional[float]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/user/balance",
        headers={"Authorization": f"Bearer {key}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        infos = body.get("balance_infos") or []
        if infos:
            return float(infos[0].get("total_balance", "nan"))
    except Exception as exc:  # balance is diagnostic only
        print(f"[balance] unavailable: {type(exc).__name__}", flush=True)
    return None


def _post_chat(prompt: str, model: str, max_tokens: int, timeout: int,
               base_url: str, key: str) -> Dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "top_p": config.LLM_TOP_P,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
        latency_ms = (time.perf_counter() - t0) * 1000.0
        body = json.loads(raw)
        choice = body["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        raw_stage = ""
        try:
            obj = json.loads(content)
            raw_stage = str(obj.get("attack_stage", "")).strip()
        except Exception:
            raw_stage = ""
        parse_ok, parse_err = True, ""
        try:
            verdict = _parse_verdict(content)
        except Exception as exc:
            parse_ok = False
            parse_err = f"{type(exc).__name__}: {exc}"[:200]
            verdict = {"is_anomaly": False, "stage": -1, "confidence": 0.0}
        return {
            "http_ok": 1, "status": status, "retryable": False,
            "latency_ms": latency_ms,
            "is_anomaly": int(bool(verdict["is_anomaly"])),
            "stage": int(verdict["stage"]), "confidence": float(verdict["confidence"]),
            "parse_ok": int(parse_ok), "parse_error": parse_err,
            "raw_stage": raw_stage,
            "finish_reason": str(choice.get("finish_reason", "")),
            "model": str(body.get("model", "")),
            "system_fingerprint": str(body.get("system_fingerprint", "")),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "reasoning_tokens": int(details.get("reasoning_tokens", 0)),
            "cached_tokens": int(usage.get("prompt_cache_hit_tokens",
                                           (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))),
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = ""
        return {
            "http_ok": 0, "status": exc.code,
            "retryable": exc.code in (408, 429, 500, 502, 503, 504),
            "latency_ms": latency_ms, "is_anomaly": 0, "stage": -1,
            "confidence": 0.0, "parse_ok": 0, "parse_error": "",
            "raw_stage": "", "finish_reason": "", "model": "",
            "system_fingerprint": "", "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0,
            "cached_tokens": 0, "error": detail,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "http_ok": 0, "status": -1, "retryable": True,
            "latency_ms": latency_ms, "is_anomaly": 0, "stage": -1,
            "confidence": 0.0, "parse_ok": 0, "parse_error": "",
            "raw_stage": "", "finish_reason": "", "model": "",
            "system_fingerprint": "", "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0,
            "cached_tokens": 0, "error": f"{type(exc).__name__}: {exc}"[:300],
        }


def _call_with_retry(prompt: str, args, key: str, abort: threading.Event) -> Dict:
    last: Dict = {}
    for attempt in range(args.retries + 1):
        if abort.is_set():
            return {"http_ok": 0, "status": -2, "retryable": False,
                    "latency_ms": 0.0, "is_anomaly": 0, "stage": -1,
                    "confidence": 0.0, "parse_ok": 0, "parse_error": "",
                    "raw_stage": "", "finish_reason": "", "model": "",
                    "system_fingerprint": "", "prompt_tokens": 0,
                    "completion_tokens": 0, "reasoning_tokens": 0,
                    "cached_tokens": 0, "error": "aborted"}
        last = _post_chat(prompt, args.model, args.max_tokens, args.timeout,
                          args.base_url, key)
        if last["http_ok"] or not last["retryable"]:
            return last
        if last["status"] in (401, 402):
            abort.set()
            return last
        time.sleep(min(2 ** attempt, 10) + 0.3 * attempt)
    return last


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _bootstrap_f1(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                  seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = classification_metrics(y[idx], p[idx])["f1"]
    return {"lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def _stage_summary(truth: np.ndarray, pred: np.ndarray) -> Dict:
    sm = stage_metrics(truth, pred)
    return {"weighted_f1": sm["weighted"]["f1"],
            "macro_f1": sm["macro"]["f1"],
            "accuracy": sm["accuracy"],
            "per_stage": sm["rows"]}


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL",
                                                         "https://api.deepseek.com"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all candidates; otherwise stratified subsample")
    ap.add_argument("--tag", default="seed0")
    args = ap.parse_args()

    key = _api_key()
    print(f"[api] base={args.base_url} model={args.model} workers={args.workers}",
          flush=True)
    bal0 = _balance(args.base_url, key)

    # ---- data, embeddings, split (identical to run_reproduction.py) ----
    df = pd.read_csv(ROOT / "data" / "extended_logs.csv")
    print(f"[data] {len(df):,} logs | anomalies {int(df.label.sum()):,}", flush=True)
    enc = SemanticEncoder(backend="auto", attention="nystrom").fit(df["raw"].tolist())
    emb = enc.encode_cached(df["raw"].tolist(), tag="repro")
    print(f"[embed] {emb.shape} backend={enc.backend}", flush=True)

    tr, te = train_test_split(np.arange(len(df)), test_size=config.TEST_FRACTION,
                              stratify=df["label"], random_state=args.seed)
    ext = StructuralFeatureExtractor().fit(df.iloc[tr])
    s_tr, s_te = ext.transform(df.iloc[tr]), ext.transform(df.iloc[te])
    X_tr = np.hstack([s_tr, emb[tr]]).astype(np.float32)
    X_te = np.hstack([s_te, emb[te]]).astype(np.float32)
    train_df = df.iloc[tr].reset_index(drop=True)
    test_df = df.iloc[te].reset_index(drop=True)

    # ---- first layer + proxy second layer ----
    model = LLMLogFusion(PipelineConfig(seed=args.seed, prefer_api=False)).fit(
        train_df, X_tr, s_tr)
    res = model.predict(test_df, X_te, s_te)
    idx = np.flatnonzero(res.candidates)
    print(f"[cascade] candidates={len(idx):,} ({len(idx)/len(test_df):.2%}) "
          f"proxy_source={res.source}", flush=True)

    eval_idx = idx
    if args.limit and args.limit < len(idx):
        rng = np.random.default_rng(args.seed)
        keep = []
        for label in (0, 1):
            pool = idx[test_df["label"].to_numpy()[idx] == label]
            k = int(round(args.limit * len(pool) / len(idx)))
            keep.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        eval_idx = np.sort(np.concatenate(keep))
        print(f"[sample] stratified subset={len(eval_idx):,} of {len(idx):,}",
              flush=True)

    cand = test_df.iloc[eval_idx].reset_index(drop=True)
    prompts = [
        build_prompt(str(cand.loc[i, "raw"]), str(cand.loc[i, "platform"]),
                     float(res.scores[eval_idx[i]]),
                     s_te[eval_idx[i]], "full")
        for i in range(len(cand))
    ]
    print(f"[prompt] mean chars={np.mean([len(p) for p in prompts]):.0f}", flush=True)

    # ---- real API calls, written incrementally ----
    calls_path = TABLE_DIR / f"real_llm_calls_{args.tag}.csv"
    fields = ["index", "platform", "label", "stage_true", "score",
              "http_ok", "status", "retryable", "latency_ms",
              "is_anomaly", "stage",
              "confidence", "parse_ok", "parse_error", "raw_stage",
              "finish_reason", "model", "system_fingerprint",
              "prompt_tokens", "completion_tokens", "reasoning_tokens",
              "cached_tokens", "error", "prompt_chars"]
    abort = threading.Event()
    lock = threading.Lock()
    t_wall0 = time.perf_counter()
    done = 0
    with open(calls_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_call_with_retry, prompts[i], args, key, abort): i
                    for i in range(len(prompts))}
            for fut in as_completed(futs):
                i = futs[fut]
                out = fut.result()
                row = {"index": i,
                       "platform": str(cand.loc[i, "platform"]),
                       "label": int(cand.loc[i, "label"]),
                       "stage_true": int(cand.loc[i, "stage"]),
                       "score": float(res.scores[eval_idx[i]]),
                       "prompt_chars": len(prompts[i])}
                row.update(out)
                with lock:
                    writer.writerow(row)
                    fh.flush()
                    done += 1
                    if done % 100 == 0 or done == len(prompts):
                        print(f"[call] {done}/{len(prompts)}", flush=True)
        if abort.is_set():
            print("[call] aborted by fatal API error", flush=True)
    wall_s = time.perf_counter() - t_wall0
    bal1 = _balance(args.base_url, key)

    # ---- analysis ----
    calls = pd.read_csv(calls_path).sort_values("index").reset_index(drop=True)
    n = len(cand)
    aligned = calls.reindex(range(n))
    http_ok = aligned["http_ok"].fillna(0).astype(int).to_numpy()
    flags = aligned["is_anomaly"].fillna(0).astype(int).to_numpy()
    stages = aligned["stage"].fillna(-1).astype(int).to_numpy()
    parse_ok = aligned["parse_ok"].fillna(0).astype(int).to_numpy()
    finish = aligned["finish_reason"].fillna("").astype(str).to_numpy()
    raw_stage = aligned["raw_stage"].fillna("").astype(str).to_numpy()

    y_true = cand["label"].to_numpy(dtype=int)
    truth_stage = cand["stage"].to_numpy(dtype=int)
    proxy_flags = res.y_pred[eval_idx].astype(int)
    proxy_stages = res.stage_pred[eval_idx].astype(int)
    stages_eff = np.where(flags == 1, stages, -1)
    proxy_stages_eff = np.where(proxy_flags == 1, proxy_stages, -1)

    n_http = int(http_ok.sum())
    n_parse_fail = int(((http_ok == 1) & (parse_ok == 0)).sum())
    n_request_fail = int((http_ok == 0).sum())
    n_truncated = int((finish == "length").sum())
    n_invalid_stage = int(((flags == 1) & (stages < 0)).sum())
    n_flagged = int(flags.sum())
    anom = y_true == 1
    out_mask = anom & (flags == 1)
    stage_mismatch_out = int(((stages != truth_stage) & out_mask).sum())
    # An anomaly that was not flagged has no usable stage output, so it counts
    # as an end-to-end stage failure; a flagged anomaly is wrong only if its
    # stage differs from the ground truth.
    stage_mismatch_all = int((((flags == 0) | (stages != truth_stage)) & anom).sum())
    proxy_out_mask = anom & (proxy_flags == 1)
    proxy_mismatch_out = int(((proxy_stages != truth_stage) & proxy_out_mask).sum())

    m_cand = classification_metrics(y_true, flags)
    m_cand_proxy = classification_metrics(y_true, proxy_flags)
    full_real = np.zeros(len(test_df), dtype=int)
    full_real[eval_idx] = flags
    m_full = classification_metrics(test_df["label"].to_numpy(dtype=int), full_real)
    m_full_proxy = classification_metrics(test_df["label"].to_numpy(dtype=int),
                                          res.y_pred)

    lat = aligned.loc[http_ok == 1, "latency_ms"].astype(float).to_numpy()
    prompt_tok = int(aligned["prompt_tokens"].fillna(0).sum())
    comp_tok = int(aligned["completion_tokens"].fillna(0).sum())
    reason_tok = int(aligned["reasoning_tokens"].fillna(0).sum())
    cached_tok = int(aligned["cached_tokens"].fillna(0).sum())
    cost_cny = (bal0 - bal1) if (bal0 is not None and bal1 is not None) else None

    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(flags, proxy_flags))
    b = int(((flags == 1) & (proxy_flags == 0)).sum())
    c = int(((flags == 0) & (proxy_flags == 1)).sum())
    try:
        from scipy.stats import binomtest
        mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else float("nan")
    except Exception:
        mcnemar_p = float("nan")

    sm_real = _stage_summary(truth_stage[anom], stages_eff[anom])
    sm_proxy = _stage_summary(truth_stage[anom], proxy_stages_eff[anom])

    summary = {
        "tag": args.tag,
        "backend": "deepseek-api",
        "model_requested": args.model,
        "model_returned": str(aligned["model"].dropna().iloc[0]) if n_http else "",
        "base_url": args.base_url,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": config.LLM_TEMPERATURE,
        "n_test": int(len(test_df)),
        "n_candidates_total": int(len(idx)),
        "n_candidates_called": int(n),
        "call_rate": float(n / len(test_df)),
        "n_http_ok": n_http,
        "n_parse_fail": n_parse_fail,
        "n_request_fail": n_request_fail,
        "n_truncated": n_truncated,
        "parse_failure_rate_among_http_ok": float(n_parse_fail / max(n_http, 1)),
        "parse_failure_rate_among_called": float(n_parse_fail / max(n, 1)),
        "request_failure_rate": float(n_request_fail / max(n, 1)),
        "n_flagged": n_flagged,
        "n_invalid_stage_outputs": n_invalid_stage,
        "invalid_stage_rate_among_flagged": float(n_invalid_stage / max(n_flagged, 1)),
        "n_true_anomalies_candidates": int(anom.sum()),
        "n_true_anomalies_flagged": int(out_mask.sum()),
        "stage_mismatch_rate_among_outputs": float(stage_mismatch_out / max(out_mask.sum(), 1)),
        "stage_mismatch_rate_among_true_anomalies": float(stage_mismatch_all / max(anom.sum(), 1)),
        "proxy_stage_mismatch_rate_among_outputs": float(proxy_mismatch_out / max(proxy_out_mask.sum(), 1)),
        "stage_weighted_f1_real": sm_real["weighted_f1"],
        "stage_macro_f1_real": sm_real["macro_f1"],
        "stage_accuracy_real": sm_real["accuracy"],
        "stage_weighted_f1_proxy": sm_proxy["weighted_f1"],
        "stage_macro_f1_proxy": sm_proxy["macro_f1"],
        "stage_accuracy_proxy": sm_proxy["accuracy"],
        "candidate_second_layer_real": m_cand,
        "candidate_second_layer_proxy": m_cand_proxy,
        "cascade_full_real": m_full,
        "cascade_full_proxy": m_full_proxy,
        "candidate_f1_ci95_real": _bootstrap_f1(y_true, flags),
        "candidate_f1_ci95_proxy": _bootstrap_f1(y_true, proxy_flags),
        "cascade_f1_ci95_real": _bootstrap_f1(test_df["label"].to_numpy(dtype=int), full_real),
        "cascade_f1_ci95_proxy": _bootstrap_f1(test_df["label"].to_numpy(dtype=int), res.y_pred),
        "agreement_real_proxy": float((flags == proxy_flags).mean()),
        "cohen_kappa_real_proxy": kappa,
        "mcnemar_b_real_only": b,
        "mcnemar_c_proxy_only": c,
        "mcnemar_p": mcnemar_p,
        "latency_ms": {
            "mean": float(lat.mean()) if lat.size else None,
            "std": float(lat.std(ddof=1)) if lat.size > 1 else None,
            "p50": float(np.percentile(lat, 50)) if lat.size else None,
            "p90": float(np.percentile(lat, 90)) if lat.size else None,
            "p95": float(np.percentile(lat, 95)) if lat.size else None,
            "p99": float(np.percentile(lat, 99)) if lat.size else None,
            "max": float(lat.max()) if lat.size else None,
            "sum_s": float(lat.sum() / 1000.0) if lat.size else None,
        },
        "wall_seconds": wall_s,
        "throughput_calls_per_s_concurrent": float(n / wall_s) if wall_s else None,
        "throughput_calls_per_s_sequential_equiv": float(1000.0 / lat.mean()) if lat.size else None,
        "tokens": {
            "prompt": prompt_tok, "completion": comp_tok,
            "reasoning": reason_tok, "cached": cached_tok,
            "total": prompt_tok + comp_tok,
            "mean_prompt_per_call": float(prompt_tok / max(n_http, 1)),
            "mean_completion_per_call": float(comp_tok / max(n_http, 1)),
        },
        "balance_cny": {"before": bal0, "after": bal1,
                        "delta": cost_cny,
                        "per_1000_calls": (cost_cny / n * 1000) if cost_cny is not None else None},
        "error_samples": aligned.loc[aligned["http_ok"] == 0, "error"].dropna().astype(str).head(5).tolist(),
    }
    summary["stage_per_stage_real"] = sm_real["per_stage"]
    summary["stage_per_stage_proxy"] = sm_proxy["per_stage"]

    def _py(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out_json = TABLE_DIR / f"real_llm_summary_{args.tag}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_py),
                        encoding="utf-8")

    # tidy comparison table for the paper
    rows = [
        ("第二层级（候选集）", "真实API", m_cand),
        ("第二层级（候选集）", "离线代理", m_cand_proxy),
        ("级联（全测试集）", "真实API", m_full),
        ("级联（全测试集）", "离线代理", m_full_proxy),
    ]
    table = pd.DataFrame([
        {"setting": s, "backend": b, "precision": m["precision"],
         "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"],
         "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]}
        for s, b, m in rows])
    table.to_csv(TABLE_DIR / f"table_real_vs_proxy_{args.tag}.csv", index=False)

    print("\n[result] candidate second layer (real): "
          f"P={m_cand['precision']:.3f} R={m_cand['recall']:.3f} F1={m_cand['f1']:.3f}", flush=True)
    print("[result] candidate second layer (proxy): "
          f"P={m_cand_proxy['precision']:.3f} R={m_cand_proxy['recall']:.3f} F1={m_cand_proxy['f1']:.3f}", flush=True)
    print("[result] cascade full (real): "
          f"P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}", flush=True)
    print("[result] cascade full (proxy): "
          f"P={m_full_proxy['precision']:.3f} R={m_full_proxy['recall']:.3f} F1={m_full_proxy['f1']:.3f}", flush=True)
    print(f"[result] parse_fail={n_parse_fail} request_fail={n_request_fail} "
          f"stage_mismatch(outputs)={summary['stage_mismatch_rate_among_outputs']:.3f} "
          f"latency_mean={summary['latency_ms']['mean']:.0f}ms", flush=True)
    print(f"[result] tokens prompt={prompt_tok:,} completion={comp_tok:,} "
          f"cost_cny={cost_cny}", flush=True)
    print(f"[write] {out_json} | {calls_path} | "
          f"{TABLE_DIR / f'table_real_vs_proxy_{args.tag}.csv'}", flush=True)


if __name__ == "__main__":
    main()
