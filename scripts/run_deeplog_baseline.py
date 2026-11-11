"""DeepLog-style baseline: log-key frequency + transition anomaly scoring.

The reviewer asked for a runnable classic deep baseline under the same split and
protocol.  A full LSTM DeepLog needs a GPU and per-session training; here we
implement the *log-key* part of DeepLog that the review explicitly allows
("日志键序列预测或模板频次异常打分"):

  1. every log line is parsed into a template by masking parameters
     (numbers, hex ids, IPs, quoted payloads) -- a Drain-style log key;
  2. a unigram template model and a per-platform bigram transition model are
     estimated on *normal* training logs only (DeepLog is trained on normal
     data);
  3. each test log is scored by the negative log-likelihood of its key given
     the preceding key in the same platform, with backoff to the unigram model;
  4. the decision threshold is the 90th percentile of the training-normal
     scores, i.e. the same 10 % budget as the Isolation Forest first layer.

Outputs
-------
results/tables/table_deeplog.csv   per-seed metrics
results/tables/deeplog_summary.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".pylibs"))

from llm_logfusion import config                                   # noqa: E402
from llm_logfusion.evaluation import classification_metrics        # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT_CSV = config.TABLE_DIR / "table_deeplog.csv"
OUT_JSON = config.TABLE_DIR / "deeplog_summary.json"
BUDGET = config.IF_CONTAMINATION          # 0.10 candidate budget

_RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_RE_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_RE_LONGHEX = re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b")
_RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_QUOTED = re.compile(r'"[^"]*"')
_RE_PATH = re.compile(r"(?:/[A-Za-z0-9_.\-]+){2,}")
_RE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_ALNUM_ID = re.compile(r"\b(?=[A-Za-z0-9_.:/\-]*\d)[A-Za-z0-9_.:/\-]{2,}\b")
_RE_WS = re.compile(r"\s+")


def log_key(raw: str) -> str:
    """Mask volatile parameters so that template instances collapse.

    Drain-style log keys: an alphanumeric token that contains at least one digit
    (node id, pid, thread, interface, block id, ...) is treated as a variable.
    """
    s = _RE_QUOTED.sub("<STR>", raw)
    s = _RE_UUID.sub("<UUID>", s)
    s = _RE_IP.sub("<IP>", s)
    s = _RE_PATH.sub("<PATH>", s)
    s = _RE_HEX.sub("<HEX>", s)
    s = _RE_LONGHEX.sub("<HEX>", s)
    s = _RE_NUM.sub("<NUM>", s)
    s = _RE_ALNUM_ID.sub("<VAR>", s)
    return _RE_WS.sub(" ", s).strip()


class LogKeyModel:
    """Unigram + per-platform bigram model over log keys (add-alpha smoothing)."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.unigram: dict[str, int] = {}
        self.bigram: dict[str, dict[str, int]] = {}
        self.n_tokens = 0
        self.vocab_size = 0

    def fit(self, frame: pd.DataFrame) -> "LogKeyModel":
        """frame: normal training logs with columns platform/timestamp/key."""
        self.unigram = frame["key"].value_counts().to_dict()
        self.n_tokens = int(len(frame))
        self.vocab_size = max(len(self.unigram), 1)
        self.bigram = {}
        for _, g in frame.sort_values(["platform", "order"]).groupby("platform"):
            keys = g["key"].tolist()
            for prev, cur in zip(keys[:-1], keys[1:]):
                self.bigram.setdefault(prev, {})
                self.bigram[prev][cur] = self.bigram[prev].get(cur, 0) + 1
        return self

    def _unigram_logprob(self, key: str) -> float:
        p = (self.unigram.get(key, 0) + self.alpha) / (
            self.n_tokens + self.alpha * self.vocab_size)
        return float(np.log(p))

    def _bigram_logprob(self, prev: str | None, key: str) -> float:
        if prev is None or prev not in self.bigram:
            return self._unigram_logprob(key)
        counts = self.bigram[prev]
        total = sum(counts.values())
        p = (counts.get(key, 0) + self.alpha) / (total + self.alpha * self.vocab_size)
        return float(np.log(p))

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Negative log-likelihood of each key given its predecessor."""
        out = np.zeros(len(frame), dtype=float)
        pos = {idx: i for i, idx in enumerate(frame.index)}
        for _, g in frame.sort_values(["platform", "order"]).groupby("platform"):
            prev = None
            for idx, key in zip(g.index, g["key"].tolist()):
                out[pos[idx]] = -self._bigram_logprob(prev, key)
                prev = key
        return out


def main() -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(config.DATA_DIR / "extended_logs.csv")
    df["key"] = [log_key(r) for r in df["raw"].tolist()]
    df["order"] = np.arange(len(df))
    print(f"[data] {len(df):,} logs, {df['key'].nunique()} distinct log keys", flush=True)

    rows = []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=config.TEST_FRACTION,
                                  stratify=df["label"], random_state=seed)
        train_df = df.iloc[tr]
        normal_train = train_df[train_df["label"] == 0]
        # Threshold calibration needs normal logs the model has *not* seen:
        # 20 % of the training-normal logs are held out for that purpose.
        cal_idx, fit_idx = train_test_split(np.arange(len(normal_train)),
                                            test_size=0.8, random_state=seed)
        cal_df = normal_train.iloc[cal_idx]
        fit_df = normal_train.iloc[fit_idx]
        test_df = df.iloc[te].reset_index(drop=True)
        y = test_df["label"].to_numpy(dtype=int)

        model = LogKeyModel().fit(fit_df)
        t_fit = time.perf_counter() - t0
        t1 = time.perf_counter()
        cal_scores = model.score(cal_df)
        test_scores = model.score(test_df)
        score_s = time.perf_counter() - t1

        thr = float(np.quantile(cal_scores, 1.0 - BUDGET))
        pred = (test_scores >= thr).astype(int)
        m = classification_metrics(y, pred)
        m.update({"seed": seed, "threshold": thr,
                  "flag_rate": float(pred.mean()),
                  "fit_seconds": t_fit, "score_seconds": score_s,
                  "throughput": len(test_df) / max(score_s, 1e-9)})
        rows.append(m)
        print(f"[deeplog] seed={seed} F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} FPR={m['fpr']:.3f} flag={m['flag_rate']:.3f}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    summ = {k: {"mean": float(out[k].mean()), "std": float(out[k].std(ddof=1))}
            for k in ("precision", "recall", "f1", "fpr", "flag_rate", "throughput")}
    print(pd.DataFrame(summ).round(4).to_string(), flush=True)
    OUT_JSON.write_text(json.dumps({
        "method": "DeepLog-style log-key baseline (frequency + transition, "
                  "normal-only training, 10% budget)",
        "n_keys": int(df["key"].nunique()),
        "summary": summ,
        "per_seed": out.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
