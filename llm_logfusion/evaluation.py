"""Evaluation metrics, confidence intervals and significance tests."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import stats

from . import config


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        half = float(stats.t.ppf(0.5 + confidence / 2, arr.size - 1) * std / np.sqrt(arr.size))
    else:
        half = 0.0
    return {"mean": mean, "std": std, "ci": half}


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2 or np.allclose(a, b):
        return {"t": float("nan"), "p": float("nan")}
    res = stats.ttest_rel(a, b)
    return {"t": float(res.statistic), "p": float(res.pvalue)}


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  n_stages: int = config.N_STAGES) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    supports = []
    for k in range(n_stages):
        m = classification_metrics((y_true == k).astype(int), (y_pred == k).astype(int))
        rows.append({**m, "stage": config.STAGES[k],
                     "support": int((y_true == k).sum())})
        supports.append(int((y_true == k).sum()))
    macro = {key: float(np.mean([r[key] for r in rows]))
             for key in ("precision", "recall", "f1")}
    total = max(1, sum(supports))
    weighted = {key: float(sum(r[key] * s for r, s in zip(rows, supports)) / total)
                for key in ("precision", "recall", "f1")}
    accuracy = float((y_true == y_pred).mean())
    return {"rows": rows, "macro": macro, "weighted": weighted,
            "accuracy": accuracy}


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"
