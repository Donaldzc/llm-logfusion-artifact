"""14-dimensional structural feature extraction (Table II).

Dimensions: timestamp validity (1) + log level one-hot ERROR/WARN/INFO (3)
+ platform one-hot (9) + normalised content length (1).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import config

TS_FORMATS = {
    "HDFS": "%y%m%d %H%M%S",
    "BGL": "%Y.%m.%d-%H.%M.%S",
    "OpenStack": "%Y-%m-%d %H:%M:%S",
    "Windows": "%Y-%m-%d %H:%M:%S",
    "Apache": "%d/%b/%Y:%H:%M:%S +0000",
    "Firewall": "%b %d %H:%M:%S",
    "Database": "%Y-%m-%dT%H:%M:%S.%fZ",
    "SSH": "%b %d %H:%M:%S",
    "DNS": "%d-%b-%Y %H:%M:%S.%f",
}

_PLATFORM_INDEX = {p: i for i, p in enumerate(config.PLATFORMS)}
_RE_ERR = re.compile(r"\b(ERROR|FATAL|SEVERE|CRITICAL)\b")
_RE_WARN = re.compile(r"\bWARN(ING)?\b")
_RE_INFO = re.compile(r"\b(INFO|NOTICE)\b")
_RE_WIN_LEVEL = re.compile(r",(Error|Warning|Info),")
_RE_APACHE_STATUS = re.compile(r'"\s+(\d{3})\s+')
_RE_PFSENSE_SEV = re.compile(r"filterlog:\s*(\d),")
_RE_SSH_ERR = re.compile(r"error:|kex_exchange_identification", re.I)
_RE_SSH_WARN = re.compile(r"Invalid user|Failed password|message repeated|preauth", re.I)
_RE_SSH_INFO = re.compile(r"Accepted |session (opened|closed)|Received disconnect", re.I)
_RE_DB_ERR = re.compile(r"ERROR|Access denied|duplicate key", re.I)
_RE_DB_WARN = re.compile(r"slow query|WARN", re.I)
_RE_DNS_ERR = re.compile(r"SERVFAIL")
_RE_DNS_WARN = re.compile(r"NXDOMAIN|REFUSED")


def is_valid_timestamp(ts: str, platform: str) -> int:
    if not isinstance(ts, str) or not ts or ts in {"N/A", "-"}:
        return 0
    fmt = TS_FORMATS.get(platform)
    if fmt is None:
        return 0
    try:
        datetime.strptime(ts, fmt)
        return 1
    except ValueError:
        return 0


def derive_level(raw: str, platform: str) -> str:
    """Rule-based log-level extraction used by the structural encoder."""
    if platform == "Windows":
        m = _RE_WIN_LEVEL.search(raw)
        if m:
            return {"Error": "ERROR", "Warning": "WARN", "Info": "INFO"}[m.group(1)]
        return "OTHER"
    if platform == "Apache":
        m = _RE_APACHE_STATUS.search(raw)
        if m:
            code = int(m.group(1))
            return "ERROR" if code >= 500 else ("WARN" if code >= 400 else "INFO")
        return "OTHER"
    if platform == "Firewall":
        m = _RE_PFSENSE_SEV.search(raw)
        if m:
            sev = int(m.group(1))
            return "ERROR" if sev <= 3 else ("WARN" if sev == 4 else "INFO")
        return "OTHER"
    if platform == "SSH":
        if _RE_SSH_ERR.search(raw):
            return "ERROR"
        if _RE_SSH_WARN.search(raw):
            return "WARN"
        if _RE_SSH_INFO.search(raw):
            return "INFO"
        return "OTHER"
    if platform == "Database":
        if _RE_DB_ERR.search(raw):
            return "ERROR"
        if _RE_DB_WARN.search(raw):
            return "WARN"
        if "Query" in raw or "Connect" in raw:
            return "INFO"
        return "OTHER"
    if platform == "DNS":
        if _RE_DNS_ERR.search(raw):
            return "ERROR"
        if _RE_DNS_WARN.search(raw):
            return "WARN"
        if "query:" in raw or "reply:" in raw:
            return "INFO"
        return "OTHER"
    if _RE_ERR.search(raw):
        return "ERROR"
    if _RE_WARN.search(raw):
        return "WARN"
    if _RE_INFO.search(raw):
        return "INFO"
    return "OTHER"


class StructuralFeatureExtractor:
    """Fits only the content-length normalisation, then emits (n, 14)."""

    _LEVEL_INDEX = {"ERROR": 0, "WARN": 1, "INFO": 2}

    def __init__(self) -> None:
        self.len_min_ = 0.0
        self.len_max_ = 1.0

    def fit(self, df: pd.DataFrame) -> "StructuralFeatureExtractor":
        lengths = df["raw"].str.len().to_numpy(dtype=float)
        self.len_min_ = float(lengths.min())
        self.len_max_ = float(max(lengths.max(), self.len_min_ + 1.0))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        out = np.zeros((n, config.N_STRUCT), dtype=np.float32)
        length_span = self.len_max_ - self.len_min_
        for i, (raw, ts, platform) in enumerate(
                zip(df["raw"].tolist(), df["timestamp"].tolist(), df["platform"].tolist())):
            out[i, 0] = is_valid_timestamp(ts, platform)
            lvl = derive_level(raw, platform)
            if lvl in self._LEVEL_INDEX:
                out[i, 1 + self._LEVEL_INDEX[lvl]] = 1.0
            out[i, 4 + _PLATFORM_INDEX.get(platform, 0)] = 1.0
            norm_len = (len(raw) - self.len_min_) / length_span
            out[i, 13] = float(np.clip(norm_len, 0.0, 1.0))
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
