"""Second-stage LLM analysis.

Two interchangeable backends produce the structured verdict required by the
paper -- ``{"is_anomaly": bool, "attack_stage": <one of 12>, "confidence": 0-1}``:

* :class:`DeepSeekClient` calls the DeepSeek chat-completions API with the
  prompt template of Section IV-A (temperature 0, max_tokens 512, top_p 1).
  It is used automatically when ``DEEPSEEK_API_KEY`` is set.
* :class:`OfflineSurrogateLLM` is a deterministic stand-in that reproduces the
  *information flow* of the second stage (raw log text, first-layer anomaly
  score, multimodal features) with a stacked text/structural/score model.  It
  exists so the whole pipeline is reproducible without network access; the
  backend actually used is always recorded in the results.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .projection import AttackStageProjection

PROMPT_TEMPLATE = """You are a senior SOC analyst. Analyse the log entry below and
return one JSON object only, with exactly these keys:
  "is_anomaly": true | false,
  "attack_stage": one of {stages},
  "confidence": a float between 0 and 1,
  "reason": a short justification (<= 30 words).

Platform: {platform}
Log: {log}
{extra}"""

STAGE_LIST = "; ".join(f"{i + 1}. {s}" for i, s in enumerate(config.STAGES))
_TRANSFER_LABEL = {
    "none": "",
    "score": "First-layer anomaly score: {score:.4f}\n",
    "features": "Structural features (timestamp-valid, ERROR, WARN, INFO, "
                "platform one-hot x9, norm-length): {features}\n",
    "full": "First-layer anomaly score: {score:.4f}\n"
            "Structural features (timestamp-valid, ERROR, WARN, INFO, "
            "platform one-hot x9, norm-length): {features}\n",
}


@dataclass
class LLMBatchResult:
    is_anomaly: np.ndarray
    stage: np.ndarray
    confidence: np.ndarray
    source: str


def build_prompt(raw: str, platform: str, score: float, struct_row: np.ndarray,
                 mode: str) -> str:
    extra = _TRANSFER_LABEL.get(mode, "").format(
        score=score,
        features="[" + ", ".join(f"{v:.2f}" for v in struct_row) + "]")
    return PROMPT_TEMPLATE.format(stages=STAGE_LIST, platform=platform,
                                  log=raw[:1500], extra=extra)


def _canonical_stage(name: str) -> int:
    """Map a model-produced stage label to its ATT&CK index.

    The prompt enumerates stages as ``1. Initial Access; 2. Execution; ...``.
    Models frequently echo the numbered form (``2. Execution``) instead of the
    bare name, so leading numbering and punctuation are stripped before the
    case-insensitive lookup.  An unparsable label returns -1 and is counted as
    a stage-attribution error, never silently coerced.
    """
    s = str(name).strip().strip("\"'")
    s = re.sub(r"^\s*\d+\s*[\.\)\-:：、]?\s*", "", s).strip()
    for i, canonical in enumerate(config.STAGES):
        if s.lower() == canonical.lower():
            return i
    return -1


def _parse_verdict(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("no JSON object in response")
    obj = json.loads(match.group(0))
    stage_idx = _canonical_stage(str(obj.get("attack_stage", "")))
    return {
        "is_anomaly": bool(obj.get("is_anomaly", False)),
        "stage": stage_idx,
        "confidence": float(np.clip(float(obj.get("confidence", 0.5)), 0.0, 1.0)),
    }


class DeepSeekClient:
    """Thin urllib client -- no third-party dependency required."""

    def __init__(self, api_key: str | None = None, model: str = config.LLM_MODEL,
                 base_url: str = config.LLM_BASE_URL, timeout: int = 60) -> None:
        self.api_key = api_key or os.environ.get(config.LLM_API_KEY_ENV, "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, prompt: str) -> Dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "top_p": config.LLM_TOP_P,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return _parse_verdict(body["choices"][0]["message"]["content"])

    def analyze(self, df: pd.DataFrame, X_struct: np.ndarray,
                X_fusion: np.ndarray, if_scores: np.ndarray, mode: str,
                fallback: "OfflineSurrogateLLM | None" = None) -> LLMBatchResult:
        n = len(df)
        flags = np.zeros(n, dtype=bool)
        stages = np.full(n, -1, dtype=int)
        conf = np.zeros(n, dtype=float)
        failed = []
        for i, (raw, platform) in enumerate(zip(df["raw"].tolist(),
                                                df["platform"].tolist())):
            prompt = build_prompt(raw, platform, float(if_scores[i]),
                                  X_struct[i], mode)
            try:
                verdict = self._call(prompt)
                flags[i] = verdict["is_anomaly"]
                stages[i] = verdict["stage"]
                conf[i] = verdict["confidence"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                    ValueError, json.JSONDecodeError, TimeoutError):
                failed.append(i)
        if failed and fallback is not None:
            sub = fallback.analyze(df.iloc[failed], X_struct[failed],
                                   X_fusion[failed], if_scores[failed], mode)
            flags[failed] = sub.is_anomaly
            stages[failed] = sub.stage
            conf[failed] = sub.confidence
        return LLMBatchResult(flags, stages, conf, "deepseek-api")


class OfflineSurrogateLLM:
    """Deterministic stand-in for the LLM second stage.

    It consumes exactly the information the paper transfers across layers and
    stacks three evidence sources with a logistic meta-model:
      * semantic evidence from the raw log text (TF-IDF logistic scorer),
      * the Isolation-Forest anomaly score,
      * the 14-d structural feature vector.
    Stage attribution uses the trainable 398->12 projection layer.
    """

    def __init__(self, projection: AttackStageProjection, seed: int = 0) -> None:
        self.projection = projection
        self.seed = seed
        self._meta: Dict[str, object] = {}
        self._if_mean = 0.0
        self._if_std = 1.0

    # ------------------------------------------------------------------ train
    @staticmethod
    def _oof_logits(model_factory, X, y, folds: int = 3, seed: int = 0,
                    is_text: bool = False):
        from sklearn.model_selection import StratifiedKFold
        X = list(X) if is_text else X
        oof = np.zeros(len(y), dtype=np.float64)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(len(y)), y):
            model = model_factory()
            x_tr = [X[i] for i in tr] if is_text else X[tr]
            x_te = [X[i] for i in te] if is_text else X[te]
            model.fit(x_tr, y[tr])
            p = model.predict_proba(x_te)[:, 1]
            oof[te] = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                             np.clip(1 - p, 1e-6, 1 - 1e-6))
        return oof

    def _text_pipeline(self, seed: int):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                            max_features=40000, sublinear_tf=True,
                            token_pattern=r"[A-Za-z0-9@_.$:/\-]+"),
            LogisticRegression(max_iter=400, C=2.0, class_weight="balanced",
                               random_state=seed, solver="liblinear"))

    def fit(self, texts: Sequence[str], labels: np.ndarray, X_struct: np.ndarray,
            if_scores: np.ndarray) -> "OfflineSurrogateLLM":
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(labels, dtype=int)
        self._if_mean = float(np.mean(if_scores))
        self._if_std = float(np.std(if_scores) or 1.0)
        if_z = (if_scores - self._if_mean) / self._if_std

        self._text_model = self._text_pipeline(self.seed).fit(texts, y)
        text_oof = self._oof_logits(lambda: self._text_pipeline(self.seed),
                                    texts, y, seed=self.seed, is_text=True)

        struct_oof = self._oof_logits(
            lambda: LogisticRegression(max_iter=400, C=1.0,
                                       class_weight="balanced",
                                       random_state=self.seed),
            X_struct, y, seed=self.seed)

        columns = {
            "text": text_oof.reshape(-1, 1),
            "struct": struct_oof.reshape(-1, 1),
            "if": if_z.reshape(-1, 1),
        }
        combos = {
            "none": ["text"],
            "score": ["text", "if"],
            "features": ["text", "struct"],
            "full": ["text", "struct", "if"],
        }
        self._meta = {}
        for mode, keys in combos.items():
            Xm = np.hstack([columns[k] for k in keys])
            meta = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                      random_state=self.seed, solver="liblinear")
            meta.fit(Xm, y)
            self._meta[mode] = (meta, keys)
        self.struct_model_ = LogisticRegression(max_iter=400, C=1.0,
                                                class_weight="balanced",
                                                random_state=self.seed)
        self.struct_model_.fit(X_struct, y)
        return self

    # ---------------------------------------------------------------- predict
    def _components(self, texts, X_struct, if_scores):
        text_p = self._text_model.predict_proba(list(texts))[:, 1]
        text_logit = np.log(np.clip(text_p, 1e-6, 1 - 1e-6) /
                            np.clip(1 - text_p, 1e-6, 1 - 1e-6))
        struct_p = self.struct_model_.predict_proba(X_struct)[:, 1]
        struct_logit = np.log(np.clip(struct_p, 1e-6, 1 - 1e-6) /
                              np.clip(1 - struct_p, 1e-6, 1 - 1e-6))
        if_z = (if_scores - self._if_mean) / self._if_std
        return {"text": text_logit.reshape(-1, 1),
                "struct": struct_logit.reshape(-1, 1),
                "if": if_z.reshape(-1, 1)}

    def analyze(self, df: pd.DataFrame, X_struct: np.ndarray,
                X_fusion: np.ndarray, if_scores: np.ndarray,
                mode: str = "full") -> LLMBatchResult:
        comps = self._components(df["raw"].tolist(), X_struct, if_scores)
        meta, keys = self._meta[mode]
        Xm = np.hstack([comps[k] for k in keys])
        prob = meta.predict_proba(Xm)[:, 1]
        flags = prob >= 0.5
        stage_probs = self.projection.predict_proba(X_fusion)
        stages = np.where(flags, stage_probs.argmax(axis=1), -1)
        conf = np.abs(2 * prob - 1)
        return LLMBatchResult(flags, stages, conf, "offline-surrogate")


def make_llm(projection: AttackStageProjection, seed: int = 0,
             prefer_api: bool = True):
    """Return (backend, name).  API backend falls back to the surrogate."""
    if prefer_api and os.environ.get(config.LLM_API_KEY_ENV):
        return DeepSeekClient(), "deepseek-api"
    return None, "offline-surrogate"
