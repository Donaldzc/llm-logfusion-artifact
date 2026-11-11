"""End-to-end LLM-LogFusion pipeline and its ablations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .filters import IsolationForestFilter, _BaseFilter, build_filter
from .llm_layer import DeepSeekClient, LLMBatchResult, OfflineSurrogateLLM
from .projection import AttackStageProjection


@dataclass
class PipelineConfig:
    mode: str = "cascade"          # cascade | filter_only | llm_only | classifier
    transfer: str = "full"         # none | score | features | full
    contamination: float = config.IF_CONTAMINATION
    seed: int = 0
    prefer_api: bool = True
    filter_name: str = "Isolation Forest"
    filter_features: str = "fusion"   # fusion | struct | semantic (ablation)


@dataclass
class PredictionResult:
    y_pred: np.ndarray
    stage_pred: np.ndarray
    scores: np.ndarray
    candidates: np.ndarray
    confidence: np.ndarray
    source: str
    n_llm_calls: int
    filter_seconds: float
    llm_seconds: float
    total_seconds: float
    timing: Dict[str, float] = field(default_factory=dict)


class LLMLogFusion:
    def __init__(self, cfg: PipelineConfig = PipelineConfig(),
                 filter_obj: Optional[_BaseFilter] = None,
                 projection: Optional[AttackStageProjection] = None) -> None:
        self.cfg = cfg
        self._filter = filter_obj
        self.projection = projection
        self.llm = None
        self.surrogate = None
        self.classifier = None
        self.source = "offline-surrogate"

    # ------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, X_fusion: np.ndarray,
            X_struct: np.ndarray) -> "LLMLogFusion":
        y = df["label"].to_numpy(dtype=int)

        if self.cfg.mode in ("cascade", "filter_only"):
            self._filter = self._filter or build_filter(
                self.cfg.filter_name, contamination=self.cfg.contamination,
                random_state=self.cfg.seed)
            self._filter.fit(self._filter_matrix(X_fusion, X_struct))

        anom = y == 1
        if self.projection is None:
            self.projection = AttackStageProjection(seed=self.cfg.seed)
            stage_labels = df.loc[anom, "stage"].to_numpy(dtype=int)
            if len(stage_labels) >= config.N_STAGES:
                self.projection.fit(X_fusion[anom], stage_labels)

        if self.cfg.mode == "classifier":
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(max_iter=500, C=1.0,
                                                 class_weight="balanced",
                                                 random_state=self.cfg.seed)
            self.classifier.fit(X_fusion, y)
            return self

        if self.cfg.mode in ("cascade", "llm_only"):
            self.surrogate = OfflineSurrogateLLM(self.projection, seed=self.cfg.seed)
            if_scores = (self._filter.score_samples(
                self._filter_matrix(X_fusion, X_struct))
                if self._filter is not None else np.zeros(len(df)))
            self.surrogate.fit(df["raw"].tolist(), y, X_struct, if_scores)
            if self.cfg.prefer_api and DeepSeekClient().api_key:
                self.llm = DeepSeekClient()
                self.source = "deepseek-api"
        return self

    # --------------------------------------------------------------- helpers
    def _filter_matrix(self, X_fusion: np.ndarray,
                       X_struct: np.ndarray) -> np.ndarray:
        if self.cfg.filter_features == "struct":
            return X_struct
        if self.cfg.filter_features == "semantic":
            return X_fusion[:, config.N_STRUCT:]
        return X_fusion

    # --------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame, X_fusion: np.ndarray,
                X_struct: np.ndarray) -> PredictionResult:
        n = len(df)
        t_start = time.perf_counter()

        if self.cfg.mode == "classifier":
            prob = self.classifier.predict_proba(X_fusion)[:, 1]
            y_pred = (prob >= 0.5).astype(int)
            stage_pred = np.where(y_pred == 1,
                                  self.projection.predict(X_fusion), -1)
            elapsed = time.perf_counter() - t_start
            return PredictionResult(y_pred, stage_pred, prob,
                                    y_pred.astype(bool), np.abs(2 * prob - 1),
                                    "logistic-398d", 0, 0.0, 0.0, elapsed)

        t0 = time.perf_counter()
        scores = (self._filter.score_samples(self._filter_matrix(X_fusion, X_struct))
                  if self._filter is not None else np.zeros(n))
        if self.cfg.mode == "llm_only":
            candidates = np.ones(n, dtype=bool)
        else:
            candidates = scores >= self._filter.threshold_
        filter_seconds = time.perf_counter() - t0

        y_pred = np.zeros(n, dtype=int)
        stage_pred = np.full(n, -1, dtype=int)
        confidence = np.zeros(n, dtype=float)
        n_calls = 0
        llm_seconds = 0.0

        if self.cfg.mode == "filter_only":
            y_pred[candidates] = 1
            stage_pred[candidates] = self.projection.predict(X_fusion[candidates])
            confidence[candidates] = 1.0
        else:
            idx = np.flatnonzero(candidates)
            n_calls = len(idx)
            if n_calls:
                t1 = time.perf_counter()
                if self.llm is not None:
                    res: LLMBatchResult = self.llm.analyze(
                        df.iloc[idx].reset_index(drop=True), X_struct[idx],
                        X_fusion[idx], scores[idx], self.cfg.transfer,
                        fallback=self.surrogate)
                else:
                    res = self.surrogate.analyze(
                        df.iloc[idx].reset_index(drop=True), X_struct[idx],
                        X_fusion[idx], scores[idx], self.cfg.transfer)
                llm_seconds = time.perf_counter() - t1
                y_pred[idx] = res.is_anomaly.astype(int)
                stage_pred[idx] = np.where(res.is_anomaly, res.stage, -1)
                confidence[idx] = res.confidence
        total = time.perf_counter() - t_start
        return PredictionResult(y_pred, stage_pred, scores, candidates,
                                confidence, self.source, n_calls,
                                filter_seconds, llm_seconds, total)
