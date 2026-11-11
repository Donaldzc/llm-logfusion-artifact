"""Trainable 398-d -> 12-d MITRE ATT&CK projection layer (Section III-C-3).

A single linear projection trained with cross-entropy (softmax regression) on
labelled (log, attack-stage) pairs.  Training uses mini-batch Adam so the layer
is fully self-contained (numpy only) and easy to inspect.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config


class AttackStageProjection:
    def __init__(self, n_features: int = config.N_FUSION,
                 n_stages: int = config.N_STAGES,
                 epochs: int = 300, lr: float = 1e-3, l2: float = 1e-4,
                 batch_size: int = 256, seed: int = 0) -> None:
        self.n_features = n_features
        self.n_stages = n_stages
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.seed = seed
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------ train
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackStageProjection":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Xs = self._standardise(X)
        rng = np.random.default_rng(self.seed)
        self.W = (rng.normal(0, 0.01, size=(self.n_features, self.n_stages))).astype(np.float32)
        self.b = np.zeros(self.n_stages, dtype=np.float32)
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = len(Xs)
        step = 0
        self.history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb, yb = Xs[idx], y[idx]
                logits = xb @ self.W + self.b
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                prob = exp / exp.sum(axis=1, keepdims=True)
                loss = -np.log(prob[np.arange(len(yb)), yb] + 1e-12).mean()
                epoch_loss += loss * len(yb)
                grad_logits = prob
                grad_logits[np.arange(len(yb)), yb] -= 1.0
                grad_logits /= len(yb)
                gW = xb.T @ grad_logits + self.l2 * self.W
                gb = grad_logits.sum(axis=0)
                step += 1
                for p, g, m_, v_ in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                    m_ *= b1
                    m_ += (1 - b1) * g
                    v_ *= b2
                    v_ += (1 - b2) * g * g
                    p -= self.lr * (m_ / (1 - b1 ** step)) / (
                        np.sqrt(v_ / (1 - b2 ** step)) + eps)
            self.history_.append(float(epoch_loss / n))
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardise(np.asarray(X, dtype=np.float32))
        logits = Xs @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
