"""First-layer filtering baselines compared in Table VI.

Isolation Forest is the paper's method (implemented in :mod:`iforest`).  The
five competing filters (HBOS, LOF, One-Class SVM, AutoEncoder, DBSCAN) are
implemented with scikit-learn / numpy so the same 398-d multimodal features can
be compared under an identical protocol.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import config
from .iforest import IsolationForest


class _BaseFilter:
    name = "base"
    standardize = False

    def __init__(self, contamination: float = config.IF_CONTAMINATION,
                 random_state: int = config.IF_RANDOM_STATE) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.threshold_: float = 0.0

    def _fit_scaler(self, X: np.ndarray) -> None:
        if not self.standardize:
            return
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0

    def _tx(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray) -> "_BaseFilter":
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) >= self.threshold_

    def _set_threshold(self, X: np.ndarray) -> None:
        self.threshold_ = float(np.quantile(self.score_samples(X),
                                            1.0 - self.contamination))


class IsolationForestFilter(_BaseFilter):
    name = "Isolation Forest"

    def fit(self, X: np.ndarray) -> "IsolationForestFilter":
        blocks = config.IF_BLOCK_SIZES if config.IF_BLOCK_BALANCE else None
        self.model_ = IsolationForest(contamination=self.contamination,
                                      random_state=self.random_state,
                                      block_sizes=blocks).fit(X)
        self.threshold_ = self.model_.threshold_
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.model_.score_samples(X)


class HBOSFilter(_BaseFilter):
    """Histogram-based outlier score (sum of negative log bin densities)."""
    name = "HBOS"

    def __init__(self, n_bins: int = 20, **kw) -> None:
        super().__init__(**kw)
        self.n_bins = n_bins

    def fit(self, X: np.ndarray) -> "HBOSFilter":
        self.edges_ = []
        self.logp_ = []
        eps = 1e-12
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.allclose(col.max(), col.min()):
                edges = np.array([col.min() - 0.5, col.max() + 0.5])
                hist = np.array([len(col)], dtype=float)
            else:
                hist, edges = np.histogram(col, bins=self.n_bins)
            prob = (hist + eps) / (hist.sum() + eps * len(hist))
            self.edges_.append(edges)
            self.logp_.append(np.log(prob))
        self._set_threshold(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        total = np.zeros(X.shape[0], dtype=np.float64)
        for j in range(X.shape[1]):
            edges, logp = self.edges_[j], self.logp_[j]
            bins = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                           0, len(logp) - 1)
            total -= logp[bins]
        return total / X.shape[1]


class LOFFilter(_BaseFilter):
    name = "LOF"
    standardize = True

    def __init__(self, n_neighbors: int = 20, **kw) -> None:
        super().__init__(**kw)
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray) -> "LOFFilter":
        from sklearn.neighbors import LocalOutlierFactor
        self._fit_scaler(X)
        Xs = self._tx(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(Xs), size=min(len(Xs), 6000), replace=False)
        self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                         novelty=True, n_jobs=-1)
        self.model_.fit(Xs[idx])
        self._set_threshold(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(self._tx(X))


class OneClassSVMFilter(_BaseFilter):
    name = "One-Class SVM"
    standardize = True

    def fit(self, X: np.ndarray) -> "OneClassSVMFilter":
        from sklearn.svm import OneClassSVM
        self._fit_scaler(X)
        Xs = self._tx(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(Xs), size=min(len(Xs), 4000), replace=False)
        self.model_ = OneClassSVM(kernel="rbf", gamma="scale", nu=self.contamination)
        self.model_.fit(Xs[idx])
        self._set_threshold(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.decision_function(self._tx(X))


class AutoEncoderFilter(_BaseFilter):
    name = "AutoEncoder"
    standardize = True

    def __init__(self, hidden=(64, 16), epochs: int = 30, lr: float = 1e-3, **kw) -> None:
        super().__init__(**kw)
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.ndarray) -> "AutoEncoderFilter":
        self._fit_scaler(X)
        Xs = self._tx(X).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        n, d = Xs.shape
        dims = [d, *self.hidden, d]
        self.weights_ = [rng.normal(0, np.sqrt(2 / dims[i]),
                                    size=(dims[i], dims[i + 1])).astype(np.float32)
                         for i in range(len(dims) - 1)]
        self.biases_ = [np.zeros(dims[i + 1], dtype=np.float32)
                        for i in range(len(dims) - 1)]
        params = self.weights_ + self.biases_
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch = 256
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                xb = Xs[idx]
                acts = [xb]
                for layer, (w, b) in enumerate(zip(self.weights_, self.biases_)):
                    pre = acts[-1] @ w + b
                    acts.append(np.tanh(pre) if layer < len(self.weights_) - 1 else pre)
                grad = 2.0 * (acts[-1] - xb) / len(idx)
                grads_w, grads_b = [None] * len(self.weights_), [None] * len(self.biases_)
                for layer in range(len(self.weights_) - 1, -1, -1):
                    grads_w[layer] = acts[layer].T @ grad
                    grads_b[layer] = grad.sum(axis=0)
                    if layer > 0:
                        grad = (grad @ self.weights_[layer].T) * (1.0 - acts[layer] ** 2)
                step += 1
                for p, g, mi, vi in zip(params, grads_w + grads_b, m, v):
                    mi *= b1
                    mi += (1 - b1) * g
                    vi *= b2
                    vi += (1 - b2) * g * g
                    p -= self.lr * (mi / (1 - b1 ** step)) / (
                        np.sqrt(vi / (1 - b2 ** step)) + eps)
        self._set_threshold(X)
        return self

    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        h = self._tx(X)
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            h = h @ w + b
            if i < len(self.weights_) - 1:
                h = np.tanh(h)
        return h

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return np.mean((self._reconstruct(X) - self._tx(X)) ** 2, axis=1)


class DBSCANFilter(_BaseFilter):
    name = "DBSCAN"
    standardize = True

    def __init__(self, min_samples: int = 10, **kw) -> None:
        super().__init__(**kw)
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> "DBSCANFilter":
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        self._fit_scaler(X)
        Xs = self._tx(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(Xs), size=min(len(Xs), 6000), replace=False)
        sub = Xs[idx]
        probe = sub[rng.choice(len(sub), size=min(len(sub), 1500), replace=False)]
        nn = NearestNeighbors(n_neighbors=self.min_samples).fit(sub)
        dist, _ = nn.kneighbors(probe)
        eps = float(np.quantile(dist[:, -1], 0.90))
        model = DBSCAN(eps=max(eps, 1e-6), min_samples=self.min_samples).fit(sub)
        core = sub[model.labels_ != -1]
        self.core_ = core if len(core) else sub
        self.model_ = NearestNeighbors(n_neighbors=1).fit(self.core_)
        self._set_threshold(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self.model_.kneighbors(self._tx(X), n_neighbors=1)
        return dist[:, 0]


def build_filter(name: str, contamination: float = config.IF_CONTAMINATION,
                 random_state: int = config.IF_RANDOM_STATE) -> _BaseFilter:
    registry: Dict[str, type] = {
        "Isolation Forest": IsolationForestFilter,
        "HBOS": HBOSFilter,
        "LOF": LOFFilter,
        "One-Class SVM": OneClassSVMFilter,
        "AutoEncoder": AutoEncoderFilter,
        "DBSCAN": DBSCANFilter,
    }
    return registry[name](contamination=contamination, random_state=random_state)
