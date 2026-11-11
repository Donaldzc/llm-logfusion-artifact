"""Nystromformer-style linearised attention (Eq. 3).

    Attention(Q, K, V) ~= Q K_S^T (K_S K_S^T)^-1 K_S V

``K_S`` contains ``k`` landmark rows sampled uniformly without replacement from
the key matrix; a regularisation term eps = 1e-6 is added before inversion.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from . import config


def nystrom_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                      landmarks: int = config.N_LANDMARKS,
                      eps: float = config.ATTN_EPS,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Linearised attention with O(n k d) complexity.

    For landmark index set S:
        out = Q K_S^T (Q_S K_S^T + eps I)^-1 (Q_S V)
    which is the Nystrom approximation A ~= F A_S^-1 G with F = Q K_S^T,
    A_S = Q_S K_S^T and G = Q_S K^T.  Without the softmax the value term
    G V collapses to Q_S V, giving exactly the expression in Eq. (3).

    Args:
        Q, K: (n, d) query / key matrices.
        V: (n, d_v) value matrix.
        landmarks: number of landmark rows k.
    Returns:
        (n, d_v) context matrix.
    """
    n = K.shape[0]
    if n == 0:
        return np.zeros_like(V)
    k = min(landmarks, n)
    rng = rng or np.random.default_rng(config.IF_RANDOM_STATE)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    Qs, Ks = Q[idx], K[idx]
    F = Q @ Ks.T                                   # (n, k)
    B = Qs @ Ks.T                                  # (k, k)
    B = B + eps * np.eye(k, dtype=B.dtype)
    G = Qs @ K.T                                   # (k, n)
    C = G @ V                                      # (k, d_v)
    return F @ np.linalg.solve(B, C)


def vanilla_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standard softmax self-attention, O(n^2 d), used for the ablation."""
    n = Q.shape[0]
    if n == 0:
        return np.zeros_like(V)
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ V


def benchmark_attention(n_tokens: int = 128, d: int = 64, repeats: int = 200,
                        landmarks: int = config.N_LANDMARKS,
                        seed: int = 0) -> Tuple[float, float]:
    """Median latency (ms) of Nystrom vs vanilla attention for one sequence."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n_tokens, d)).astype(np.float32)
    K = rng.normal(size=(n_tokens, d)).astype(np.float32)
    V = rng.normal(size=(n_tokens, d)).astype(np.float32)

    def _time(fn, **kw):
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(Q, K, V, **kw)
            samples.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(samples))

    return (_time(nystrom_attention, landmarks=landmarks, rng=rng),
            _time(vanilla_attention))
