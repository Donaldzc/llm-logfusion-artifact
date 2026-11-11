"""384-dimensional semantic embedding with Nystromformer linearised attention.

Backends
--------
* ``sbert``  : all-MiniLM-L6-v2 (requires ``sentence-transformers`` + torch).
* ``lsa``    : offline fallback used in this environment -- word-level TF-IDF
               factorised with truncated SVD; each vocabulary entry obtains a
               384-d vector, logs are encoded as a token sequence refined by the
               Nystromformer attention of Eq. (3) and mean-pooled to 384-d.

Both backends return the 384-d vector required by the paper so the downstream
398-d fusion is unchanged.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from . import config
from .nystrom import nystrom_attention, vanilla_attention

_TOKEN_RE = re.compile(r"[A-Za-z0-9@_.$:/\-]+")
_MAX_TOKENS = 64
_HEAD_DIM = 64


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())[:_MAX_TOKENS]


class SemanticEncoder:
    def __init__(self, backend: str = "auto", dim: int = config.N_SEMANTIC,
                 landmarks: int = config.N_LANDMARKS, seed: int = 0,
                 attention: str = "nystrom") -> None:
        self.dim = dim
        self.landmarks = landmarks
        self.seed = seed
        self.attention = attention
        self.backend = self._resolve_backend(backend)
        self._rng = np.random.default_rng(seed)
        self._wq = self._projection(dim, _HEAD_DIM)
        self._wk = self._projection(dim, _HEAD_DIM)
        self._wv = self._projection(dim, _HEAD_DIM)
        self._wo = self._projection(_HEAD_DIM, dim)
        self._vocab: dict[str, int] = {}
        self._token_vectors: np.ndarray | None = None
        self._sbert = None
        self.encode_seconds_ = 0.0

    # ------------------------------------------------------------------ setup
    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            try:
                import sentence_transformers  # noqa: F401
                import torch  # noqa: F401
                return "sbert"
            except Exception:
                return "lsa"
        return backend

    def _projection(self, fan_in: int, fan_out: int) -> np.ndarray:
        return (self._rng.normal(size=(fan_in, fan_out)) / np.sqrt(fan_in)).astype(np.float32)

    def fit(self, texts: Sequence[str]) -> "SemanticEncoder":
        if self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(config.SBERT_MODEL)
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                              lowercase=True, min_df=2, max_features=12000,
                              sublinear_tf=True, dtype=np.float32)
        mat = vec.fit_transform(texts)
        n_comp = min(self.dim, max(2, min(mat.shape) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed, n_iter=4)
        # Fitting the factorisation on a subsample keeps the one-off cost low;
        # the resulting term-topic matrix still covers the full vocabulary.
        if mat.shape[0] > 8000:
            sample_idx = self._rng.choice(mat.shape[0], size=8000, replace=False)
            svd.fit(mat[sample_idx])
        else:
            svd.fit(mat)
        token_space = svd.components_.T                  # (vocab, n_comp)
        if n_comp < self.dim:
            token_space = np.pad(token_space, ((0, 0), (0, self.dim - n_comp)))
        self._vocab = {t: i for i, t in enumerate(vec.get_feature_names_out())}
        self._token_vectors = np.ascontiguousarray(token_space, dtype=np.float32)
        return self

    # ---------------------------------------------------------------- encoding
    def _token_matrix(self, text: str) -> np.ndarray:
        if self._token_vectors is None:
            raise RuntimeError("SemanticEncoder must be fitted first")
        idx = [self._vocab[t] for t in tokenize(text) if t in self._vocab]
        if not idx:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._token_vectors[idx]

    def _refine(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n <= 1:
            return X.mean(axis=0, keepdims=True) if n else np.zeros((1, self.dim), np.float32)
        Q = X @ self._wq
        K = X @ self._wk
        V = X @ self._wv
        if self.attention == "vanilla":
            ctx = vanilla_attention(Q, K, V)
        else:
            ctx = nystrom_attention(Q, K, V, landmarks=self.landmarks,
                                    eps=config.ATTN_EPS, rng=self._rng)
        return ctx @ self._wo

    def encode_one(self, text: str) -> np.ndarray:
        if self.backend == "sbert":
            vec = self._sbert.encode([text], batch_size=config.EMBED_BATCH,
                                     normalize_embeddings=False)[0]
            vec = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        X = self._token_matrix(text)
        if X.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        pooled = self._refine(X).mean(axis=0)
        norm = np.linalg.norm(pooled)
        return (pooled / norm if norm > 0 else pooled).astype(np.float32)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        t0 = time.perf_counter()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        step = max(1, len(texts) // 10)
        for i, text in enumerate(texts):
            out[i] = self.encode_one(text)
            if show_progress and (i + 1) % step == 0:
                print(f"    embedded {i + 1}/{len(texts)}", flush=True)
        self.encode_seconds_ = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _cache_key(texts: Sequence[str], backend: str, attention: str) -> str:
        h = hashlib.md5()
        h.update(backend.encode())
        h.update(attention.encode())
        h.update(str(len(texts)).encode())
        for t in texts[::max(1, len(texts) // 500)]:
            h.update(t.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def encode_cached(self, texts: Sequence[str], cache_dir: Path = config.CACHE_DIR,
                      tag: str = "") -> np.ndarray:
        key = self._cache_key(texts, f"{self.backend}-{tag}", self.attention)
        path = Path(cache_dir) / f"emb_{key}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape == (len(texts), self.dim):
                return arr
        arr = self.encode(texts)
        np.save(path, arr)
        return arr


def build_semantic_features(df: pd.DataFrame, attention: str = "nystrom",
                            backend: str = "auto", tag: str = "") -> tuple[np.ndarray, SemanticEncoder]:
    encoder = SemanticEncoder(backend=backend, attention=attention).fit(df["raw"].tolist())
    emb = encoder.encode_cached(df["raw"].tolist(), tag=tag)
    return emb, encoder
