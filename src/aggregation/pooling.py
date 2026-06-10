"""Pooling functions over a (n, dim) embedding matrix → (dim,) vector.

All return L2-normalized output so the choice of similarity metric stays
consistent across representation types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.aggregation import AGGREGATORS


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)


class BaseAggregator(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        ...


@AGGREGATORS.register("mean")
class MeanAggregator(BaseAggregator):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        return _l2(X.mean(axis=0))


@AGGREGATORS.register("max")
class MaxAggregator(BaseAggregator):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        return _l2(X.max(axis=0))


@AGGREGATORS.register("topk_mean")
class TopKMeanAggregator(BaseAggregator):
    """Mean of the top-k segments by L2 norm (proxy for vocal energy)."""
    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        if X.shape[0] <= self.k:
            return _l2(X.mean(axis=0))
        norms = np.linalg.norm(X, axis=1)
        idx = np.argpartition(-norms, self.k - 1)[: self.k]
        return _l2(X[idx].mean(axis=0))


@AGGREGATORS.register("median")
class MedianAggregator(BaseAggregator):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        return _l2(np.median(X, axis=0))


@AGGREGATORS.register("attention")
class AttentionAggregator(BaseAggregator):
    """Self-attention–style pooling against the centroid as the query.

    weights = softmax(<x_i, centroid> / T); output = sum w_i * x_i.
    Cheap, parameter-free, and emphasizes segments that align with the
    dominant direction.
    """
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        if X.shape[0] == 1:
            return _l2(X[0])
        centroid = _l2(X.mean(axis=0))
        scores = X @ centroid / max(self.temperature, 1e-6)
        scores = scores - scores.max()
        w = np.exp(scores)
        w /= w.sum() + 1e-12
        return _l2((w[:, None] * X).sum(axis=0))


@AGGREGATORS.register("spe")
class SpatialPyramidEncoder(BaseAggregator):
    """Spatial Pyramid Encoding — multi-scale temporal pooling.

    For each level L in `levels`, split the segment matrix into L equal
    consecutive chunks along time and pool each chunk; the per-level pooled
    vectors are concatenated. Captures both global character (L=1) and
    local sub-clip structure (L=2,4,…) in a single fixed-length vector.

    Output dim = sum(levels) * input_dim  (× 2 if `include_max=True`).
    The query-side aggregator must use the same SPE config so query and
    doc vectors share dimensionality — set `fusion.params.aggregator: spe`.
    Not compatible with `representation: segment` (which bypasses the
    aggregator).
    """
    def __init__(self, levels: list[int] | None = None, include_max: bool = False):
        self.levels = list(levels) if levels else [1, 2, 4]
        if any(L <= 0 for L in self.levels):
            raise ValueError("SPE levels must be positive")
        self.include_max = include_max

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        n = X.shape[0]
        parts: list[np.ndarray] = []
        for L in self.levels:
            # Even split with the remainder distributed to the first chunks.
            edges = np.linspace(0, n, L + 1).round().astype(int)
            for i in range(L):
                lo, hi = int(edges[i]), int(edges[i + 1])
                if hi <= lo:  # fewer segments than level cells → repeat last available
                    lo, hi = max(0, n - 1), n
                chunk = X[lo:hi]
                parts.append(chunk.mean(axis=0))
                if self.include_max:
                    parts.append(chunk.max(axis=0))
        return _l2(np.concatenate(parts, axis=0))
