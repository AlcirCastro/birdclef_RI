"""Exact (brute-force) FAISS index — the gold-standard baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("flat")
class FlatIndex(BaseIndex):
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._index = None
        self._dim = 0
        self._n = 0

    def _build_index(self, dim: int):
        import faiss
        if self.metric == "cosine":
            return faiss.IndexFlatIP(dim)  # cosine via L2-normalized inputs
        if self.metric == "l2":
            return faiss.IndexFlatL2(dim)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def build(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._index = self._build_index(self._dim)
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._n = int(vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        D, I = self._index.search(q, min(k, self._n))
        # FAISS returns L2 distances when metric=l2 — flip to higher-is-better.
        if self.metric == "l2":
            D = -D
        return D, I

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        return self._n * self._dim * 4

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path))
        self._n = int(self._index.ntotal)
        self._dim = int(self._index.d)
