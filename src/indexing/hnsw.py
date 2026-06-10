"""HNSW graph index — sub-linear search with high recall."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("hnsw")
class HNSWIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", M: int = 32,
                 ef_construction: int = 200, ef_search: int = 64):
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None
        self._dim = 0
        self._n = 0

    def build(self, vectors: np.ndarray) -> None:
        import faiss
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        if self.metric == "cosine":
            self._index = faiss.IndexHNSWFlat(self._dim, self.M, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == "l2":
            self._index = faiss.IndexHNSWFlat(self._dim, self.M, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        D, I = self._index.search(q, min(k, self._n))
        if self.metric == "l2":
            D = -D
        return D, I

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        # rough: vectors + graph links
        return self._n * (self._dim * 4 + self.M * 8)

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path))
        self._n = int(self._index.ntotal)
        self._dim = int(self._index.d)
