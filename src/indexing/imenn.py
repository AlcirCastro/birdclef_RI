"""IMENN — In-Memory ExactNN Index.

Exact brute-force k-NN via FAISS flat index (inner-product or L2).
This is the theoretical upper-bound for recall at the cost of O(n·d)
search time per query. Used as the quality ceiling in index benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("imenn")
class IMENNIndex(BaseIndex):
    """In-Memory ExactNN — identical recall to flat but named for benchmarking clarity."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
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
            self._index = faiss.IndexFlatIP(self._dim)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self._dim)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not built")
        q = np.ascontiguousarray(queries, dtype=np.float32)
        D, I = self._index.search(q, min(k, self._n))
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
