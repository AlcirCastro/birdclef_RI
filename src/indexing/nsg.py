"""NSG — Navigating Spread-out Graph index (FAISS backend).

NSG builds a sparse directed graph where each node connects to R neighbors
selected by a Relative Neighborhood Graph (RNG) pruning rule. Unlike HNSW
there is no hierarchical structure: all nodes live in a single flat layer,
and search is a greedy beam traversal from a fixed medoid entry point.

Reference: Fu et al., "Fast Approximate Nearest Neighbor Search With The
Navigating Spread-out Graphs", VLDB 2019.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("nsg")
class NSGIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", R: int = 32, search_L: int = 64):
        self.metric = metric
        self.R = R          # max out-degree (graph degree)
        self.search_L = search_L   # beam width at query time
        self._index = None
        self._dim = 0
        self._n = 0

    def build(self, vectors: np.ndarray) -> None:
        import faiss
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        x = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.metric == "cosine":
            self._index = faiss.IndexNSGFlat(self._dim, self.R, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == "l2":
            self._index = faiss.IndexNSGFlat(self._dim, self.R, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        self._index.nsg.search_L = self.search_L
        self._index.add(x)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        D, I = self._index.search(q, min(k, self._n))
        if self.metric == "l2":
            D = -D
        return D, I

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        # vectors + graph edges (each edge = 4 bytes int)
        return self._n * (self._dim * 4 + self.R * 4)

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path))
        self._n = int(self._index.ntotal)
        self._dim = int(self._index.d)
