"""Inverted-file (IVF-Flat) — coarse quantizer + per-cluster scan.

Trains on the corpus itself. nlist auto-defaults to ~sqrt(n) which is the
common rule of thumb. Probe count `nprobe` trades recall for latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("ivf")
class IVFIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", nlist: int | None = None,
                 nprobe: int = 8):
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None
        self._dim = 0
        self._n = 0

    def build(self, vectors: np.ndarray) -> None:
        import faiss
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        nlist = self.nlist or max(1, min(self._n, int(round(np.sqrt(self._n)))))
        if self.metric == "cosine":
            quant = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIVFFlat(quant, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == "l2":
            quant = faiss.IndexFlatL2(self._dim)
            self._index = faiss.IndexIVFFlat(quant, self._dim, nlist, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        x = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.train(x)
        self._index.add(x)
        self._index.nprobe = min(self.nprobe, nlist)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
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
