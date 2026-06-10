"""IVF + Product Quantization — compressed memory at modest recall cost."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("ivfpq")
class IVFPQIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", nlist: int | None = None,
                 m: int = 16, nbits: int = 8, nprobe: int = 8):
        self.metric = metric
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._index = None
        self._dim = 0
        self._n = 0

    def build(self, vectors: np.ndarray) -> None:
        import faiss
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        if self._dim % self.m != 0:
            raise ValueError(f"PQ requires dim ({self._dim}) divisible by m ({self.m})")
        self._n = int(vectors.shape[0])
        nlist = self.nlist or max(1, min(self._n, int(round(np.sqrt(self._n)))))
        metric_id = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        quant = faiss.IndexFlatIP(self._dim) if self.metric == "cosine" else faiss.IndexFlatL2(self._dim)
        self._index = faiss.IndexIVFPQ(quant, self._dim, nlist, self.m, self.nbits, metric_id)

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
        # PQ codes: m bytes per vector (when nbits=8)
        return self._n * self.m * (self.nbits // 8 if self.nbits >= 8 else 1)

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path))
        self._n = int(self._index.ntotal)
        self._dim = int(self._index.d)
