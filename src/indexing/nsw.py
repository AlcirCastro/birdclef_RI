"""NSW — Navigable Small World graph index (pure Python + FAISS distances).

NSW is the single-layer predecessor of HNSW. Construction builds a
bidirectional k-NN graph using exact FAISS search; retrieval uses the
standard greedy beam expansion (Algorithm 2 from Malkov & Yashunin 2014,
before the hierarchical extension was added).

Search complexity: O(ef_search · R · d) per query where R is the graph
degree and d the embedding dimension. No hierarchical routing layer means
larger ef_search is needed than HNSW to achieve the same recall.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("nsw")
class NSWIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", M: int = 32,
                 ef_search: int = 128, n_entry: int = 16):
        self.metric = metric
        self.M = M              # k-NN graph degree (bidirectional)
        self.ef_search = ef_search
        self.n_entry = n_entry  # number of random entry points per query
        self._adj: list[list[int]] = []
        self._vectors: np.ndarray | None = None
        self._dim = 0
        self._n = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray) -> None:
        import faiss
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        self._vectors = x

        k = min(self.M, self._n - 1)

        # Exact k-NN graph via FAISS flat (bidirectional)
        if self.metric == "cosine":
            flat = faiss.IndexFlatIP(self._dim)
        else:
            flat = faiss.IndexFlatL2(self._dim)
        flat.add(x)
        _, I = flat.search(x, k + 1)  # +1 to include self (filtered below)

        graph: list[set[int]] = [set() for _ in range(self._n)]
        for i in range(self._n):
            for j in I[i]:
                j = int(j)
                if j >= 0 and j != i:
                    graph[i].add(j)
                    graph[j].add(i)

        self._adj = [sorted(s) for s in graph]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _sim(self, a: int, q: np.ndarray) -> float:
        v = self._vectors[a]
        if self.metric == "cosine":
            return float(v @ q)
        return -float(np.sum((v - q) ** 2))

    def _beam_search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy beam search on the NSW graph (Algorithm 2, Malkov 2014)."""
        rng = np.random.default_rng(int(np.abs(q).sum() * 1e6) % (2 ** 31))
        entries = rng.choice(self._n, size=min(self.n_entry, self._n), replace=False)

        visited: set[int] = set()
        # W: result candidates, min-heap of (score, id) — worst at top
        W: list[Tuple[float, int]] = []
        # C: exploration frontier, min-heap of (-score, id) — best at top
        C: list[Tuple[float, int]] = []

        for ep in entries:
            ep = int(ep)
            if ep not in visited:
                visited.add(ep)
                s = self._sim(ep, q)
                heapq.heappush(W, (s, ep))
                heapq.heappush(C, (-s, ep))

        ef = self.ef_search

        while C:
            neg_best, c = heapq.heappop(C)
            best_score = -neg_best

            # Stop: best unexplored candidate is worse than worst kept result
            if len(W) >= ef and best_score <= W[0][0]:
                break

            for nb in self._adj[c]:
                if nb not in visited:
                    visited.add(nb)
                    s = self._sim(nb, q)
                    if len(W) < ef or s > W[0][0]:
                        heapq.heappush(W, (s, nb))
                        heapq.heappush(C, (-s, nb))
                        if len(W) > ef:
                            heapq.heappop(W)

        top = sorted(W, reverse=True)[:k]
        ids = np.array([t[1] for t in top], dtype=np.int64)
        scores = np.array([t[0] for t in top], dtype=np.float32)
        return scores, ids

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        m = queries.shape[0]
        all_scores = np.full((m, k), -np.inf, dtype=np.float32)
        all_ids = np.full((m, k), -1, dtype=np.int64)
        for qi in range(m):
            sc, ids = self._beam_search(queries[qi], k)
            r = len(sc)
            all_scores[qi, :r] = sc
            all_ids[qi, :r] = ids
        return all_scores, all_ids

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        graph_edges = sum(len(a) for a in self._adj)
        return self._n * self._dim * 4 + graph_edges * 4

    def save(self, path: Path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"adj": self._adj, "vectors": self._vectors,
                         "metric": self.metric, "n": self._n, "dim": self._dim}, f)

    def load(self, path: Path) -> None:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._adj = d["adj"]
        self._vectors = d["vectors"]
        self.metric = d["metric"]
        self._n = d["n"]
        self._dim = d["dim"]
