"""Vamana (DiskANN) graph index — pure Python implementation.

Vamana builds a directed graph where each node has at most R out-edges
chosen by a *robust pruning* rule (α-RNG): given a candidate set, add
the globally closest node first, then discard any candidate q where the
new edge covers q well enough (α · dist(new, q) ≤ dist(p, q)).
With α > 1 the pruning is less aggressive, producing longer-range edges
that improve recall at the same graph degree.

The entry point for search is the dataset medoid (most central vector).
A single greedy beam expansion from there finds the query's neighborhood.

Reference: Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-point
Nearest Neighbor Search on a Single Node", NeurIPS 2019.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("vamana")
class VamanaIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", R: int = 32, L: int = 100,
                 alpha: float = 1.2, n_iters: int = 2):
        self.metric = metric
        self.R = R          # max out-degree
        self.L = L          # beam width during construction and search
        self.alpha = alpha  # robust-pruning threshold (> 1 = longer range)
        self.n_iters = n_iters
        self._out: list[list[int]] = []
        self._vectors: np.ndarray | None = None
        self._medoid: int = 0
        self._dim = 0
        self._n = 0

    # ------------------------------------------------------------------
    # Distance helpers (higher score = closer, for both metrics)
    # ------------------------------------------------------------------

    def _score_vec(self, ref_vec: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Return similarity scores: (|ids|,) array, higher is better."""
        vecs = self._vectors[ids]
        if self.metric == "cosine":
            return (vecs @ ref_vec).astype(np.float32)
        return -np.sum((vecs - ref_vec) ** 2, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _find_medoid(self) -> int:
        """Index of the vector closest to the dataset centroid."""
        centroid = self._vectors.mean(axis=0)
        if self.metric == "cosine":
            c = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = self._vectors @ c
        else:
            sims = -np.sum((self._vectors - centroid) ** 2, axis=1)
        return int(np.argmax(sims))

    def _greedy_search_build(self, target_id: int) -> list[Tuple[float, int]]:
        """Beam search from medoid toward target_id; returns sorted [(score, id)]."""
        visited: set[int] = {self._medoid}
        s0 = float(self._score_vec(self._vectors[target_id], np.array([self._medoid]))[0])

        W: list[Tuple[float, int]] = [(s0, self._medoid)]
        C: list[Tuple[float, int]] = [(-s0, self._medoid)]

        while C:
            neg_best, c = heapq.heappop(C)
            best_score = -neg_best
            if len(W) >= self.L and best_score <= W[0][0]:
                break
            neighbors = [nb for nb in self._out[c] if nb not in visited]
            if neighbors:
                visited.update(neighbors)
                nb_arr = np.array(neighbors, dtype=np.int64)
                scores = self._score_vec(self._vectors[target_id], nb_arr)
                for nb, s in zip(neighbors, scores.tolist()):
                    s = float(s)
                    if len(W) < self.L or s > W[0][0]:
                        heapq.heappush(W, (s, nb))
                        heapq.heappush(C, (-s, nb))
                        if len(W) > self.L:
                            heapq.heappop(W)

        return sorted(W, reverse=True)

    def _robust_prune(self, p_id: int,
                      candidates: list[Tuple[float, int]]) -> list[int]:
        """α-RNG pruning: greedily pick closest, evict neighbours it covers."""
        p_vec = self._vectors[p_id]
        result: list[int] = []
        available = [c for c in candidates if c[1] != p_id]  # exclude self

        while available and len(result) < self.R:
            # Pick the candidate closest to p (highest score)
            best_score, best_id = available[0]
            result.append(best_id)

            if len(result) == self.R:
                break

            best_vec = self._vectors[best_id]
            surviving: list[Tuple[float, int]] = []
            for s_pq, qid in available[1:]:
                s_best_q = float(self._score_vec(best_vec, np.array([qid]))[0])
                # Keep q if α² · (1 − sim(best,q)) > (1 − sim(p,q))
                # (cosine version of α · dist(best,q) > dist(p,q))
                if self.metric == "cosine":
                    keep = self.alpha ** 2 * (1.0 - s_best_q) > (1.0 - s_pq)
                else:
                    # s_pq = -dist²(p,q), s_best_q = -dist²(best,q)
                    keep = self.alpha ** 2 * (-s_best_q) > (-s_pq)
                if keep:
                    surviving.append((s_pq, qid))

            available = surviving

        return result

    def build(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        rng = np.random.default_rng(42)

        # Random initialisation: R random out-edges per node
        self._out = [
            list(rng.choice(
                [j for j in range(self._n) if j != i],
                size=min(self.R, self._n - 1),
                replace=False,
            ))
            for i in range(self._n)
        ]

        self._medoid = self._find_medoid()

        for _iter in range(self.n_iters):
            order = rng.permutation(self._n)
            for i in order:
                # Beam search from medoid → candidate set near i
                candidates = self._greedy_search_build(i)

                # Merge with current out-edges of i
                existing_scores = self._score_vec(
                    self._vectors[i], np.array(self._out[i], dtype=np.int64)
                )
                existing_cands = list(zip(existing_scores.tolist(), self._out[i]))

                merged: dict[int, float] = {}
                for s, nid in candidates + existing_cands:
                    if nid != i:
                        if nid not in merged or merged[nid] < s:
                            merged[nid] = float(s)
                sorted_merged = sorted(merged.items(), key=lambda x: -x[1])
                sorted_merged = [(s, nid) for nid, s in sorted_merged]

                # Robust prune → new out-edges for i
                self._out[i] = self._robust_prune(i, sorted_merged)

                # Back-edges: add i to each neighbour's edge list; prune if overflow
                for nb in self._out[i]:
                    if i not in self._out[nb]:
                        self._out[nb].append(i)
                        if len(self._out[nb]) > self.R:
                            nb_scores = self._score_vec(
                                self._vectors[nb],
                                np.array(self._out[nb], dtype=np.int64),
                            )
                            nb_cands = sorted(
                                zip(nb_scores.tolist(), self._out[nb]),
                                reverse=True,
                            )
                            self._out[nb] = self._robust_prune(
                                nb, [(s, nid) for s, nid in nb_cands]
                            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _greedy_search_query(self, q: np.ndarray, k: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy beam search from medoid entry point."""
        s0 = float(self._score_vec(q, np.array([self._medoid]))[0])
        visited: set[int] = {self._medoid}

        W: list[Tuple[float, int]] = [(s0, self._medoid)]
        C: list[Tuple[float, int]] = [(-s0, self._medoid)]

        while C:
            neg_best, c = heapq.heappop(C)
            best_score = -neg_best
            if len(W) >= self.L and best_score <= W[0][0]:
                break
            neighbors = [nb for nb in self._out[c] if nb not in visited]
            if neighbors:
                visited.update(neighbors)
                nb_arr = np.array(neighbors, dtype=np.int64)
                scores = self._score_vec(q, nb_arr)
                for nb, s in zip(neighbors, scores.tolist()):
                    s = float(s)
                    if len(W) < self.L or s > W[0][0]:
                        heapq.heappush(W, (s, nb))
                        heapq.heappush(C, (-s, nb))
                        if len(W) > self.L:
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
            sc, ids = self._greedy_search_query(
                np.ascontiguousarray(queries[qi], dtype=np.float32), k
            )
            r = len(sc)
            all_scores[qi, :r] = sc
            all_ids[qi, :r] = ids
        return all_scores, all_ids

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        graph_edges = sum(len(e) for e in self._out)
        return self._n * self._dim * 4 + graph_edges * 4

    def save(self, path: Path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "out": self._out, "vectors": self._vectors,
                "medoid": self._medoid, "metric": self.metric,
                "n": self._n, "dim": self._dim,
            }, f)

    def load(self, path: Path) -> None:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._out = d["out"]
        self._vectors = d["vectors"]
        self._medoid = d["medoid"]
        self.metric = d["metric"]
        self._n = d["n"]
        self._dim = d["dim"]
