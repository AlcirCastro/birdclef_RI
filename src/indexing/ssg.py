"""SSG — Satellite System Graph index.

SSG augments a base k-NN graph with *satellite* edges: for each existing
edge p→q, any point s that lies inside the angular cone centred on the
direction p→q (cos(angle) > cos_theta) AND is closer to p than q is
becomes a satellite of q with respect to p, and an edge p→s is added.

This broadens graph connectivity compared to NSG's RNG pruning, giving
better recall-per-degree in practice.

Reference: Fu et al., "Satellite System Graph: Towards the Efficiency
Up-Boundary of Graph-Based Approximate Nearest Neighbor Search", IEEE
TPAMI 2021.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("ssg")
class SSGIndex(BaseIndex):
    def __init__(self, metric: str = "cosine", M: int = 32,
                 cos_theta: float = 0.5, ef_search: int = 128,
                 n_entry: int = 8):
        self.metric = metric
        self.M = M              # base k-NN graph degree
        self.cos_theta = cos_theta  # satellite cone threshold (cosine of angle)
        self.ef_search = ef_search
        self.n_entry = n_entry
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

        # --- Step 1: exact k-NN base graph ---
        k = min(self.M, self._n - 1)
        if self.metric == "cosine":
            flat = faiss.IndexFlatIP(self._dim)
        else:
            flat = faiss.IndexFlatL2(self._dim)
        flat.add(x)
        _, I = flat.search(x, k + 1)

        graph: list[set[int]] = [set() for _ in range(self._n)]
        for i in range(self._n):
            for j in I[i]:
                j = int(j)
                if j >= 0 and j != i:
                    graph[i].add(j)
                    graph[j].add(i)

        # --- Step 2: satellite augmentation ---
        # For each node p and each neighbour q of p:
        #   add edges p→s for all s that are in the cone of (p,q)
        #   and dist(p,s) ≤ dist(p,q)  [i.e. s is at least as close as q]
        # Satellite condition (cosine metric):
        #   direction p→q ≈ (v_q - v_p) / ||v_q - v_p||
        #   cos_angle(s) = dir_pq · (v_s - v_p)/||v_s - v_p||  ≥  cos_theta
        # AND sim(p,s) ≥ sim(p,q)  (s at least as similar to p as q is)
        for p in range(self._n):
            v_p = x[p]
            new_edges: set[int] = set()

            for q in list(graph[p]):
                v_q = x[q]
                sim_pq = float(v_p @ v_q) if self.metric == "cosine" else -float(np.sum((v_p - v_q) ** 2))

                # Direction vector p→q (unnormalised)
                diff_pq = v_q - v_p
                norm_pq = float(np.linalg.norm(diff_pq)) + 1e-12
                dir_pq = diff_pq / norm_pq

                # Check all current neighbours of q as satellite candidates
                for s in graph[q]:
                    if s == p or s in graph[p]:
                        continue
                    v_s = x[s]
                    sim_ps = float(v_p @ v_s) if self.metric == "cosine" else -float(np.sum((v_p - v_s) ** 2))

                    # s must be at least as close to p as q is
                    if sim_ps < sim_pq:
                        continue

                    # Cone angle check
                    diff_ps = v_s - v_p
                    norm_ps = float(np.linalg.norm(diff_ps)) + 1e-12
                    cos_angle = float(dir_pq @ (diff_ps / norm_ps))
                    if cos_angle >= self.cos_theta:
                        new_edges.add(s)

            graph[p].update(new_edges)

        # Trim to M + extra satellite budget (cap at 2*M to bound memory)
        cap = self.M * 2
        self._adj = []
        for p in range(self._n):
            nbrs = list(graph[p])
            if len(nbrs) > cap:
                # Keep closest cap neighbours
                scores = x[np.array(nbrs, dtype=np.int64)] @ x[p] if self.metric == "cosine" else (
                    -np.sum((x[np.array(nbrs, dtype=np.int64)] - x[p]) ** 2, axis=1)
                )
                order = np.argsort(-scores)[:cap]
                nbrs = [nbrs[i] for i in order]
            self._adj.append(sorted(nbrs))

    # ------------------------------------------------------------------
    # Search (same greedy beam as NSW)
    # ------------------------------------------------------------------

    def _sim(self, a: int, q: np.ndarray) -> float:
        v = self._vectors[a]
        if self.metric == "cosine":
            return float(v @ q)
        return -float(np.sum((v - q) ** 2))

    def _beam_search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(np.abs(q).sum() * 1e6) % (2 ** 31))
        entries = rng.choice(self._n, size=min(self.n_entry, self._n), replace=False)

        visited: set[int] = set()
        W: list[Tuple[float, int]] = []
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
            if -neg_best <= (W[0][0] if len(W) >= ef else -np.inf):
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
