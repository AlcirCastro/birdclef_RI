import time

import numpy as np

from bird_search.embedding import Embedder
from bird_search.models import Record
from bird_search.search import SearchIndex


def _ap(rels: list[int]) -> float:
    total = sum(rels)
    if total == 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, rel in enumerate(rels, start=1):
        if rel == 1:
            hits += 1
            score += hits / i
    return score / total


def _rr(rels: list[int]) -> float:
    for i, rel in enumerate(rels, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def _pk(rels: list[int], k: int) -> float:
    chunk = rels[:k]
    if not chunk:
        return 0.0
    return sum(chunk) / len(chunk)


def _hit(rels: list[int], k: int) -> float:
    return 1.0 if any(r == 1 for r in rels[:k]) else 0.0


def evaluate(test: list[Record], embedder: Embedder, search_index: SearchIndex, k: int) -> dict:
    maps: list[float] = []
    mrrs: list[float] = []
    p5s: list[float] = []
    t1s: list[float] = []
    t5s: list[float] = []
    query_times: list[float] = []

    for rec in test:
        vec = embedder.embed_path(rec.local_path)
        if vec is None:
            continue

        start = time.perf_counter()
        # Leave-one-out style guard: remove the query item itself from ranked hits.
        hits = search_index.knn_search(vec, k=k, exclude_item_id=rec.item_id)
        query_times.append(time.perf_counter() - start)

        rels = [1 if h["_source"]["primary_label"] == rec.primary_label else 0 for h in hits]

        maps.append(_ap(rels))
        mrrs.append(_rr(rels))
        p5s.append(_pk(rels, 5))
        t1s.append(_hit(rels, 1))
        t5s.append(_hit(rels, 5))

    mean = lambda arr: float(np.mean(arr)) if arr else 0.0
    return {
        "MAP": mean(maps),
        "MRR": mean(mrrs),
        "P@5": mean(p5s),
        "Top-1": mean(t1s),
        "Top-5": mean(t5s),
        "avg_query_ms": mean(query_times) * 1000.0,
        "queries": len(maps),
        "eval_mode": "leave-one-out (self-match excluded)",
    }
