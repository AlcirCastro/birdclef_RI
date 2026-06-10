"""Aggregate metrics over a test set.

Knows nothing about Perch, FAISS, or experiments — it just takes per-query
ranked label lists and returns a metrics dict + per-query rows for CSV.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from src.data.records import Record
from src.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.ranking.base import RankedResult


@dataclass
class QueryResult:
    record_id: int
    true_label: str
    predictions: List[RankedResult]
    latency_ms: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)


class Evaluator:
    def __init__(self, p_at: Sequence[int], recall_at: Sequence[int], top_k: int):
        self.p_at = list(p_at)
        self.recall_at = list(recall_at)
        self.top_k = top_k

    def evaluate(self, results: Sequence[QueryResult],
                 corpus_label_counts: Dict[str, int]) -> dict:
        maps, mrrs = [], []
        p_at: Dict[int, list[float]] = {k: [] for k in self.p_at}
        r_at: Dict[int, list[float]] = {k: [] for k in self.recall_at}
        ndcgs = []
        latencies = []
        per_query_rows: list[dict] = []

        confusion_pairs = Counter()

        for r in results:
            labels = [p.label for p in r.predictions[: self.top_k]]
            rels = [1 if lbl == r.true_label else 0 for lbl in labels]
            total_rel = max(1, corpus_label_counts.get(r.true_label, 1))

            ap = average_precision(rels)
            rr = reciprocal_rank(rels)
            ndcg = ndcg_at_k(rels, self.top_k)
            maps.append(ap)
            mrrs.append(rr)
            ndcgs.append(ndcg)

            for k in self.p_at:
                p_at[k].append(precision_at_k(rels, k))
            for k in self.recall_at:
                r_at[k].append(recall_at_k(rels, k, total_rel))

            latencies.append(r.latency_ms)
            top1 = labels[0] if labels else ""
            if top1 and top1 != r.true_label:
                confusion_pairs[(r.true_label, top1)] += 1

            per_query_rows.append({
                "record_id": r.record_id,
                "true_label": r.true_label,
                "top1_pred": top1,
                "AP": ap,
                "RR": rr,
                "nDCG": ndcg,
                "latency_ms": r.latency_ms,
                **{f"P@{k}": precision_at_k(rels, k) for k in self.p_at},
                **{f"R@{k}": recall_at_k(rels, k, total_rel) for k in self.recall_at},
            })

        def m(arr): return float(np.mean(arr)) if arr else 0.0

        summary = {
            "n_queries": len(results),
            "MAP": m(maps),
            "MRR": m(mrrs),
            "nDCG": m(ndcgs),
            "avg_latency_ms": m(latencies),
            **{f"P@{k}": m(p_at[k]) for k in self.p_at},
            **{f"R@{k}": m(r_at[k]) for k in self.recall_at},
        }
        return {
            "summary": summary,
            "per_query": per_query_rows,
            "top_confusions": confusion_pairs.most_common(20),
        }
