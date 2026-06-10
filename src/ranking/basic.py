"""Basic per-label score aggregations across query segments.

These rankers all collapse multiple per-query result lists into a single
{label: score} dict, then return the top-k labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

import numpy as np

from src.ranking import RANKERS
from src.ranking.base import BaseRanker, Hit, RankedResult, _take_top_k


@RANKERS.register("segment")
class SegmentLevelRanker(BaseRanker):
    """No aggregation: flatten every hit into the final list, dedup by label keeping
    the best score. Pure top-1 hit retrieval."""

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = {}
        for hits in per_query_hits:
            for h in hits:
                if h.score > scores.get(h.label, -np.inf):
                    scores[h.label] = float(h.score)
        return _take_top_k(scores, k)


@RANKERS.register("mean")
class MeanScoreRanker(BaseRanker):
    """Per-label mean of all hit scores across queries."""

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for hits in per_query_hits:
            for h in hits:
                sums[h.label] += float(h.score)
                counts[h.label] += 1
        scores = {lbl: sums[lbl] / counts[lbl] for lbl in sums}
        return _take_top_k(scores, k)


@RANKERS.register("max")
class MaxScoreRanker(BaseRanker):
    """Per-label max score (= SegmentLevelRanker by another name; kept for clarity)."""

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        return SegmentLevelRanker().rank(per_query_hits, k)


@RANKERS.register("topk_mean")
class TopKMeanRanker(BaseRanker):
    """Per-label mean of the top-`per_label_k` scores."""
    def __init__(self, per_label_k: int = 3):
        self.per_label_k = per_label_k

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for hits in per_query_hits:
            for h in hits:
                buckets[h.label].append(float(h.score))
        scores = {
            lbl: float(np.mean(sorted(vals, reverse=True)[: self.per_label_k]))
            for lbl, vals in buckets.items()
        }
        return _take_top_k(scores, k)


@RANKERS.register("hit")
class HitBasedRanker(BaseRanker):
    """Vote count: a label's score is the number of query segments it appeared in."""

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = defaultdict(float)
        for hits in per_query_hits:
            for lbl in {h.label for h in hits}:
                scores[lbl] += 1.0
        return _take_top_k(scores, k)


@RANKERS.register("median")
class MedianScoreRanker(BaseRanker):
    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for hits in per_query_hits:
            for h in hits:
                buckets[h.label].append(float(h.score))
        scores = {lbl: float(np.median(vals)) for lbl, vals in buckets.items()}
        return _take_top_k(scores, k)


@RANKERS.register("threshold")
class ThresholdRanker(BaseRanker):
    """Mean over only those scores above `tau`. Labels with no score above tau drop."""
    def __init__(self, tau: float = 0.5):
        self.tau = tau

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for hits in per_query_hits:
            for h in hits:
                if h.score >= self.tau:
                    buckets[h.label].append(float(h.score))
        scores = {lbl: float(np.mean(vals)) for lbl, vals in buckets.items() if vals}
        return _take_top_k(scores, k)


@RANKERS.register("weighted_topk")
class WeightedTopKRanker(BaseRanker):
    """Top-k mean with linearly decaying rank weights."""
    def __init__(self, per_label_k: int = 5):
        self.per_label_k = per_label_k

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for hits in per_query_hits:
            for h in hits:
                buckets[h.label].append(float(h.score))
        scores: dict[str, float] = {}
        for lbl, vals in buckets.items():
            top = sorted(vals, reverse=True)[: self.per_label_k]
            weights = np.linspace(1.0, 0.5, num=len(top))
            scores[lbl] = float(np.dot(top, weights) / weights.sum())
        return _take_top_k(scores, k)
