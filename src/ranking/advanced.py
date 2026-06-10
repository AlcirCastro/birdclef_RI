"""Advanced rank-aggregation strategies."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

import numpy as np

from src.ranking import RANKERS
from src.ranking.base import BaseRanker, Hit, RankedResult, _take_top_k
from src.utils.taxonomy import TaxonomyInfo, taxonomy_info


@RANKERS.register("softmax")
class SoftmaxWeightedRanker(BaseRanker):
    """Per-query softmax over scores; sum softmax weights per label across queries."""
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = defaultdict(float)
        for hits in per_query_hits:
            if not hits:
                continue
            arr = np.asarray([h.score for h in hits], dtype=np.float64)
            arr = (arr - arr.max()) / max(self.temperature, 1e-8)
            w = np.exp(arr)
            w /= w.sum() + 1e-12
            for h, wi in zip(hits, w):
                scores[h.label] += float(wi)
        return _take_top_k(scores, k)


@RANKERS.register("rrf")
class ReciprocalRankFusionRanker(BaseRanker):
    """RRF: sum_q 1 / (k_const + rank_q(label)) using each label's best rank per query."""
    def __init__(self, k_const: float = 60.0):
        self.k_const = k_const

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = defaultdict(float)
        for hits in per_query_hits:
            best_rank: dict[str, int] = {}
            for h in hits:
                r = h.rank if h.rank > 0 else 1
                if h.label not in best_rank or r < best_rank[h.label]:
                    best_rank[h.label] = r
            for lbl, r in best_rank.items():
                scores[lbl] += 1.0 / (self.k_const + r)
        return _take_top_k(scores, k)


@RANKERS.register("borda")
class BordaCountRanker(BaseRanker):
    """Borda: sum_q (n_q - rank_q(label)). Labels missing from a list contribute 0."""

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = defaultdict(float)
        for hits in per_query_hits:
            n = len(hits)
            best_rank: dict[str, int] = {}
            for h in hits:
                r = h.rank if h.rank > 0 else 1
                if h.label not in best_rank or r < best_rank[h.label]:
                    best_rank[h.label] = r
            for lbl, r in best_rank.items():
                scores[lbl] += float(max(0, n - r))
        return _take_top_k(scores, k)


@RANKERS.register("attention")
class AttentionWeightedRanker(BaseRanker):
    """Per-query attention over scores (softmax with learnable T) — weighted votes."""
    def __init__(self, temperature: float = 0.05, weight_by_query_norm: bool = False):
        self.temperature = temperature
        self.weight_by_query_norm = weight_by_query_norm

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        scores: dict[str, float] = defaultdict(float)
        for q_idx, hits in enumerate(per_query_hits):
            if not hits:
                continue
            arr = np.asarray([h.score for h in hits], dtype=np.float64)
            top_score = float(arr.max())
            arr = (arr - top_score) / max(self.temperature, 1e-8)
            w = np.exp(arr)
            w /= w.sum() + 1e-12
            # Weight queries by their peak similarity (more confident queries count more).
            q_weight = top_score if self.weight_by_query_norm else 1.0
            for h, wi in zip(hits, w):
                scores[h.label] += float(wi) * q_weight
        return _take_top_k(scores, k)


@RANKERS.register("taxonomy_boost")
class TaxonomyBoostRanker(BaseRanker):
    """Boost labels that share genus or common-name cues with the top hit.

    This is a pragmatic reranker for the case where top-1 is already strong,
    but the tail should surface species that are taxonomically close.
    """

    def __init__(self, genus_boost: float = 0.20, common_name_boost: float = 0.10):
        self.genus_boost = genus_boost
        self.common_name_boost = common_name_boost
        self._label_meta: dict[str, TaxonomyInfo] = {}

    def set_label_meta(self, label_meta) -> None:
        self._label_meta = dict(label_meta or {})

    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        base_scores: dict[str, float] = defaultdict(float)
        for hits in per_query_hits:
            for h in hits:
                base_scores[h.label] += float(h.score)

        if not base_scores:
            return []

        top_label = max(base_scores.items(), key=lambda kv: kv[1])[0]
        top_info = self._label_meta.get(top_label, taxonomy_info())
        top_tokens = {token for token in top_info.common_name.lower().split() if token}

        scores: dict[str, float] = dict(base_scores)
        for lbl, info in self._label_meta.items():
            if lbl == top_label:
                continue
            if top_info.genus and info.genus and info.genus == top_info.genus:
                scores[lbl] = scores.get(lbl, 0.0) + self.genus_boost
            if top_tokens:
                lbl_tokens = {token for token in info.common_name.lower().split() if token}
                if lbl_tokens and top_tokens.intersection(lbl_tokens):
                    scores[lbl] = scores.get(lbl, 0.0) + self.common_name_boost

        return _take_top_k(scores, k)
