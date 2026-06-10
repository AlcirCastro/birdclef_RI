"""IR metrics. All take a binary `relevances` list (1 = correct species hit, 0 = not).

Conventions:
- AP averages precision at each relevant position. If `total_relevant` is
  given, normalize by that (recall-aware); otherwise normalize by the
  number of positives observed in the ranking.
- nDCG uses the standard 2^rel - 1 numerator with log2(i+1) discount.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def average_precision(relevances: Sequence[int], total_relevant: int | None = None) -> float:
    if not relevances:
        return 0.0
    hits = 0
    score = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel == 1:
            hits += 1
            score += hits / i
    denom = total_relevant if total_relevant else hits
    return score / denom if denom else 0.0


def reciprocal_rank(relevances: Sequence[int]) -> float:
    for i, rel in enumerate(relevances, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    head = list(relevances[:k])
    if not head:
        return 0.0
    return sum(head) / len(head)


def recall_at_k(relevances: Sequence[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    head = list(relevances[:k])
    if not head:
        return 0.0
    gains = [(2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(head)]
    dcg = sum(gains)
    ideal = sorted(head, reverse=True)
    igains = [(2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal)]
    idcg = sum(igains)
    return dcg / idcg if idcg > 0 else 0.0
