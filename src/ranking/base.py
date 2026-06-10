"""Ranker contract.

Input: a list of *result lists*, one per query vector. Each result list is a
list of `Hit(label, score, doc_id)` ordered by score (high-is-better).
Output: a final, deduplicated, score-ordered list of `RankedResult(label, score)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Hit:
    label: str
    score: float
    doc_id: int
    rank: int = 0   # 1-based position in this query's result list


@dataclass
class RankedResult:
    label: str
    score: float


class BaseRanker(ABC):
    """Stateless. Same instance is reused across all queries in an experiment."""

    def set_label_meta(self, label_meta) -> None:
        """Optional hook for rankers that use label-level metadata."""
        return None

    @abstractmethod
    def rank(self, per_query_hits: Sequence[Sequence[Hit]], k: int) -> List[RankedResult]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ------------------ helpers shared by concrete rankers ------------------ #
def _take_top_k(scores: dict[str, float], k: int) -> List[RankedResult]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [RankedResult(label=lbl, score=float(s)) for lbl, s in items[:k]]
