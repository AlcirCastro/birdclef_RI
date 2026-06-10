"""Rankers: turn raw hits from one or more queries into a final per-label ranking."""

from src.utils.registry import Registry

RANKERS = Registry("rankers")

from src.ranking import basic, advanced  # noqa: F401,E402
from src.ranking.base import BaseRanker, Hit, RankedResult  # noqa: E402

__all__ = ["RANKERS", "BaseRanker", "Hit", "RankedResult"]
