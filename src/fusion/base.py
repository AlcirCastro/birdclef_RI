"""Fusion contract: turn a per-query embedding matrix into the queries actually issued."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFusion(ABC):
    name: str = "base"

    @abstractmethod
    def queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        """`query_embeddings` is (n_segments, dim). Return (m, dim) — the m queries to run."""
