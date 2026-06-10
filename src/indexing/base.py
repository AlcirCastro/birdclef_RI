"""Index interface.

`build` ingests a (n, d) matrix; `search` returns (scores, doc_ids) for
each query in a (m, d) batch. Scores are *higher-is-better*: cosine sim
when `metric="cosine"`, negative L2 when `metric="l2"`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np


class BaseIndex(ABC):
    metric: str = "cosine"

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids) of shape (m, k). Missing slots filled with -1 / -inf."""

    @abstractmethod
    def size(self) -> int:
        ...

    def memory_bytes(self) -> int:
        """Approximate index footprint. Subclasses can override."""
        return 0

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
