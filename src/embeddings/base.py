from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from src.segmentation.base import Segment


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def native_sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def native_window_sec(self) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier — part of the cache key."""

    @abstractmethod
    def embed_segments(self, segments: Sequence[Segment]) -> np.ndarray:
        """Return a (len(segments), dim) array of L2-normalized embeddings."""
