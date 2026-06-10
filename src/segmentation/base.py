from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Segment:
    start_sample: int
    end_sample: int
    audio: np.ndarray  # the actual sliced signal
    scale: float = 1.0  # window length seconds; multi-scale tags this

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


class BaseSegmenter(ABC):
    """Audio → list of Segments. All segments share the parent's sample rate."""

    @abstractmethod
    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
