from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePreprocessor(ABC):
    """Audio → audio. Same sample rate in and out."""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
