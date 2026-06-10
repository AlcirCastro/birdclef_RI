from __future__ import annotations

import numpy as np

from src.preprocessing import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor


@PREPROCESSORS.register("none")
class IdentityPreprocessor(BasePreprocessor):
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio.astype(np.float32)
