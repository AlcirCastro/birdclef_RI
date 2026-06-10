"""Butterworth bandpass — keep typical bird vocalization range, drop rumble + hiss."""

from __future__ import annotations

import numpy as np

from src.preprocessing import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor


@PREPROCESSORS.register("bandpass")
class BandpassPreprocessor(BasePreprocessor):
    def __init__(self, low_hz: float = 250.0, high_hz: float = 12_000.0, order: int = 4):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return audio
        from scipy.signal import butter, sosfiltfilt
        nyq = 0.5 * sample_rate
        low = max(self.low_hz / nyq, 1e-4)
        high = min(self.high_hz / nyq, 0.999)
        if low >= high:
            return audio.astype(np.float32)
        sos = butter(self.order, [low, high], btype="band", output="sos")
        return sosfiltfilt(sos, audio).astype(np.float32)
