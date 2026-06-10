"""Event-based segmentation via simple onset detection.

Uses librosa's spectral-flux onset envelope to locate vocal events,
then crops a fixed window around each onset. A more elaborate detector
(e.g., a CNN voice-activity model) can be dropped in by registering a
new key — this implementation is the cheap, classical baseline.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter, Segment


@SEGMENTERS.register("event")
class EventBasedSegmenter(BaseSegmenter):
    def __init__(self, window_sec: float = 5.0, hop_sec: float = 0.05,
                 max_segments: int = 8, min_gap_sec: float = 1.0):
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.max_segments = max_segments
        self.min_gap_sec = min_gap_sec

    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        if audio.size == 0:
            return []
        win = int(round(self.window_sec * sample_rate))
        if audio.size <= win:
            return [Segment(0, win, np.pad(audio, (0, max(0, win - audio.size))).astype(np.float32),
                            self.window_sec)]

        import librosa
        hop_length = max(1, int(round(self.hop_sec * sample_rate)))
        onsets = librosa.onset.onset_detect(
            y=audio, sr=sample_rate, hop_length=hop_length, units="samples", backtrack=True
        )
        if onsets.size == 0:
            # Fall back to a single central window.
            start = max(0, audio.size // 2 - win // 2)
            return [Segment(start, start + win, audio[start : start + win].astype(np.float32),
                            self.window_sec)]

        min_gap = int(round(self.min_gap_sec * sample_rate))
        placed: list[tuple[int, int]] = []
        for o in onsets:
            start = max(0, min(audio.size - win, int(o) - win // 4))
            end = start + win
            if placed and (start - placed[-1][0]) < min_gap:
                continue
            placed.append((start, end))
            if len(placed) >= self.max_segments:
                break

        return [
            Segment(s, e, audio[s:e].astype(np.float32), self.window_sec)
            for s, e in placed
        ]
