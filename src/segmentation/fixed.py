"""Non-overlapping fixed-length windows. Native input format for Perch v2 (5s @ 32kHz)."""

from __future__ import annotations

from typing import List

import numpy as np

from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter, Segment


@SEGMENTERS.register("fixed")
class FixedWindowSegmenter(BaseSegmenter):
    def __init__(self, window_sec: float = 5.0, drop_last_short: bool = False,
                 pad_last: bool = True):
        self.window_sec = window_sec
        self.drop_last_short = drop_last_short
        self.pad_last = pad_last

    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        if audio.size == 0:
            return []
        win = int(round(self.window_sec * sample_rate))
        out: list[Segment] = []
        for start in range(0, audio.size, win):
            end = min(start + win, audio.size)
            chunk = audio[start:end]
            if chunk.size < win:
                if self.drop_last_short:
                    continue
                if self.pad_last:
                    chunk = np.pad(chunk, (0, win - chunk.size))
                    end = start + win
            out.append(Segment(start, end, chunk.astype(np.float32), self.window_sec))
        return out
