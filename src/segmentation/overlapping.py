"""Sliding window with stride < window. Produces denser coverage of vocal events."""

from __future__ import annotations

from typing import List

import numpy as np

from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter, Segment


@SEGMENTERS.register("overlapping")
class OverlappingWindowSegmenter(BaseSegmenter):
    def __init__(self, window_sec: float = 5.0, hop_sec: float = 2.5, pad_last: bool = True):
        if hop_sec <= 0 or hop_sec > window_sec:
            raise ValueError("hop_sec must be in (0, window_sec]")
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.pad_last = pad_last

    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        if audio.size == 0:
            return []
        win = int(round(self.window_sec * sample_rate))
        hop = int(round(self.hop_sec * sample_rate))
        out: list[Segment] = []
        if audio.size <= win:
            chunk = audio
            if self.pad_last and chunk.size < win:
                chunk = np.pad(chunk, (0, win - chunk.size))
            return [Segment(0, win, chunk.astype(np.float32), self.window_sec)]
        last_start = audio.size - win
        for start in range(0, last_start + 1, hop):
            chunk = audio[start : start + win]
            out.append(Segment(start, start + win, chunk.astype(np.float32), self.window_sec))
        # Tail window if the loop didn't cover the very end.
        if out[-1].end_sample < audio.size:
            tail = audio[-win:]
            out.append(Segment(audio.size - win, audio.size, tail.astype(np.float32), self.window_sec))
        return out
