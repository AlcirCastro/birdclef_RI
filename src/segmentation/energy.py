"""Energy-based segmentation: extract fixed-length windows centered on
high-energy frames, skipping silent stretches.

Frame energy is computed on a hop grid; we threshold at the
`percentile`-th percentile of the per-frame RMS, then place a window of
`window_sec` around each retained frame. Overlapping windows are merged.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter, Segment


@SEGMENTERS.register("energy")
class EnergyBasedSegmenter(BaseSegmenter):
    def __init__(self, window_sec: float = 5.0, frame_sec: float = 0.5,
                 percentile: float = 60.0, max_segments: int = 8):
        self.window_sec = window_sec
        self.frame_sec = frame_sec
        self.percentile = percentile
        self.max_segments = max_segments

    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        if audio.size == 0:
            return []
        win = int(round(self.window_sec * sample_rate))
        if audio.size <= win:
            chunk = np.pad(audio, (0, max(0, win - audio.size)))
            return [Segment(0, win, chunk.astype(np.float32), self.window_sec)]

        frame = max(1, int(round(self.frame_sec * sample_rate)))
        n_frames = audio.size // frame
        rms = np.sqrt(np.mean(
            audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1
        ))
        if rms.size == 0:
            return []

        thresh = np.percentile(rms, self.percentile)
        peaks = np.where(rms >= thresh)[0]
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(rms))])

        # Order by descending energy, then place windows greedily without overlap.
        order = peaks[np.argsort(-rms[peaks])]
        placed: list[tuple[int, int]] = []
        for idx in order:
            center = idx * frame + frame // 2
            start = max(0, min(audio.size - win, center - win // 2))
            end = start + win
            if any(not (end <= s or start >= e) for s, e in placed):
                continue
            placed.append((start, end))
            if len(placed) >= self.max_segments:
                break

        placed.sort()
        return [
            Segment(s, e, audio[s:e].astype(np.float32), self.window_sec)
            for s, e in placed
        ]
