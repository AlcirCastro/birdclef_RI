"""Concatenate fixed segmentations at several window sizes — short windows catch
short calls, long windows give Perch v2 its native 5 s context."""

from __future__ import annotations

from typing import List

import numpy as np

from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter, Segment
from src.segmentation.fixed import FixedWindowSegmenter


@SEGMENTERS.register("multi_scale")
class MultiScaleSegmenter(BaseSegmenter):
    def __init__(self, scales_sec: List[float] = (3.0, 5.0, 10.0),
                 drop_last_short: bool = False, pad_last: bool = True):
        self.scales_sec = list(scales_sec)
        self.children = [
            FixedWindowSegmenter(window_sec=s, drop_last_short=drop_last_short,
                                 pad_last=pad_last)
            for s in self.scales_sec
        ]

    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Segment]:
        out: list[Segment] = []
        for child, scale in zip(self.children, self.scales_sec):
            for seg in child.segment(audio, sample_rate):
                seg.scale = scale
                out.append(seg)
        return out
