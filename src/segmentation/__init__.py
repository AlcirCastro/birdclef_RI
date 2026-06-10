"""Audio segmentation strategies. Each yields a list of (start_sample, end_sample) windows."""

from src.utils.registry import Registry

SEGMENTERS = Registry("segmenters")

from src.segmentation import fixed, overlapping, multiscale, energy, event  # noqa: F401,E402
from src.segmentation.base import BaseSegmenter, Segment  # noqa: E402

__all__ = ["SEGMENTERS", "BaseSegmenter", "Segment"]
