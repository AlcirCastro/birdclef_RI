"""Late fusion: keep each query segment as its own query — fusion happens in the ranker."""

from __future__ import annotations

import numpy as np

from src.fusion import FUSIONS
from src.fusion.base import BaseFusion


@FUSIONS.register("late")
class LateFusion(BaseFusion):
    name = "late"

    def queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        return query_embeddings.astype(np.float32)
