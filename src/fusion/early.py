"""Early fusion: collapse the query side first, then issue a single search."""

from __future__ import annotations

import numpy as np

from src.fusion import FUSIONS
from src.fusion.base import BaseFusion
from src.aggregation import AGGREGATORS


@FUSIONS.register("early")
class EarlyFusion(BaseFusion):
    name = "early"

    def __init__(self, aggregator: str = "mean", **agg_params):
        self.aggregator = AGGREGATORS.get(aggregator)(**agg_params)

    def queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        if query_embeddings.size == 0:
            return query_embeddings
        v = self.aggregator(query_embeddings)
        return v.reshape(1, -1).astype(np.float32)
