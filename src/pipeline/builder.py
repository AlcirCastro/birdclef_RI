"""Config → component instances. Single source of truth for stage→registry mapping."""

from __future__ import annotations

from src.aggregation import AGGREGATORS
from src.config.schema import ExperimentConfig, StageConfig
from src.embeddings import EMBEDDERS
from src.embeddings.base import BaseEmbedder
from src.fusion import FUSIONS
from src.fusion.base import BaseFusion
from src.indexing import INDEXES
from src.indexing.base import BaseIndex
from src.preprocessing import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor
from src.ranking import RANKERS
from src.ranking.base import BaseRanker
from src.representation import REPRESENTATIONS
from src.representation.base import BaseDocumentStore
from src.segmentation import SEGMENTERS
from src.segmentation.base import BaseSegmenter


def _build(stage: StageConfig, registry):
    cls = registry.get(stage.type)
    return cls(**(stage.params or {}))


class PipelineBuilder:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def preprocessor(self) -> BasePreprocessor:
        return _build(self.cfg.preprocessing, PREPROCESSORS)

    def segmenter(self) -> BaseSegmenter:
        return _build(self.cfg.segmentation, SEGMENTERS)

    def embedder(self) -> BaseEmbedder:
        return _build(self.cfg.embedding, EMBEDDERS)

    def aggregator(self):
        return _build(self.cfg.aggregation, AGGREGATORS)

    def doc_store(self) -> BaseDocumentStore:
        cls = REPRESENTATIONS.get(self.cfg.representation.type)
        agg = self.aggregator()
        # Cluster store wants extras like n_clusters; pass them through params.
        params = dict(self.cfg.representation.params or {})
        params.setdefault("cache_dir", self.cfg.output.cache_dir / "documents" / self.cfg.name)
        return cls(aggregator=agg, **params)

    def index(self) -> BaseIndex:
        params = dict(self.cfg.indexing.params or {})
        params.setdefault("metric", self.cfg.similarity)
        cls = INDEXES.get(self.cfg.indexing.type)
        return cls(**params)

    def fusion(self) -> BaseFusion:
        return _build(self.cfg.fusion, FUSIONS)

    def ranker(self) -> BaseRanker:
        return _build(self.cfg.ranking, RANKERS)
