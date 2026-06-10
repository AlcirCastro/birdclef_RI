"""Document model + abstract store.

A Document is whatever the index searches over. Its `label` is always the
species (the retrieval target). `source_record_ids` and `source_segment_ids`
let the evaluator and visualizers trace a hit back to specific audio clips.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.aggregation.pooling import BaseAggregator
from src.data.records import Record


@dataclass
class Document:
    doc_id: int
    label: str                    # primary_label / species code
    vector: np.ndarray            # (dim,), L2-normalized
    scientific_name: str = ""
    common_name: str = ""
    source_record_ids: List[int] = field(default_factory=list)
    source_segment_ids: List[int] = field(default_factory=list)


# Per-record bundle the doc-store consumes.
RecordEmbeddings = Tuple[Record, np.ndarray | Path]  # (record, embeddings matrix or cache path)


def load_record_embeddings(value: np.ndarray | Path) -> np.ndarray:
    if isinstance(value, Path):
        with np.load(value) as z:
            return z["embeddings"].astype(np.float32)
    return np.asarray(value, dtype=np.float32)


class BaseDocumentStore(ABC):
    name: str = "base"

    def __init__(self, aggregator: BaseAggregator | None = None):
        self.aggregator = aggregator
        self._docs: List[Document] = []
        self._vector_mm: np.ndarray | None = None
        self._vector_path: Path | None = None

    @abstractmethod
    def build(self, items: Sequence[RecordEmbeddings]) -> List[Document]:
        ...

    @property
    def documents(self) -> List[Document]:
        return self._docs

    @property
    def vectors(self) -> np.ndarray:
        if self._vector_mm is not None:
            return self._vector_mm
        if not self._docs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([d.vector for d in self._docs], axis=0).astype(np.float32)

    @property
    def labels(self) -> List[str]:
        return [d.label for d in self._docs]

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterable[Document]:
        return iter(self._docs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self._docs)})"
