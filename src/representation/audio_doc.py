"""Each audio file is a document. Aggregator pools its segments."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.representation import REPRESENTATIONS
from src.representation.base import BaseDocumentStore, Document, RecordEmbeddings, load_record_embeddings


@REPRESENTATIONS.register("audio")
class AudioDocumentStore(BaseDocumentStore):
    name = "audio"

    def __init__(self, aggregator=None, cache_dir: Path | None = None):
        super().__init__(aggregator)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _vector_store_path(self) -> Path:
        if self.cache_dir is None:
            raise ValueError("AudioDocumentStore needs cache_dir for low-memory build")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / "audio_vectors.npy"

    def build(self, items: Sequence[RecordEmbeddings]) -> List[Document]:
        if self.aggregator is None:
            raise ValueError("AudioDocumentStore needs an aggregator")

        entries = list(items)
        docs_count = 0
        dim = None
        for _, emb_source in entries:
            mat = load_record_embeddings(emb_source)
            if mat.shape[0] == 0:
                continue
            docs_count += 1
            if dim is None:
                dim = int(mat.shape[1])
        if docs_count == 0 or dim is None:
            self._docs = []
            self._vector_mm = None
            return self._docs

        path = self._vector_store_path()
        vectors = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(docs_count, dim))
        self._vector_mm = vectors
        self._vector_path = path
        self._docs = []
        row = 0
        for record, emb_source in entries:
            mat = load_record_embeddings(emb_source)
            if mat.shape[0] == 0:
                continue
            v = self.aggregator(mat)
            vectors[row] = v.astype(np.float32)
            self._docs.append(
                Document(
                    doc_id=row,
                    label=record.primary_label,
                    vector=vectors[row],
                    scientific_name=record.scientific_name,
                    common_name=record.common_name,
                    source_record_ids=[record.item_id],
                    source_segment_ids=list(range(mat.shape[0])),
                )
            )
            row += 1
        return self._docs
