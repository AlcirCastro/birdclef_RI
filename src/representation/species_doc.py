"""One document per species. All segments from all recordings are pooled."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.representation import REPRESENTATIONS
from src.representation.base import BaseDocumentStore, Document, RecordEmbeddings, load_record_embeddings


@REPRESENTATIONS.register("species")
class SpeciesDocumentStore(BaseDocumentStore):
    name = "species"

    def __init__(self, aggregator=None, cache_dir: Path | None = None):
        super().__init__(aggregator)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def build(self, items: Sequence[RecordEmbeddings]) -> List[Document]:
        if self.aggregator is None:
            raise ValueError("SpeciesDocumentStore needs an aggregator")

        by_label: Dict[str, List[np.ndarray]] = {}
        label_meta: Dict[str, tuple[str, str]] = {}
        sources: Dict[str, List[int]] = {}
        for record, emb_source in items:
            mat = load_record_embeddings(emb_source)
            if mat.shape[0] == 0:
                continue
            by_label.setdefault(record.primary_label, []).append(mat)
            label_meta.setdefault(record.primary_label, (record.scientific_name, record.common_name))
            sources.setdefault(record.primary_label, []).append(record.item_id)

        self._docs = []
        for label, mats in by_label.items():
            stacked = np.concatenate(mats, axis=0)
            v = self.aggregator(stacked)
            self._docs.append(
                Document(
                    doc_id=len(self._docs),
                    label=label,
                    vector=v,
                    scientific_name=label_meta.get(label, ("", ""))[0],
                    common_name=label_meta.get(label, ("", ""))[1],
                    source_record_ids=sources[label],
                )
            )
        return self._docs
