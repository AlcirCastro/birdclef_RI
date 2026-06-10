"""Single prototype vector per species: the segment closest to the species centroid.

Cheaper than cluster representation, more selective than mean pooling. Useful
as a baseline when k-prototype variants are too expensive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.representation import REPRESENTATIONS
from src.representation.base import BaseDocumentStore, Document, RecordEmbeddings, load_record_embeddings


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)


@REPRESENTATIONS.register("prototype")
class PrototypeSpeciesDocumentStore(BaseDocumentStore):
    name = "prototype"

    def __init__(self, aggregator=None, cache_dir: Path | None = None):
        super().__init__(aggregator)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def build(self, items: Sequence[RecordEmbeddings]) -> List[Document]:
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
            X = np.concatenate(mats, axis=0)
            centroid = _l2(X.mean(axis=0))
            sims = X @ centroid
            best = X[int(np.argmax(sims))]
            self._docs.append(
                Document(
                    doc_id=len(self._docs),
                    label=label,
                    vector=_l2(best),
                    scientific_name=label_meta.get(label, ("", ""))[0],
                    common_name=label_meta.get(label, ("", ""))[1],
                    source_record_ids=sources[label],
                )
            )
        return self._docs
