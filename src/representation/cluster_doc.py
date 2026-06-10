"""K-means cluster centroids per species — captures multimodal call repertoires.

If a species has fewer segments than `n_clusters`, falls back to one
centroid per segment (no clustering).
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


@REPRESENTATIONS.register("cluster")
class ClusterSpeciesDocumentStore(BaseDocumentStore):
    name = "cluster"

    def __init__(self, aggregator=None, n_clusters: int = 4, random_state: int = 0, cache_dir: Path | None = None):
        super().__init__(aggregator)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def build(self, items: Sequence[RecordEmbeddings]) -> List[Document]:
        from sklearn.cluster import KMeans

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
            n = X.shape[0]
            k = min(self.n_clusters, n)
            if k <= 1:
                centers = X.mean(axis=0, keepdims=True)
            else:
                km = KMeans(n_clusters=k, n_init=4, random_state=self.random_state)
                km.fit(X)
                centers = km.cluster_centers_
            for c in centers:
                self._docs.append(
                    Document(
                        doc_id=len(self._docs),
                        label=label,
                        vector=_l2(c),
                        scientific_name=label_meta.get(label, ("", ""))[0],
                        common_name=label_meta.get(label, ("", ""))[1],
                        source_record_ids=sources[label],
                    )
                )
        return self._docs
