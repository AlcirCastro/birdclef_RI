"""Stateless query orchestrator.

Holds (index, doc_store, fusion, ranker) and turns a query embedding matrix
into a final ranked list of species. `candidate_factor` controls how many
raw hits per query we ask the index for before label-aggregation; the final
list is truncated to `top_k`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.fusion.base import BaseFusion
from src.indexing.base import BaseIndex
from src.ranking.base import BaseRanker, Hit, RankedResult
from src.representation.base import BaseDocumentStore
from src.utils.taxonomy import taxonomy_info


class Retriever:
    def __init__(
        self,
        index: BaseIndex,
        doc_store: BaseDocumentStore,
        fusion: BaseFusion,
        ranker: BaseRanker,
        candidate_factor: int = 10,
        max_candidates: int = 200,
    ):
        self.index = index
        self.doc_store = doc_store
        self.fusion = fusion
        self.ranker = ranker
        self.candidate_factor = candidate_factor
        self.max_candidates = max_candidates
        self.label_meta = {
            doc.label: taxonomy_info(doc.scientific_name, doc.common_name)
            for doc in doc_store.documents
        }
        if hasattr(self.ranker, "set_label_meta"):
            self.ranker.set_label_meta(self.label_meta)

    # ------------------------------------------------------------------ #
    def retrieve(self, query_embeddings: np.ndarray, top_k: int) -> List[RankedResult]:
        if self.index.size() == 0:
            return []
        if query_embeddings.size == 0:
            return []

        queries = self.fusion.queries(query_embeddings)
        per_q_k = min(self.max_candidates, max(top_k, top_k * self.candidate_factor))
        per_q_k = min(per_q_k, self.index.size())
        scores, ids = self.index.search(queries, per_q_k)

        labels = self.doc_store.labels
        per_query_hits: List[List[Hit]] = []
        for q_scores, q_ids in zip(scores, ids):
            hits: List[Hit] = []
            rank = 0
            for s, doc_id in zip(q_scores, q_ids):
                if doc_id < 0 or doc_id >= len(labels):
                    continue
                rank += 1
                hits.append(
                    Hit(label=labels[doc_id], score=float(s), doc_id=int(doc_id), rank=rank)
                )
            per_query_hits.append(hits)

        return self.ranker.rank(per_query_hits, top_k)
