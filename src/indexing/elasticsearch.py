"""Elasticsearch dense_vector backend.

Stores per-doc embeddings in an ES index and serves kNN queries via the
`knn` search API. ES returns scores in [0, 1] for cosine similarity
(`(1 + cos)/2`); we convert back to raw cosine in [-1, 1] so callers
get the same higher-is-better range as the FAISS-based indexes.

Notes
-----
- ES 8.x dense_vector dim limit is 4096. Pair this index with aggregators
  that stay under that ceiling (mean/max/attention are 1280-dim for Perch v2;
  `spe` with default levels=[1,2,4] is 8960-dim and will not fit — drop to
  levels=[1,2] for ES, or use a FAISS index instead).
- Vectors are assumed L2-normalized upstream (Perch v2 + the aggregators
  here both normalize). The ES `cosine` similarity does not require it but
  preserves the contract of the FAISS path.
- save/load only persist the index name; the documents live on the ES
  cluster and survive across runs unless `recreate=True` (the default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from src.indexing import INDEXES
from src.indexing.base import BaseIndex


@INDEXES.register("es")
class ElasticsearchIndex(BaseIndex):
    def __init__(
        self,
        metric: str = "cosine",
        hosts: str | list[str] = "http://localhost:9200",
        index_name: str = "birdclef_embeddings",
        recreate: bool = True,
        bulk_chunk: int = 500,
        knn_num_candidates: int = 100,
        request_timeout: float = 60.0,
        api_key: str | None = None,
        basic_auth: tuple[str, str] | None = None,
    ):
        if metric != "cosine":
            raise ValueError("ElasticsearchIndex only supports metric='cosine'")
        self.metric = metric
        self.hosts = hosts
        self.index_name = index_name
        self.recreate = recreate
        self.bulk_chunk = bulk_chunk
        self.knn_num_candidates = knn_num_candidates
        self.request_timeout = request_timeout
        self._api_key = api_key
        self._basic_auth = tuple(basic_auth) if basic_auth else None
        self._es = None
        self._dim = 0
        self._n = 0

    def _client(self):
        if self._es is None:
            from elasticsearch import Elasticsearch
            kwargs: dict = {"request_timeout": self.request_timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._basic_auth:
                kwargs["basic_auth"] = self._basic_auth
            self._es = Elasticsearch(self.hosts, **kwargs)
        return self._es

    def _create_index(self, dim: int) -> None:
        es = self._client()
        if es.indices.exists(index=self.index_name):
            if not self.recreate:
                raise RuntimeError(
                    f"ES index {self.index_name!r} already exists and recreate=False"
                )
            es.indices.delete(index=self.index_name)
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        es.indices.create(index=self.index_name, **mapping)

    def build(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vectors")
        from elasticsearch.helpers import bulk

        self._dim = int(vectors.shape[1])
        self._n = int(vectors.shape[0])
        self._create_index(self._dim)
        es = self._client()

        def _actions():
            for i, v in enumerate(vectors):
                yield {
                    "_index": self.index_name,
                    "_id": str(i),
                    "_source": {"vector": v.astype(np.float32).tolist()},
                }

        bulk(es, _actions(), chunk_size=self.bulk_chunk, refresh="wait_for")

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._n == 0:
            raise RuntimeError("Index not built")
        es = self._client()
        m = int(queries.shape[0])
        k = min(k, self._n)
        scores = np.full((m, k), -np.inf, dtype=np.float32)
        ids = np.full((m, k), -1, dtype=np.int64)

        for qi, q in enumerate(queries):
            body = {
                "knn": {
                    "field": "vector",
                    "query_vector": q.astype(np.float32).tolist(),
                    "k": k,
                    "num_candidates": max(self.knn_num_candidates, k),
                },
                "_source": False,
                "size": k,
            }
            resp = es.search(index=self.index_name, **body)
            hits = resp["hits"]["hits"]
            for hi, hit in enumerate(hits):
                # ES cosine score is (1 + cos)/2 ∈ [0, 1]. Invert to raw cos ∈ [-1, 1]
                # so this matches FlatIndex/HNSWIndex output ranges.
                ids[qi, hi] = int(hit["_id"])
                scores[qi, hi] = float(hit["_score"]) * 2.0 - 1.0
        return scores, ids

    def size(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        return self._n * self._dim * 4

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "index_name": self.index_name,
            "n": self._n,
            "dim": self._dim,
        }))

    def load(self, path: Path) -> None:
        meta = json.loads(Path(path).read_text())
        self.index_name = meta["index_name"]
        self._n = int(meta["n"])
        self._dim = int(meta["dim"])
