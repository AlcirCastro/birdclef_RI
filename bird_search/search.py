from elasticsearch import Elasticsearch, helpers

import numpy as np

from bird_search.embedding import Embedder
from bird_search.models import Record
from bird_search.settings import Settings


class SearchIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.es = Elasticsearch(settings.es_url)

    def ensure_connected(self) -> None:
        try:
            self.es.info()
        except Exception as e:
            raise RuntimeError(f"Cannot reach Elasticsearch at {self.settings.es_url}: {e}")

    def create_index(self, dims: int, reset: bool = True) -> None:
        index = self.settings.es_index

        if self.es.indices.exists(index=index):
            if not reset:
                return
            self.es.indices.delete(index=index)

        self.es.indices.create(
            index=index,
            settings={"index": {"refresh_interval": "-1"}},
            mappings={
                "properties": {
                    "item_id": {"type": "integer"},
                    "primary_label": {"type": "keyword"},
                    "scientific_name": {"type": "text"},
                    "common_name": {"type": "text"},
                    "class_name": {"type": "keyword"},
                    "rating": {"type": "float"},
                    "local_path": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )

    def index_records(self, records: list[Record], embedder: Embedder, reset: bool = True) -> tuple[int, int]:
        sample = None
        for rec in records:
            sample = embedder.embed_path(rec.local_path)
            if sample is not None:
                break

        if sample is None:
            raise RuntimeError("Could not generate any embedding for index creation")

        self.create_index(dims=int(sample.shape[0]), reset=reset)

        actions = []
        batch_size = 500
        indexed = 0
        skipped = 0
        total = len(records)

        def flush() -> None:
            nonlocal actions
            if not actions:
                return
            helpers.bulk(self.es, actions, refresh=False, request_timeout=120)
            actions = []

        def log_progress(processed: int, total: int) -> None:
            if processed % batch_size == 0 or processed == total:
                pct = int((processed / total) * 100) if total > 0 else 0
                print(f"  Indexing: {processed}/{total} ({pct}%)")

        for rec in records:
            vec = embedder.embed_path(rec.local_path)
            if vec is None:
                skipped += 1
                log_progress(indexed + skipped, total)
                continue

            actions.append(
                {
                    "_index": self.settings.es_index,
                    "_id": rec.item_id,
                    "_source": {
                        "item_id": rec.item_id,
                        "primary_label": rec.primary_label,
                        "scientific_name": rec.scientific_name,
                        "common_name": rec.common_name,
                        "class_name": rec.class_name,
                        "rating": rec.rating,
                        "local_path": rec.local_path.as_posix(),
                        "embedding": vec.tolist(),
                    },
                }
            )
            indexed += 1
            log_progress(indexed + skipped, total)

            if len(actions) >= batch_size:
                flush()

        if actions:
            flush()

        if indexed > 0:
            self.es.indices.put_settings(index=self.settings.es_index, body={"index": {"refresh_interval": "1s"}})
            self.es.indices.refresh(index=self.settings.es_index)

        return indexed, skipped

    def knn_search(
        self,
        vec: np.ndarray,
        k: int | None = None,
        exclude_item_id: int | None = None,
    ) -> list[dict]:
        query_k = k if k is not None else self.settings.top_k
        fetch_k = query_k + 1 if exclude_item_id is not None else query_k

        resp = self.es.search(
            index=self.settings.es_index,
            knn={
                "field": "embedding",
                "query_vector": vec.tolist(),
                "k": fetch_k,
                "num_candidates": max(100, fetch_k * 10),
            },
            size=fetch_k,
        )

        hits = []
        for hit in resp["hits"]["hits"]:
            source = hit.get("_source", {})
            if exclude_item_id is not None and source.get("item_id") == exclude_item_id:
                continue
            hits.append(hit)
            if len(hits) >= query_k:
                break

        return hits