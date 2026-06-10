"""Vector indexing backends. All FAISS-based."""

from src.utils.registry import Registry

INDEXES = Registry("indexes")

from src.indexing import flat, ivf, hnsw, pq, elasticsearch  # noqa: F401,E402
from src.indexing.base import BaseIndex  # noqa: E402

__all__ = ["INDEXES", "BaseIndex"]
