"""Embedding stage.

Perch v2 remains available for the original experiments, and BirdNET+ V3.0
is registered as an alternate embedder for the same retrieval pipeline.
"""

from src.utils.registry import Registry

EMBEDDERS = Registry("embedders")

from src.embeddings import birdnet_v3  # noqa: F401,E402
from src.embeddings import perch  # noqa: F401,E402
from src.embeddings import perch_torch  # noqa: F401,E402
from src.embeddings.base import BaseEmbedder  # noqa: E402
from src.embeddings.cache import EmbeddingCache  # noqa: E402

__all__ = ["EMBEDDERS", "BaseEmbedder", "EmbeddingCache"]
