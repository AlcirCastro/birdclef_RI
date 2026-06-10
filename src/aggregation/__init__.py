"""Vector aggregation. (n_segments, dim) → (dim,).

Used both to build species/audio-level documents and to early-fuse query embeddings.
"""

from src.utils.registry import Registry

AGGREGATORS = Registry("aggregators")

from src.aggregation import pooling  # noqa: F401,E402

__all__ = ["AGGREGATORS"]
