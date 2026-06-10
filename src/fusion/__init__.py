"""Query-side fusion strategies.

`early`: pool all query segment embeddings into one query vector → 1 search.
`late`:  search with each query segment, hand all hit-lists to the ranker.
"""

from src.utils.registry import Registry

FUSIONS = Registry("fusions")

from src.fusion import early, late  # noqa: F401,E402
from src.fusion.base import BaseFusion  # noqa: E402

__all__ = ["FUSIONS", "BaseFusion"]
