"""Document-store strategies: how the corpus is broken into searchable units."""

from src.utils.registry import Registry

REPRESENTATIONS = Registry("representations")

from src.representation import segment_doc, audio_doc, species_doc, cluster_doc, prototype_doc  # noqa: F401,E402
from src.representation.base import Document, BaseDocumentStore  # noqa: E402

__all__ = ["REPRESENTATIONS", "Document", "BaseDocumentStore"]
