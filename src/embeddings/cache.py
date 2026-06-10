"""On-disk embedding cache.

Cache key is `sha1(audio_path | size | mtime | embedder.name | preproc_id |
segmenter_id)` — any change in upstream pipeline forces re-embedding.
Stored as a single .npz per audio file holding the (n_segments, dim) matrix
plus segment offsets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.embeddings.base import BaseEmbedder
from src.segmentation.base import Segment


@dataclass(frozen=True)
class CacheKey:
    audio_path: Path
    embedder_name: str
    preproc_id: str
    segmenter_id: str

    def digest(self) -> str:
        st = self.audio_path.stat()
        payload = "|".join([
            self.audio_path.as_posix(),
            str(st.st_size),
            str(int(st.st_mtime)),
            self.embedder_name,
            self.preproc_id,
            self.segmenter_id,
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir: Path, embedder: BaseEmbedder,
                 preproc_id: str, segmenter_id: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.preproc_id = preproc_id
        self.segmenter_id = segmenter_id
        self.enabled = enabled

    def _key(self, audio_path: Path) -> CacheKey:
        return CacheKey(audio_path, self.embedder.name, self.preproc_id, self.segmenter_id)

    def _path(self, audio_path: Path) -> Path:
        return self.cache_dir / f"{self._key(audio_path).digest()}.npz"

    def path_for(self, audio_path: Path) -> Path:
        """Return the on-disk cache path for an audio file."""
        return self._path(audio_path)

    def load(self, audio_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.enabled:
            return None
        p = self._path(audio_path)
        if not p.exists():
            return None
        with np.load(p) as z:
            return z["embeddings"].astype(np.float32), z["offsets"].astype(np.int64)

    def save(self, audio_path: Path, embeddings: np.ndarray, segments: List[Segment]) -> Path:
        if not self.enabled:
            return self._path(audio_path)
        offsets = np.array([[s.start_sample, s.end_sample] for s in segments], dtype=np.int64)
        path = self._path(audio_path)
        np.savez_compressed(path, embeddings=embeddings.astype(np.float32), offsets=offsets)
        return path
