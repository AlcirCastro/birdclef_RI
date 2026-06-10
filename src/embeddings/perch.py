"""Google Perch v2 (bird vocalization classifier) — the project's only embedder.

Loaded lazily from Kaggle Models via `kagglehub` (preferred) or directly via
TF-Hub. Native input is 5 s of mono audio at 32 kHz; native output is a
1280-dim embedding vector. We expose just the embedding head — classifier
logits are ignored.

Segments shorter than 5 s are zero-padded; longer segments are center-cropped
(the segmentation stage is the right place to control window length).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from src.embeddings import EMBEDDERS
from src.embeddings.base import BaseEmbedder
from src.segmentation.base import Segment

NATIVE_SR = 32_000
NATIVE_WIN_SEC = 5.0
NATIVE_WIN_SAMPLES = int(NATIVE_SR * NATIVE_WIN_SEC)
DEFAULT_DIM = 1280

# Default Kaggle model handle for Perch v2. Newer versions may exist; configurable.
DEFAULT_KAGGLE_MODEL = "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"
DEFAULT_KAGGLE_VERSION: int | None = None  # use latest


@EMBEDDERS.register("perch_v2")
class PerchV2Embedder(BaseEmbedder):
    def __init__(
        self,
        kaggle_model: str = DEFAULT_KAGGLE_MODEL,
        kaggle_version: int | None = DEFAULT_KAGGLE_VERSION,
        tfhub_url: str | None = None,
        batch_size: int = 16,
        embedding_signature: str = "embedding",
    ):
        self.kaggle_model = kaggle_model
        self.kaggle_version = kaggle_version
        self.tfhub_url = tfhub_url
        self.batch_size = batch_size
        self.embedding_signature = embedding_signature
        self._model = None
        self._dim = DEFAULT_DIM

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def native_sample_rate(self) -> int:
        return NATIVE_SR

    @property
    def native_window_sec(self) -> float:
        return NATIVE_WIN_SEC

    @property
    def name(self) -> str:
        v = f"@{self.kaggle_version}" if self.kaggle_version is not None else ""
        return f"perch_v2[{self.kaggle_model}{v}]"

    # ------------------------------------------------------------------ #
    def _load(self):
        if self._model is not None:
            return
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf  # noqa: F401
            import tensorflow_hub as hub
        except ImportError as e:
            raise ImportError(
                "Perch v2 requires tensorflow and tensorflow_hub.\n"
                "  pip install tensorflow tensorflow_hub kagglehub"
            ) from e

        if self.tfhub_url:
            self._model = hub.load(self.tfhub_url)
        else:
            try:
                import kagglehub
            except ImportError as e:
                raise ImportError(
                    "Install kagglehub or set tfhub_url:  pip install kagglehub"
                ) from e
            handle = self.kaggle_model
            if self.kaggle_version is not None:
                handle = f"{handle}/{self.kaggle_version}"
            local = kagglehub.model_download(handle)
            self._model = hub.load(local)

    def _normalize(self, segment_audio: np.ndarray) -> np.ndarray:
        """Pad or center-crop to the native window length."""
        if segment_audio.size == NATIVE_WIN_SAMPLES:
            return segment_audio.astype(np.float32)
        if segment_audio.size < NATIVE_WIN_SAMPLES:
            pad = NATIVE_WIN_SAMPLES - segment_audio.size
            return np.pad(segment_audio, (0, pad)).astype(np.float32)
        start = (segment_audio.size - NATIVE_WIN_SAMPLES) // 2
        return segment_audio[start : start + NATIVE_WIN_SAMPLES].astype(np.float32)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """Run the model on a (B, NATIVE_WIN_SAMPLES) batch → (B, dim)."""
        import tensorflow as tf
        x = tf.constant(batch, dtype=tf.float32)

        # Most TF-Hub Perch builds expose `infer_tf` returning a dict with
        # 'embedding' (and 'label_logits'). Some expose a callable directly.
        infer = getattr(self._model, "infer_tf", None) or self._model
        out = infer(x)
        if isinstance(out, (tuple, list)):
            emb = out[0]  # convention: (logits, embedding) — guard below.
            for o in out:
                if hasattr(o, "shape") and len(o.shape) == 2 and int(o.shape[-1]) >= 512:
                    emb = o
                    break
        elif hasattr(out, "keys"):
            if self.embedding_signature in out:
                emb = out[self.embedding_signature]
            else:
                emb = next(iter(out.values()))
        else:
            emb = out
        return np.asarray(emb).astype(np.float32)

    # ------------------------------------------------------------------ #
    def embed_segments(self, segments: Sequence[Segment]) -> np.ndarray:
        if not segments:
            return np.zeros((0, self._dim), dtype=np.float32)

        self._load()
        batch_audio = np.stack([self._normalize(s.audio) for s in segments]).astype(np.float32)

        outputs: list[np.ndarray] = []
        for i in range(0, batch_audio.shape[0], self.batch_size):
            chunk = batch_audio[i : i + self.batch_size]
            outputs.append(self._forward(chunk))
        emb = np.concatenate(outputs, axis=0)

        if emb.shape[1] != self._dim:
            self._dim = int(emb.shape[1])

        # L2-normalize so cosine similarity reduces to inner product everywhere.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (emb / norms).astype(np.float32)
