"""BirdNET+ V3.0 embedder.

This adapter mirrors the existing embedder interface so the experiment
pipeline can swap from Perch v2 to BirdNET+ V3.0 through config only.
It loads the published TorchScript checkpoint, normalizes audio segments to
BirdNET's native 3 second / 32 kHz window, and returns L2-normalized
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

from src.embeddings import EMBEDDERS
from src.embeddings.base import BaseEmbedder
from src.segmentation.base import Segment

NATIVE_SR = 32_000
NATIVE_WIN_SEC = 3.0
NATIVE_WIN_SAMPLES = int(NATIVE_SR * NATIVE_WIN_SEC)
DEFAULT_DIM = 1280

DEFAULT_MODEL_URL = (
    "https://zenodo.org/records/18247420/files/"
    "BirdNET%2B_V3.0-preview3_Global_11K_FP32.pt?download=1"
)


@EMBEDDERS.register("birdnet_v3")
@dataclass
class BirdNetV3Embedder(BaseEmbedder):
    model_path: str = DEFAULT_MODEL_URL
    batch_size: int = 16
    device: str = "auto"
    cache_dir: str | Path = Path("birdclef_cache") / "birdnet_v3_models"

    _model: object | None = field(default=None, init=False, repr=False)
    _torch_device: object | None = field(default=None, init=False, repr=False)
    _dim: int = field(default=DEFAULT_DIM, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)

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
        return f"birdnet_v3[{self.model_path}]"

    def _normalize(self, segment_audio: np.ndarray) -> np.ndarray:
        if segment_audio.size == NATIVE_WIN_SAMPLES:
            return segment_audio.astype(np.float32)
        if segment_audio.size < NATIVE_WIN_SAMPLES:
            pad = NATIVE_WIN_SAMPLES - segment_audio.size
            return np.pad(segment_audio, (0, pad)).astype(np.float32)
        start = (segment_audio.size - NATIVE_WIN_SAMPLES) // 2
        return segment_audio[start : start + NATIVE_WIN_SAMPLES].astype(np.float32)

    def _model_filename(self) -> str:
        parsed = urlparse(self.model_path)
        if parsed.scheme in {"http", "https"}:
            name = Path(parsed.path).name
            if name:
                return name
        return Path(self.model_path).name

    def _resolve_model_path(self) -> Path:
        model_ref = Path(self.model_path).expanduser()
        if model_ref.exists():
            return model_ref

        parsed = urlparse(self.model_path)
        if parsed.scheme not in {"http", "https"}:
            raise FileNotFoundError(f"BirdNET v3 model not found: {self.model_path}")

        target = self.cache_dir / self._model_filename()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urlretrieve(self.model_path, target)
        return target

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError("BirdNET v3 embedder requires torch.") from exc

        model_path = self._resolve_model_path()
        if self.device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(self.device)

        self._torch_device = torch_device
        model = torch.jit.load(str(model_path), map_location=torch_device)
        model.eval()
        self._model = model

    def _extract_embedding(self, output):
        if isinstance(output, dict):
            if "embeddings" in output:
                return output["embeddings"]
            if "embedding" in output:
                return output["embedding"]
            return next(iter(output.values()))

        if isinstance(output, (tuple, list)):
            if len(output) == 2:
                first, second = output
                for candidate in (first, second):
                    shape = getattr(candidate, "shape", None)
                    if shape is not None and len(shape) == 2 and int(shape[-1]) == DEFAULT_DIM:
                        return candidate
                return first

            for candidate in output:
                shape = getattr(candidate, "shape", None)
                if shape is not None and len(shape) == 2 and int(shape[-1]) == DEFAULT_DIM:
                    return candidate
            return output[0]

        return output

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        import torch

        x = torch.as_tensor(batch, dtype=torch.float32, device=self._torch_device)
        with torch.inference_mode():
            output = self._model(x)

        emb = self._extract_embedding(output)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        return np.asarray(emb, dtype=np.float32)

    def embed_segments(self, segments: Sequence[Segment]) -> np.ndarray:
        if not segments:
            return np.zeros((0, self._dim), dtype=np.float32)

        self._load()
        batch_audio = np.stack([self._normalize(segment.audio) for segment in segments]).astype(np.float32)

        outputs: list[np.ndarray] = []
        for start in range(0, batch_audio.shape[0], self.batch_size):
            chunk = batch_audio[start : start + self.batch_size]
            outputs.append(self._forward(chunk))

        emb = np.concatenate(outputs, axis=0)
        if emb.shape[1] != self._dim:
            self._dim = int(emb.shape[1])

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (emb / norms).astype(np.float32)