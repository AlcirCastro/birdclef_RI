"""Experimental PyTorch backend for the official Google Perch v2 model.

This path keeps the source of truth as the published Perch v2 checkpoint, but
exports the SavedModel to ONNX once and executes the converted graph with
PyTorch thereafter.

The export step still needs a TensorFlow-capable environment the first time the
cache is populated. If the conversion fails because the published graph uses
unsupported ops, the embedder falls back to running the official SavedModel
directly and returns NumPy outputs to the rest of the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from src.embeddings import EMBEDDERS
from src.embeddings.base import BaseEmbedder
from src.segmentation.base import Segment

NATIVE_SR = 32_000
NATIVE_WIN_SEC = 5.0
NATIVE_WIN_SAMPLES = int(NATIVE_SR * NATIVE_WIN_SEC)
DEFAULT_DIM = 1280

DEFAULT_KAGGLE_MODEL = "google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier"
DEFAULT_KAGGLE_VERSION: int | None = None


@EMBEDDERS.register("perch_v2_torch")
@dataclasses.dataclass
class PerchV2TorchEmbedder(BaseEmbedder):
    kaggle_model: str = DEFAULT_KAGGLE_MODEL
    kaggle_version: int | None = DEFAULT_KAGGLE_VERSION
    tfhub_url: str | None = None
    batch_size: int = 16
    embedding_signature: str = "embedding"
    model_cache_dir: str | Path = Path("birdclef_cache") / "torch_models"
    onnx_opset: int = 17
    device: str = "auto"
    devices: tuple[str, ...] | None = None

    _model: object | None = dataclasses.field(default=None, init=False, repr=False)
    _replicas: list[object] = dataclasses.field(default_factory=list, init=False, repr=False)
    _runtime: str = dataclasses.field(default="torch", init=False, repr=False)
    _torch_device: object | None = dataclasses.field(default=None, init=False, repr=False)
    _torch_devices: list[object] = dataclasses.field(default_factory=list, init=False, repr=False)
    _dim: int = dataclasses.field(default=DEFAULT_DIM, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_cache_dir = Path(self.model_cache_dir)
        if self.devices is not None:
            self.devices = tuple(self.devices)

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
        version = f"@{self.kaggle_version}" if self.kaggle_version is not None else ""
        return f"perch_v2_torch[{self.kaggle_model}{version}]"

    def _normalize(self, segment_audio: np.ndarray) -> np.ndarray:
        if segment_audio.size == NATIVE_WIN_SAMPLES:
            return segment_audio.astype(np.float32)
        if segment_audio.size < NATIVE_WIN_SAMPLES:
            pad = NATIVE_WIN_SAMPLES - segment_audio.size
            return np.pad(segment_audio, (0, pad)).astype(np.float32)
        start = (segment_audio.size - NATIVE_WIN_SAMPLES) // 2
        return segment_audio[start : start + NATIVE_WIN_SAMPLES].astype(np.float32)

    def _model_id(self) -> str:
        version = f"@{self.kaggle_version}" if self.kaggle_version is not None else ""
        return f"{self.kaggle_model.replace('/', '__')}{version}"

    def _source_model_dir(self) -> Path:
        if self.tfhub_url:
            path = Path(self.tfhub_url)
            if path.exists():
                return path
            raise NotImplementedError(
                "The PyTorch backend currently expects a local SavedModel path or "
                "the standard Kaggle model handle."
            )

        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "Perch v2 torch export requires kagglehub for model download."
            ) from e

        handle = self.kaggle_model
        if self.kaggle_version is not None:
            handle = f"{handle}/{self.kaggle_version}"
        return Path(kagglehub.model_download(handle))

    def _onnx_path(self) -> Path:
        return self.model_cache_dir / self._model_id() / "model.onnx"

    def _export_onnx(self, source_model_dir: Path, onnx_path: Path) -> None:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(source_model_dir),
            "--output",
            str(onnx_path),
            "--opset",
            str(self.onnx_opset),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to export Perch v2 to ONNX.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    def _load_torch_model(self):
        try:
            import onnx
        except ImportError as e:
            raise ImportError(
                "Perch v2 torch backend requires `onnx`. Install it with `pip install onnx`."
            ) from e

        try:
            from onnx2torch import convert as onnx2torch_convert
        except ImportError as e:
            raise ImportError(
                "Perch v2 torch backend requires `onnx2torch`. Install it with `pip install onnx2torch`."
            ) from e

        onnx_path = self._onnx_path()
        if not onnx_path.exists():
            self._export_onnx(self._source_model_dir(), onnx_path)

        torch_model = onnx2torch_convert(onnx.load(str(onnx_path)))
        torch_model.eval()

        import torch

        if self.devices:
            torch_devices = [torch.device(d) for d in self.devices]
        elif self.device == "auto":
            if torch.cuda.is_available():
                n_cuda = torch.cuda.device_count()
                torch_devices = [torch.device(f"cuda:{i}") for i in range(n_cuda)] if n_cuda > 1 else [torch.device("cuda")]
            else:
                torch_devices = [torch.device("cpu")]
        else:
            torch_devices = [torch.device(self.device)]

        if not torch_devices:
            torch_devices = [torch.device("cpu")]

        self._torch_devices = torch_devices
        self._torch_device = torch_devices[0]
        torch_model.to(self._torch_device)

        self._replicas = [torch_model]
        if len(torch_devices) > 1:
            import copy

            self._replicas = [torch_model] + [copy.deepcopy(torch_model).to(device) for device in torch_devices[1:]]

        return torch_model

    def _load_tensorflow_model(self):
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "Perch v2 fallback runtime requires tensorflow."
            ) from e

        if self.tfhub_url:
            return tf.saved_model.load(self.tfhub_url)

        try:
            import kagglehub
        except ImportError as e:
            raise ImportError(
                "Perch v2 fallback runtime requires kagglehub for model download."
            ) from e

        handle = self.kaggle_model
        if self.kaggle_version is not None:
            handle = f"{handle}/{self.kaggle_version}"
        local = kagglehub.model_download(handle)
        return tf.saved_model.load(local)

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            self._model = self._load_torch_model()
            self._runtime = "torch"
        except Exception as exc:
            warnings.warn(
                "Perch v2 ONNX export failed; falling back to TensorFlow runtime "
                f"for this backend. Cause: {exc}",
                RuntimeWarning,
            )
            self._model = self._load_tensorflow_model()
            self._runtime = "tensorflow-fallback"

    def _extract_embedding(self, output):
        if isinstance(output, dict):
            if self.embedding_signature in output:
                return output[self.embedding_signature]
            if "embedding" in output:
                return output["embedding"]
            return next(iter(output.values()))

        if isinstance(output, (tuple, list)):
            chosen = output[0]
            for item in output:
                shape = getattr(item, "shape", None)
                if shape is not None and len(shape) == 2 and int(shape[-1]) >= 512:
                    chosen = item
                    break
            return chosen

        return output

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        if self._runtime == "torch":
            import torch

            if len(self._replicas) <= 1:
                x = torch.as_tensor(batch, dtype=torch.float32, device=self._torch_device)
                with torch.no_grad():
                    output = self._model(x)
                emb = self._extract_embedding(output)
                if hasattr(emb, "detach"):
                    emb = emb.detach().cpu().numpy()
                else:
                    emb = np.asarray(emb)
                return np.asarray(emb, dtype=np.float32)

            chunks = np.array_split(batch, len(self._replicas), axis=0)
            chunks = [chunk for chunk in chunks if chunk.size > 0]

            def run_chunk(model, device, chunk):
                x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
                with torch.no_grad():
                    output = model(x)
                emb = self._extract_embedding(output)
                if hasattr(emb, "detach"):
                    emb = emb.detach().cpu().numpy()
                else:
                    emb = np.asarray(emb)
                return np.asarray(emb, dtype=np.float32)

            outputs: list[np.ndarray] = []
            with ThreadPoolExecutor(max_workers=min(len(chunks), len(self._replicas))) as pool:
                futures = []
                for i, chunk in enumerate(chunks):
                    model = self._replicas[i % len(self._replicas)]
                    device = self._torch_devices[i % len(self._torch_devices)]
                    futures.append(pool.submit(run_chunk, model, device, chunk))
                for fut in futures:
                    outputs.append(fut.result())

            return np.concatenate(outputs, axis=0)

        import tensorflow as tf

        x = tf.constant(batch, dtype=tf.float32)
        if hasattr(self._model, "signatures") and "serving_default" in self._model.signatures:
            output = self._model.signatures["serving_default"](inputs=x)
        else:
            infer = getattr(self._model, "infer_tf", None) or self._model
            output = infer(x)
        emb = self._extract_embedding(output)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return np.asarray(emb, dtype=np.float32)

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

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (emb / norms).astype(np.float32)