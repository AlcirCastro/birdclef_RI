import hashlib
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
from panns_inference import AudioTagging

from bird_search.settings import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PANNs model (Cnn14)
        # Use a models directory in cache
        self.models_dir = self.settings.cache_dir / "panns_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(self.models_dir / "Cnn14.pth")
        self.model = AudioTagging(checkpoint_path=checkpoint_path, device='cpu')

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        payload = f"{path.as_posix()}|{stat.st_size}|{int(stat.st_mtime)}|{self.settings.embedding_name}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, path: Path) -> Path:
        return self.settings.embedding_cache_dir / f"{self._cache_key(path)}.npy"

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return (v / max(norm, 1e-8)).astype(np.float32)

    def _embed_audio(self, y: np.ndarray, sr: int) -> np.ndarray | None:
        if y.size == 0:
            return None

        try:
            if sr != 32000:
                y = librosa.resample(y, orig_sr=sr, target_sr=32000)
            
            # CNN14 precisa de no mínimo ~1s (32000 samples)
            # CNN14 precisa de no mínimo ~5s para as camadas de pooling não colapsarem
                min_samples = 32000 * 5  # 160000 samples
                if y.shape[0] < min_samples:
                    pad_length = min_samples - y.shape[0]
                    y = np.pad(y, (0, pad_length), mode='constant')
            
            audio_input = y[np.newaxis, :]  # → (1, N)
            
            clipwise_output, embedding = self.model.inference(audio_input)
            embedding = embedding[0]  # (1, 2048) → (2048,)
            
            return self._normalize(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def embed_path(self, path: Path, use_cache: bool = True) -> np.ndarray | None:
        cache_path = self._cache_path(path)
        if use_cache and cache_path.exists():
            return np.load(cache_path)

        try:
            y, _ = librosa.load(
                path.as_posix(),
                sr=self.settings.sample_rate,
                mono=True,
            )
        except Exception:
            return None

        if y.size == 0:
            return None

        vec = self._embed_audio(y, self.settings.sample_rate)
        if vec is None:
            return None

        if use_cache:
            np.save(cache_path, vec.astype(np.float32))

        return vec.astype(np.float32)

    def embed_bytes(self, payload: bytes) -> np.ndarray | None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        try:
            return self.embed_path(tmp_path, use_cache=False)
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    @property
    def runtime_backend(self) -> str:
        return self.settings.embedding_name