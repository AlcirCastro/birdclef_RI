import hashlib
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np

from bird_search.settings import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        payload = f"{path.as_posix()}|{stat.st_size}|{int(stat.st_mtime)}|{self.settings.embedding_name}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, path: Path) -> Path:
        return self.settings.embedding_cache_dir / f"{self._cache_key(path)}.npy"

    def _mel_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """v0 baseline: global log-mel mean/std over full audio."""
        n_fft = 2048
        while n_fft > len(y) and n_fft > 64:
            n_fft //= 2

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=32,
            n_fft=n_fft,
            hop_length=max(256, n_fft // 4),
            fmin=50,
            fmax=sr // 2,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        parts = [
            log_mel.mean(axis=1),
            log_mel.std(axis=1),
        ]
        vec = np.concatenate(parts).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / max(norm, 1e-8)).astype(np.float32)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return (v / max(norm, 1e-8)).astype(np.float32)

    def _embed_audio(self, y: np.ndarray, sr: int) -> np.ndarray | None:
        if y.size == 0:
            return None

        return self._normalize(self._mel_features(y, sr))

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