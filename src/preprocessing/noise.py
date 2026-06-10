"""Noise injection used by the robustness sweep — *not* a registered preprocessor.

`mix_noise` adds noise of a chosen type at a target SNR. It's invoked by
the noise-robustness evaluator, not the main preprocessing stage, so it
deliberately stays out of the PREPROCESSORS registry.

Supported types:
- "white": Gaussian noise
- "pink": 1/f noise via FFT shaping
- "rain", "wind", "urban", "speech": loaded from `noise_dir/<type>.wav`
  (any subfolder of clips works — one is picked at random)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    # Voss-McCartney via FFT shaping: 1/sqrt(f).
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(1, spec.size + 1)
    spec = spec / np.sqrt(f)
    pink = np.fft.irfft(spec, n=n)
    pink = pink / (np.std(pink) + 1e-12)
    return pink.astype(np.float32)


def _from_dir(noise_type: str, n: int, sample_rate: int,
              noise_dir: Path, rng: np.random.Generator) -> np.ndarray:
    import librosa
    candidates = sorted(Path(noise_dir).glob(f"{noise_type}*.wav")) + \
                 sorted((Path(noise_dir) / noise_type).glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No noise files for type={noise_type!r} under {noise_dir}")
    pick = candidates[int(rng.integers(0, len(candidates)))]
    y, _ = librosa.load(str(pick), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty noise file: {pick}")
    if y.size < n:
        reps = int(np.ceil(n / y.size))
        y = np.tile(y, reps)
    start = int(rng.integers(0, y.size - n + 1))
    return y[start : start + n].astype(np.float32)


def mix_noise(
    audio: np.ndarray,
    sample_rate: int,
    noise_type: str,
    snr_db: float,
    noise_dir: Optional[Path] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return `audio` mixed with `noise_type` at the given SNR (dB)."""
    rng = np.random.default_rng(seed)
    n = audio.size
    if n == 0:
        return audio

    if noise_type == "white":
        noise = _white(n, rng)
    elif noise_type == "pink":
        noise = _pink(n, rng)
    else:
        if noise_dir is None:
            raise ValueError(f"noise_dir required for noise_type={noise_type!r}")
        noise = _from_dir(noise_type, n, sample_rate, noise_dir, rng)

    sig_p = float(np.mean(audio ** 2)) + 1e-12
    noise_p = float(np.mean(noise ** 2)) + 1e-12
    target_noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_p / noise_p)
    return (audio + scale * noise).astype(np.float32)
