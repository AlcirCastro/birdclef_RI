"""Spectral-gating denoising.

Estimates a noise floor from the quietest STFT frames and gates frequencies
below `noise_threshold * floor`. Uses `noisereduce` if installed, otherwise
falls back to manual spectral subtraction.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing import PREPROCESSORS
from src.preprocessing.base import BasePreprocessor


@PREPROCESSORS.register("spectral_gating")
class SpectralGatingPreprocessor(BasePreprocessor):
    def __init__(self, prop_decrease: float = 1.0, n_std_thresh: float = 1.5,
                 stationary: bool = True):
        self.prop_decrease = prop_decrease
        self.n_std_thresh = n_std_thresh
        self.stationary = stationary

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return audio
        try:
            import noisereduce as nr
            try:
                return nr.reduce_noise(
                    y=audio,
                    sr=sample_rate,
                    stationary=self.stationary,
                    prop_decrease=self.prop_decrease,
                    n_std_thresh_stationary=self.n_std_thresh,
                ).astype(np.float32)
            except (ValueError, RuntimeError):
                return self._fallback_spectral_subtraction(audio).astype(np.float32)
        except ImportError:
            return self._fallback_spectral_subtraction(audio).astype(np.float32)

    def _fallback_spectral_subtraction(self, y: np.ndarray) -> np.ndarray:
        import librosa
        if y.size == 0:
            return y.astype(np.float32)

        n_fft = min(2048, int(y.size))
        while n_fft > y.size and n_fft > 64:
            n_fft //= 2
        if n_fft < 64:
            return y.astype(np.float32)

        hop_length = max(1, n_fft // 4)
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(S), np.angle(S)
        if mag.size == 0 or mag.shape[1] == 0:
            return y.astype(np.float32)

        # Estimate noise from the quietest 10% of frames.
        frame_energy = mag.mean(axis=0)
        q = np.quantile(frame_energy, 0.1)
        noise_frames = mag[:, frame_energy <= q]
        if noise_frames.size == 0:
            noise_frames = mag[:, :10]
        if noise_frames.size == 0:
            return y.astype(np.float32)

        noise = noise_frames.mean(axis=1, keepdims=True) * self.n_std_thresh
        mag_clean = np.maximum(mag - self.prop_decrease * noise, 0.0)
        cleaned = librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop_length, length=y.size)
        return cleaned.astype(np.float32)
