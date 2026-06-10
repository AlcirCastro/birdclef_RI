"""Noise-handling preprocessors. Register new ones via @PREPROCESSORS.register("name")."""

from src.utils.registry import Registry

PREPROCESSORS = Registry("preprocessors")

# Side-effect imports populate the registry.
from src.preprocessing import identity, spectral_gating, bandpass  # noqa: F401,E402

from src.preprocessing.base import BasePreprocessor  # noqa: E402

__all__ = ["PREPROCESSORS", "BasePreprocessor"]
