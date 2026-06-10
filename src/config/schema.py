"""Config schema as nested frozen dataclasses.

Each `StageConfig` carries a `type` (registry key) plus a `params` dict
forwarded to the component's `__init__`. This is the only contract the
pipeline builder needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StageConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    dataset_zip: Path
    audio_dir: Path
    max_records: Optional[int] = None
    min_rating: float = 0.0
    train_ratio: float = 0.8
    sample_rate: int = 32_000


@dataclass(frozen=True)
class NoiseEvalConfig:
    """Robustness-to-noise sweep over (noise_type, snr_db)."""
    enabled: bool = False
    noise_types: List[str] = field(default_factory=lambda: ["white", "pink"])
    snr_db: List[float] = field(default_factory=lambda: [20.0, 10.0, 0.0, -5.0])
    noise_dir: Optional[Path] = None  # for rain/wind/urban/speech samples


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = 10
    p_at: List[int] = field(default_factory=lambda: [1, 5])
    recall_at: List[int] = field(default_factory=lambda: [1, 5, 10])
    measure_latency: bool = True
    measure_memory: bool = True
    noise: NoiseEvalConfig = field(default_factory=NoiseEvalConfig)


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    logs_dir: Path
    cache_dir: Path
    plots: bool = True
    save_rankings: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    data: DataConfig
    preprocessing: StageConfig
    segmentation: StageConfig
    embedding: StageConfig
    aggregation: StageConfig
    representation: StageConfig
    indexing: StageConfig
    fusion: StageConfig
    ranking: StageConfig
    similarity: str  # "cosine" | "l2"
    evaluation: EvaluationConfig
    output: OutputConfig
    notes: str = ""
