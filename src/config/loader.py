"""YAML → ExperimentConfig.

Uses PyYAML if available; falls back to a tiny single-file YAML parser
for our deliberately simple config format. Paths are resolved relative
to a configurable base directory (typically the project root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config.schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    NoiseEvalConfig,
    OutputConfig,
    StageConfig,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load configs (pip install pyyaml)") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(d: Dict[str, Any], base: Path, keys: list[str]) -> Dict[str, Any]:
    out = dict(d)
    for k in keys:
        if k in out and out[k] is not None:
            p = Path(out[k])
            out[k] = p if p.is_absolute() else (base / p)
    return out


def _stage(d: Dict[str, Any] | None) -> StageConfig:
    if not d:
        raise ValueError("Stage config missing")
    return StageConfig(type=d["type"], params=d.get("params", {}) or {})


def load_config(yaml_path: Path, base_dir: Path | None = None) -> ExperimentConfig:
    yaml_path = Path(yaml_path).resolve()
    base = Path(base_dir).resolve() if base_dir else yaml_path.parent.parent
    raw = _read_yaml(yaml_path)

    data_raw = _resolve_paths(raw["data"], base, ["dataset_zip", "audio_dir"])
    output_raw = _resolve_paths(raw["output"], base, ["results_dir", "logs_dir", "cache_dir"])
    output_kwargs = {
        "results_dir": output_raw["results_dir"],
        "logs_dir": output_raw["logs_dir"],
        "cache_dir": output_raw["cache_dir"],
        "plots": bool(output_raw.get("plots", True)),
        "save_rankings": bool(output_raw.get("save_rankings", True)),
    }

    eval_raw = raw.get("evaluation", {}) or {}
    noise_raw = eval_raw.get("noise", {}) or {}
    if "noise_dir" in noise_raw and noise_raw["noise_dir"] is not None:
        noise_raw = _resolve_paths(noise_raw, base, ["noise_dir"])

    return ExperimentConfig(
        name=raw["name"],
        seed=int(raw.get("seed", 42)),
        data=DataConfig(**data_raw),
        preprocessing=_stage(raw["preprocessing"]),
        segmentation=_stage(raw["segmentation"]),
        embedding=_stage(raw["embedding"]),
        aggregation=_stage(raw["aggregation"]),
        representation=_stage(raw["representation"]),
        indexing=_stage(raw["indexing"]),
        fusion=_stage(raw["fusion"]),
        ranking=_stage(raw["ranking"]),
        similarity=raw.get("similarity", "cosine"),
        evaluation=EvaluationConfig(
            top_k=int(eval_raw.get("top_k", 10)),
            p_at=list(eval_raw.get("p_at", [1, 5])),
            recall_at=list(eval_raw.get("recall_at", [1, 5, 10])),
            measure_latency=bool(eval_raw.get("measure_latency", True)),
            measure_memory=bool(eval_raw.get("measure_memory", True)),
            noise=NoiseEvalConfig(
                enabled=bool(noise_raw.get("enabled", False)),
                noise_types=list(noise_raw.get("noise_types", ["white"])),
                snr_db=list(noise_raw.get("snr_db", [20.0, 10.0, 0.0, -5.0])),
                noise_dir=noise_raw.get("noise_dir"),
            ),
        ),
        output=OutputConfig(**output_kwargs),
        notes=raw.get("notes", ""),
    )
