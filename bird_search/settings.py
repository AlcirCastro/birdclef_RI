from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    dataset_zip: Path
    cache_dir: Path
    audio_dir: Path
    embedding_cache_dir: Path
    es_url: str
    es_index: str
    sample_rate: int
    window_sec: float
    window_hop_sec: float
    top_k: int
    eval_top_k: int
    max_records: int | None
    min_rating: float
    train_ratio: float
    random_seed: int
    run_eval: bool
    enable_denoise: bool
    embedding_backend: str
    embedding_name: str


def load_settings(base_dir: Path) -> Settings:
    cache_dir = base_dir / "birdclef_cache"
    
    # Allow override via environment variable: BIRDCLEF_MAX_RECORDS
    #max_records_env = os.getenv("BIRDCLEF_MAX_RECORDS")
    #max_records = None if max_records_env is None else int(max_records_env)
    max_records = 3000  # Limite para desenvolvimento rápido; defina como None para usar todos os registros
    return Settings(
        base_dir=base_dir,
        dataset_zip=base_dir / "birdclef-2026.zip",
        cache_dir=cache_dir,
        audio_dir=cache_dir / "train_audio",
        embedding_cache_dir=cache_dir / "embeddings_panns_v1",
        es_url="http://localhost:9200",
        es_index="birdclef-panns-v1",
        sample_rate=32_000,
        window_sec=6.0,
        window_hop_sec=3.0,
        top_k=5,
        eval_top_k=10,
        max_records=max_records,
        min_rating=3.0,
        train_ratio=0.8,
        random_seed=42,
        run_eval=True,
        enable_denoise=False,
        embedding_backend="panns-cnn14",
        embedding_name="panns-cnn14-v1",
    )
