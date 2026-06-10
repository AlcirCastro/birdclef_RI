"""Lazy loader for the BirdCLEF-style zip.

Walks `train.csv` inside the zip, optionally stratified-samples to
`max_records`, and lazily extracts only the audio files we'll actually
use into `audio_dir`. Records carry an integer `item_id` assigned in
load order — used as primary key throughout the pipeline.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import List

import numpy as np

from src.config.schema import DataConfig
from src.data.records import Record


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _stratified_sample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(rows):
        return rows
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r.get("primary_label", ""), []).append(r)

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    pools: dict[str, list[dict]] = {}
    for label, group in by_label.items():
        idx = list(range(len(group)))
        rng.shuffle(idx)
        selected.append(group[idx[0]])
        pools[label] = [group[i] for i in idx[1:]]

    remaining = n - len(selected)
    labels = list(pools.keys())
    while remaining > 0:
        rng.shuffle(labels)
        progressed = False
        for label in labels:
            if not pools[label]:
                continue
            selected.append(pools[label].pop())
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return selected[:n]


def load_records(cfg: DataConfig, seed: int) -> List[Record]:
    if not cfg.dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {cfg.dataset_zip}")
    cfg.audio_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cfg.dataset_zip) as zf:
        names = set(zf.namelist())
        with zf.open("train.csv") as f:
            rows = [
                row for row in csv.DictReader(io.TextIOWrapper(f))
                if f"train_audio/{row['filename'].strip()}" in names
            ]
        rows = _stratified_sample(rows, cfg.max_records, seed)

        records: list[Record] = []
        for row in rows:
            rating = _safe_float(row.get("rating", "0"))
            if cfg.min_rating > 0 and 0 < rating < cfg.min_rating:
                continue
            rel = row["filename"].strip()
            local = cfg.audio_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                try:
                    with zf.open(f"train_audio/{rel}") as src, open(local, "wb") as dst:
                        dst.write(src.read())
                except KeyError:
                    continue
            records.append(
                Record(
                    item_id=len(records),
                    primary_label=row.get("primary_label", ""),
                    scientific_name=row.get("scientific_name", ""),
                    common_name=row.get("common_name", ""),
                    rating=rating,
                    rel_path=rel,
                    local_path=local,
                )
            )
    return records


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load mono float32 audio at the requested sample rate."""
    import librosa
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)
