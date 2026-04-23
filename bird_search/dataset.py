import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from bird_search.models import Record
from bird_search.settings import Settings


def _safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
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


def load_records(settings: Settings) -> list[Record]:
    if not settings.dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {settings.dataset_zip}")

    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    with zipfile.ZipFile(settings.dataset_zip) as zf:
        with zf.open("train.csv") as f:
            csv_rows = csv.DictReader(io.TextIOWrapper(f))
            for row in csv_rows:
                audio_key = f"train_audio/{row['filename'].strip()}"
                if audio_key in zf.namelist():
                    rows.append(row)

        rows = _stratified_sample(rows, settings.max_records, settings.random_seed)

        records: list[Record] = []
        for row in rows:
            rating = _safe_float(row.get("rating", "0"))
            if settings.min_rating > 0 and 0 < rating < settings.min_rating:
                continue

            rel = row["filename"].strip()
            local = settings.audio_dir / rel
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
                    class_name=row.get("class_name", ""),
                    rating=rating,
                    rel_path=rel,
                    local_path=local,
                )
            )

    return records


def stratified_split(records: list[Record], val_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Record]] = {}
    for r in records:
        by_label.setdefault(r.primary_label, []).append(r)

    train: list[Record] = []
    val: list[Record] = []

    for group in by_label.values():
        order = np.arange(len(group))
        rng.shuffle(order)
        if len(group) == 1:
            train.append(group[0])
            continue

        n_val = max(1, min(int(round(len(group) * val_ratio)), len(group) - 1))
        val_idx = set(order[:n_val].tolist())
        for i, rec in enumerate(group):
            if i in val_idx:
                val.append(rec)
            else:
                train.append(rec)

    return train, val
