from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.data.records import Record


def stratified_split(
    records: List[Record], val_ratio: float, seed: int
) -> Tuple[List[Record], List[Record]]:
    """Split per `primary_label` so every species appears in both halves.

    Singleton-species records go to train (no test query possible).
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Record]] = {}
    for r in records:
        by_label.setdefault(r.primary_label, []).append(r)

    train, val = [], []
    for group in by_label.values():
        order = np.arange(len(group))
        rng.shuffle(order)
        if len(group) == 1:
            train.append(group[0])
            continue
        n_val = max(1, min(int(round(len(group) * val_ratio)), len(group) - 1))
        val_idx = set(order[:n_val].tolist())
        for i, rec in enumerate(group):
            (val if i in val_idx else train).append(rec)
    return train, val
