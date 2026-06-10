"""Confusion matrix heatmap and top-confusion table."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def plot_confusion_matrix(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    out_path: Path,
    max_classes: int = 30,
    normalize: bool = True,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(true_labels) != len(pred_labels):
        raise ValueError("Mismatched true/pred lengths")

    classes = sorted(set(list(true_labels) + list(pred_labels)))
    if len(classes) > max_classes:
        # Keep the most frequent classes among truths.
        counts: dict[str, int] = {}
        for t in true_labels:
            counts[t] = counts.get(t, 0) + 1
        classes = sorted(counts, key=counts.get, reverse=True)[:max_classes]
        keep = set(classes)
        kept = [(t, p) for t, p in zip(true_labels, pred_labels) if t in keep and p in keep]
        if not kept:
            return out_path
        true_labels, pred_labels = zip(*kept)

    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    M = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(true_labels, pred_labels):
        M[idx[t], idx[p]] += 1
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        M = M / row_sums

    fig, ax = plt.subplots(figsize=(0.4 * n + 2, 0.4 * n + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("predicted (top-1)")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def top_confusions_table(
    true_labels: Sequence[str], pred_labels: Sequence[str], top: int = 20,
) -> List[Tuple[str, str, int]]:
    counts: dict[tuple[str, str], int] = {}
    for t, p in zip(true_labels, pred_labels):
        if t == p:
            continue
        counts[(t, p)] = counts.get((t, p), 0) + 1
    return sorted(((t, p, c) for (t, p), c in counts.items()), key=lambda r: -r[2])[:top]
