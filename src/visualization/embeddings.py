"""2-D embedding scatter via t-SNE or UMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    method: str = "umap",
    max_points: int = 4000,
    random_state: int = 0,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if embeddings.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=30)
    elif method != "umap":
        raise ValueError(f"Unknown method {method!r}")

    coords = reducer.fit_transform(embeddings.astype(np.float32))

    unique = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20")
    color_for = {lbl: cmap(i % 20) for i, lbl in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 7))
    for lbl in unique:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7,
                   color=color_for[lbl], label=lbl)
    if len(unique) <= 25:
        ax.legend(fontsize=6, loc="best", markerscale=1.2)
    ax.set_title(f"{method.upper()} of Perch v2 embeddings ({len(coords)} points)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
