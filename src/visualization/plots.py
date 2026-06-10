"""Comparison plots over runs of an experiment family.

Each function takes a list of dicts (one per run) so callers don't need to
shape data into pandas. All return the saved file path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence


def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    return plt


def plot_latency_vs_metric(runs: Sequence[Mapping], metric: str, out_path: Path,
                           latency_key: str = "avg_latency_ms",
                           label_key: str = "name") -> Path:
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        ax.scatter(r[latency_key], r[metric])
        ax.annotate(str(r.get(label_key, "")), (r[latency_key], r[metric]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("avg query latency (ms)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_noise_robustness(rows: Sequence[Mapping], metric: str, out_path: Path,
                          x_key: str = "snr_db", group_key: str = "noise_type") -> Path:
    """rows: each dict has {snr_db, noise_type, <metric>}. One line per noise type."""
    plt = _ensure_mpl()
    by_group: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_group.setdefault(str(r[group_key]), []).append((float(r[x_key]), float(r[metric])))

    fig, ax = plt.subplots(figsize=(7, 5))
    for group, pts in by_group.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=group)
    ax.set_xlabel(x_key)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {x_key}")
    ax.invert_xaxis()  # higher SNR (cleaner) on the left
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_bars(runs: Sequence[Mapping], metric: str, out_path: Path,
                     label_key: str = "name") -> Path:
    plt = _ensure_mpl()
    names = [str(r.get(label_key, i)) for i, r in enumerate(runs)]
    values = [float(r[metric]) for r in runs]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(runs)), 5))
    ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
