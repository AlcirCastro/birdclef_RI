"""E1 — Index benchmark: quality vs speed vs memory for graph-based ANN algorithms.

Runs the same exp1_baseline pipeline (clean audio, fixed 5s segments, audio-level
docs, early fusion, max ranker) but sweeps across six index algorithms:

  imenn   In-Memory ExactNN — exact brute-force (quality ceiling)
  hnsw    Hierarchical Navigable Small World (FAISS)
  nsw     Navigable Small World — single-layer flat graph (custom)
  nsg     Navigating Spread-out Graph (FAISS)
  vamana  Vamana / DiskANN — α-RNG robust pruning (custom)
  ssg     Satellite System Graph — cone-augmented kNN (custom)

Output (results/exp_index_benchmark/):
  comparison.json   — full result records
  comparison.md     — markdown table (quality + timing + memory)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.config.schema import StageConfig  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402

# ---------------------------------------------------------------------------
# Index variants to benchmark
# ---------------------------------------------------------------------------

DEFAULT_INDEXES = [
    ("imenn",  {},                                          "In-Memory ExactNN (exact)"),
    ("hnsw",   {"M": 32, "ef_construction": 200,
                "ef_search": 64},                          "HNSW (FAISS)"),
    ("nsw",    {"M": 32, "ef_search": 128, "n_entry": 16}, "NSW (single-layer)"),
    ("nsg",    {"R": 32, "search_L": 64},                  "NSG (FAISS)"),
    ("vamana", {"R": 32, "L": 100, "alpha": 1.2,
                "n_iters": 2},                             "Vamana (α-RNG)"),
    ("ssg",    {"M": 32, "cos_theta": 0.5,
                "ef_search": 128, "n_entry": 8},           "SSG (satellite)"),
]


def _run(cfg_path: Path, index_type: str, index_params: dict) -> dict:
    cfg = load_config(cfg_path, base_dir=ROOT)
    cfg = replace(cfg, indexing=StageConfig(type=index_type, params=index_params))
    t0 = time.perf_counter()
    report = ExperimentRunner(cfg).run()
    report["_wall_s"] = round(time.perf_counter() - t0, 2)
    return report


def _fmt_lat(report: dict) -> str:
    lat = report.get("clean", {}).get("latency") or {}
    m = lat.get("mean_ms")
    return f"{m:.2f}" if m is not None else "—"


def _fmt_mem(report: dict) -> str:
    mem = report.get("index_memory_bytes")
    if mem is None:
        return "—"
    return f"{mem / 1024 / 1024:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark ANN index algorithms on the E1 baseline pipeline."
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp_index_benchmark.yaml"),
        help="Base YAML config (indexing.type will be overridden).",
    )
    ap.add_argument(
        "--indexes",
        nargs="+",
        default=[name for name, _, _ in DEFAULT_INDEXES],
        help="Subset of index keys to run.",
    )
    ap.add_argument("--out-name", default="exp_index_benchmark")
    args = ap.parse_args()

    index_map = {name: (params, label) for name, params, label in DEFAULT_INDEXES}
    unknown = [n for n in args.indexes if n not in index_map]
    if unknown:
        raise KeyError(f"Unknown index keys: {unknown}. Available: {sorted(index_map)}")

    cfg_path = Path(args.config).expanduser().resolve()
    out_dir = ensure_dir(ROOT / "results" / args.out_name)

    runs: list[dict] = []
    for index_name in args.indexes:
        params, label = index_map[index_name]
        print(f"\n{'='*60}")
        print(f"  Index: {label}  ({index_name})")
        print(f"{'='*60}")
        try:
            report = _run(cfg_path, index_name, params)
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            runs.append({
                "index": index_name,
                "label": label,
                "error": str(exc),
            })
            continue

        clean = report.get("clean", {})
        runs.append({
            "index": index_name,
            "label": label,
            "params": params,
            "MAP": clean.get("MAP"),
            "MRR": clean.get("MRR"),
            "P@1": clean.get("P@1"),
            "P@5": clean.get("P@5"),
            "R@1": clean.get("R@1"),
            "R@5": clean.get("R@5"),
            "R@10": clean.get("R@10"),
            "nDCG": clean.get("nDCG"),
            "latency_mean_ms": (report.get("clean", {}).get("latency") or {}).get("mean_ms"),
            "latency_p95_ms": (report.get("clean", {}).get("latency") or {}).get("p95_ms"),
            "index_build_time_s": report.get("index_build_time_s"),
            "index_memory_bytes": report.get("index_memory_bytes"),
            "n_documents": report.get("n_documents"),
            "embedding_dim": report.get("embedding_dim"),
            "wall_s": report.get("_wall_s"),
        })

    save_json({"runs": runs}, out_dir / "comparison.json")

    # --- Markdown table ---
    cols = ["index", "label", "MAP", "MRR", "P@1", "P@5",
            "R@1", "R@5", "R@10", "nDCG",
            "lat_ms", "p95_ms", "build_s", "memory"]
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---" if i < 2 else "---:" for i in range(len(cols))]) + "|"
    lines  = [header, sep]

    for r in runs:
        if "error" in r:
            row = f"| {r['index']} | {r['label']} | — | — | — | — | — | — | — | — | — | — | — | ERROR: {r['error'][:60]} |"
        else:
            def _f(v, fmt=".4f"):
                return f"{v:{fmt}}" if v is not None else "—"

            build_s = r.get("index_build_time_s")
            mem_bytes = r.get("index_memory_bytes")
            row = (
                f"| {r['index']} | {r['label']} "
                f"| {_f(r.get('MAP'))} | {_f(r.get('MRR'))} "
                f"| {_f(r.get('P@1'))} | {_f(r.get('P@5'))} "
                f"| {_f(r.get('R@1'))} | {_f(r.get('R@5'))} | {_f(r.get('R@10'))} "
                f"| {_f(r.get('nDCG'))} "
                f"| {_f(r.get('latency_mean_ms'), '.2f')} "
                f"| {_f(r.get('latency_p95_ms'), '.2f')} "
                f"| {_f(build_s, '.2f') if build_s is not None else '—'} "
                f"| {f'{mem_bytes/1024/1024:.1f} MB' if mem_bytes else '—'} |"
            )
        lines.append(row)

    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
