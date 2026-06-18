"""E2 — Ranker benchmark: all aggregation strategies on the segment + late-fusion path.

Uses exp_ranker_benchmark.yaml as the base (segment docs, HNSW index, late fusion)
so each test audio produces one result list *per segment* — giving all rankers
meaningful multi-list input to aggregate into a final species ranking.

Rankers evaluated:
  max          Max score per label (baseline)
  mean         Mean score per label
  median       Median score per label
  segment      Flat dedup keeping best score (same as max, explicit)
  topk_mean    Mean of top-3 scores per label
  weighted_topk Top-5 mean with linearly decaying rank weights
  hit          Vote count (number of result lists a label appears in)
  threshold    Mean over scores ≥ 0.5 only
  softmax      Per-list softmax weights, summed per label
  rrf          Reciprocal Rank Fusion
  borda        Borda count (n − rank contribution)
  attention    Attention-weighted votes (softmax with peak-score query weight)
  taxonomy_boost Max score + genus/common-name proximity boost

Output (results/exp_ranker_benchmark/):
  comparison.json   — full result records
  comparison.md     — markdown table
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.config.schema import StageConfig  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402

DEFAULT_RANKINGS = [
    ("max",          {},                                           "Max score"),
    ("mean",         {},                                           "Mean score"),
    ("median",       {},                                           "Median score"),
    ("segment",      {},                                           "Segment-level (dedup max)"),
    ("topk_mean",    {"per_label_k": 3},                          "Top-3 mean"),
    ("weighted_topk",{"per_label_k": 5},                          "Weighted top-5"),
    ("hit",          {},                                           "Hit count (vote)"),
    ("threshold",    {"tau": 0.5},                                 "Threshold mean (τ=0.5)"),
    ("softmax",      {"temperature": 0.1},                        "Softmax weighted"),
    ("rrf",          {"k_const": 60.0},                           "RRF (k=60)"),
    ("borda",        {},                                           "Borda count"),
    ("attention",    {"temperature": 0.05,
                      "weight_by_query_norm": False},              "Attention weighted"),
    ("taxonomy_boost",{"genus_boost": 0.20,
                       "common_name_boost": 0.10},                 "Taxonomy boost"),
]


def _run(cfg_path: Path, ranking_type: str, ranking_params: dict) -> dict:
    cfg = load_config(cfg_path, base_dir=ROOT)
    cfg = replace(cfg, ranking=StageConfig(type=ranking_type, params=ranking_params))
    return ExperimentRunner(cfg).run()


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "—"


def _lat(r: dict) -> str:
    m = r.get("latency_mean_ms")
    p = r.get("latency_p95_ms")
    if m is None:
        return "—"
    return f"{m:.2f}" + (f" / {p:.2f}" if p is not None else "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark all ranking strategies on the E2 segment+late-fusion pipeline."
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "exp_ranker_benchmark.yaml"),
        help="Base YAML config (ranking.type will be overridden).",
    )
    ap.add_argument(
        "--rankings",
        nargs="+",
        default=[name for name, _, _ in DEFAULT_RANKINGS],
        help="Subset of ranker keys to evaluate.",
    )
    ap.add_argument("--out-name", default="exp_ranker_benchmark")
    args = ap.parse_args()

    ranking_map = {name: (params, label) for name, params, label in DEFAULT_RANKINGS}
    unknown = [n for n in args.rankings if n not in ranking_map]
    if unknown:
        raise KeyError(f"Unknown ranking keys: {unknown}. Available: {sorted(ranking_map)}")

    cfg_path = Path(args.config).expanduser().resolve()
    out_dir = ensure_dir(ROOT / "results" / args.out_name)

    runs: list[dict] = []
    for ranking_name in args.rankings:
        params, label = ranking_map[ranking_name]
        print(f"\n{'='*60}")
        print(f"  Ranker: {label}  ({ranking_name})")
        print(f"{'='*60}")
        try:
            report = _run(cfg_path, ranking_name, params)
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            runs.append({"ranking": ranking_name, "label": label, "error": str(exc)})
            continue

        clean = report.get("clean", {})
        runs.append({
            "ranking": ranking_name,
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
            "latency_mean_ms": (clean.get("latency") or {}).get("mean_ms"),
            "latency_p95_ms": (clean.get("latency") or {}).get("p95_ms"),
            "n_documents": report.get("n_documents"),
            "index_build_time_s": report.get("index_build_time_s"),
            "elapsed_s": report.get("elapsed_s"),
        })

    save_json({"runs": runs}, out_dir / "comparison.json")

    cols = ["ranking", "label", "MAP", "MRR",
            "P@1", "P@5", "R@1", "R@5", "R@10", "nDCG", "lat_mean/p95 (ms)"]
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---" if i < 2 else "---:" for i in range(len(cols))]) + "|"
    lines  = [header, sep]

    for r in runs:
        if "error" in r:
            row = f"| {r['ranking']} | {r['label']} | — | — | — | — | — | — | — | — | ERROR |"
        else:
            row = (
                f"| {r['ranking']} | {r['label']} "
                f"| {_fmt(r.get('MAP'))} | {_fmt(r.get('MRR'))} "
                f"| {_fmt(r.get('P@1'))} | {_fmt(r.get('P@5'))} "
                f"| {_fmt(r.get('R@1'))} | {_fmt(r.get('R@5'))} | {_fmt(r.get('R@10'))} "
                f"| {_fmt(r.get('nDCG'))} | {_lat(r)} |"
            )
        lines.append(row)

    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
