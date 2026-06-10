"""Run a larger strategy comparison across overlap and no-overlap variants.

Defaults use the Torch-backed configs for every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402


METRIC_ROWS = [
    ("MAP", "MAP"),
    ("MRR", "MRR"),
    ("P@1", "P@1"),
    ("P@5", "P@5"),
    ("R@1", "R@1"),
    ("R@5", "R@5"),
    ("R@10", "R@10"),
    ("nDCG", "nDCG"),
]


def _summary_value(report: dict, key: str) -> str:
    v = report.get("clean", {}).get(key)
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _latency_ms(report: dict) -> str:
    lat = report.get("clean", {}).get("latency") or {}
    mean = lat.get("mean_ms")
    p95 = lat.get("p95_ms")
    if mean is None:
        return "—"
    p95_str = f" / p95 {p95:.1f}" if p95 is not None else ""
    return f"{mean:.1f}{p95_str} ms"


def _run(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path, base_dir=ROOT)
    return ExperimentRunner(cfg).run()


def _format_table(reports: list[dict]) -> str:
    names = [r["experiment"] for r in reports]
    lines = [
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, key in METRIC_ROWS:
        row = [label] + [_summary_value(report, key) for report in reports]
        lines.append("| " + " | ".join(row) + " |")
    lat_row = ["latency"] + [_latency_ms(report) for report in reports]
    lines.append("| " + " | ".join(lat_row) + " |")
    docs_row = ["n_documents"] + [str(report.get("n_documents", "—")) for report in reports]
    lines.append("| " + " | ".join(docs_row) + " |")
    idx_row = ["index_build_time_s"] + [f"{float(report.get('index_build_time_s', 0.0)):.1f}" for report in reports]
    lines.append("| " + " | ".join(idx_row) + " |")
    dim_row = ["embedding_dim"] + [str(report.get("embedding_dim", "—")) for report in reports]
    lines.append("| " + " | ".join(dim_row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            str(ROOT / "configs" / "strategy1_segments_torch.yaml"),
            str(ROOT / "configs" / "strategy2_super_embedding_torch.yaml"),
            str(ROOT / "configs" / "strategy1_segments_no_overlap_torch.yaml"),
            str(ROOT / "configs" / "strategy1_segments_no_overlap_noise_torch.yaml"),
            str(ROOT / "configs" / "strategy2_super_embedding_no_overlap_torch.yaml"),
            str(ROOT / "configs" / "strategy2_super_embedding_no_overlap_noise_torch.yaml"),
        ],
        help="YAML files to compare.",
    )
    ap.add_argument("--out-name", default="strategy_compare_giant")
    args = ap.parse_args()

    reports: list[dict] = []
    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str).expanduser().resolve()
        reports.append(_run(cfg_path))

    table = _format_table(reports)
    print("\n=== Giant strategy comparison ===\n")
    print(table)

    out_dir = ensure_dir(ROOT / "results" / args.out_name)
    save_json({"runs": reports}, out_dir / "comparison.json")
    (out_dir / "comparison.md").write_text(table + "\n")
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())