"""Run strategy 1 (segments) and strategy 2 (super-embedding) back-to-back
and print a side-by-side comparison.

    python experiments/strategy_compare.py

Outputs:
  - Streams each experiment's normal report to its own results/<name>/ dir.
  - Writes a merged results/strategy_compare/comparison.json plus a Markdown
    table the runner prints to stdout.

Override defaults via CLI:
  python experiments/strategy_compare.py --left configs/strategy1_segments.yaml \
                                         --right configs/strategy2_super_embedding.yaml
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


def _format_table(left: dict, right: dict) -> str:
    name_l, name_r = left["experiment"], right["experiment"]
    lines = [
        f"| Metric | {name_l} | {name_r} |",
        "|---|---|---|",
    ]
    for label, key in METRIC_ROWS:
        lines.append(f"| {label} | {_summary_value(left, key)} | {_summary_value(right, key)} |")
    lines.append(f"| latency | {_latency_ms(left)} | {_latency_ms(right)} |")
    lines.append(f"| n_documents | {left.get('n_documents','—')} | {right.get('n_documents','—')} |")
    lines.append(f"| index_build_time_s | {left.get('index_build_time_s', 0):.1f} | {right.get('index_build_time_s', 0):.1f} |")
    lines.append(f"| embedding_dim | {left.get('embedding_dim','—')} | {right.get('embedding_dim','—')} |")
    return "\n".join(lines)


def _run(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path, base_dir=ROOT)
    return ExperimentRunner(cfg).run()


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=("tf", "torch"), default="tf")
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(parents=[pre])
    if pre_args.backend == "torch":
        left_default = ROOT / "configs" / "strategy1_segments_torch.yaml"
        right_default = ROOT / "configs" / "strategy2_super_embedding_torch.yaml"
    else:
        left_default = ROOT / "configs" / "strategy1_segments.yaml"
        right_default = ROOT / "configs" / "strategy2_super_embedding.yaml"

    ap.add_argument("--left",  default=str(left_default))
    ap.add_argument("--right", default=str(right_default))
    ap.add_argument("--out-name", default="strategy_compare")
    args = ap.parse_args()

    left  = _run(Path(args.left))
    right = _run(Path(args.right))

    table = _format_table(left, right)
    print("\n=== Strategy comparison ===\n")
    print(table)

    out_dir = ensure_dir(ROOT / "results" / args.out_name)
    save_json({"left": left, "right": right}, out_dir / "comparison.json")
    (out_dir / "comparison.md").write_text(table + "\n")
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
