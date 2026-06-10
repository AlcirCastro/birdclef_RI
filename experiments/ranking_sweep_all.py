"""Evaluate ranking strategies on one or more experiment configs.

Defaults use the Torch-backed strategy configs because those are the ones
currently being compared and they exercise the late-fusion path.
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
    ("segment", {}),
    ("mean", {}),
    ("max", {}),
    ("topk_mean", {"per_label_k": 3}),
    ("hit", {}),
    ("median", {}),
    ("threshold", {"tau": 0.5}),
    ("weighted_topk", {"per_label_k": 5}),
    ("softmax", {"temperature": 0.1}),
    ("rrf", {"k_const": 60.0}),
    ("borda", {}),
    ("attention", {"temperature": 0.05, "weight_by_query_norm": False}),
    ("taxonomy_boost", {"genus_boost": 0.20, "common_name_boost": 0.10}),
]


def _run(cfg_path: Path, ranking_type: str, ranking_params: dict) -> dict:
    cfg = load_config(cfg_path, base_dir=ROOT)
    cfg = replace(cfg, ranking=StageConfig(type=ranking_type, params=ranking_params))
    return ExperimentRunner(cfg).run()


def _latency_ms(report: dict) -> str:
    lat = report.get("clean", {}).get("latency") or {}
    mean = lat.get("mean_ms")
    if mean is None:
        return "—"
    p95 = lat.get("p95_ms")
    return f"{mean:.1f}" + (f" / {p95:.1f}" if p95 is not None else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            str(ROOT / "configs" / "strategy1_segments_torch.yaml"),
            str(ROOT / "configs" / "strategy2_super_embedding_torch.yaml"),
        ],
        help="Experiment YAML files to sweep rankings on.",
    )
    ap.add_argument(
        "--rankings",
        nargs="+",
        default=[name for name, _ in DEFAULT_RANKINGS],
        help="Ranking keys to evaluate.",
    )
    ap.add_argument("--out-name", default="ranking_sweep_all")
    args = ap.parse_args()

    ranking_map = {name: params for name, params in DEFAULT_RANKINGS}
    missing = [name for name in args.rankings if name not in ranking_map]
    if missing:
        raise KeyError(f"Unknown ranking options: {missing}. Known: {sorted(ranking_map)}")

    runs: list[dict] = []
    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str).expanduser().resolve()
        base_name = cfg_path.stem
        for ranking_name in args.rankings:
            report = _run(cfg_path, ranking_name, ranking_map[ranking_name])
            runs.append(
                {
                    "base_config": base_name,
                    "ranking": ranking_name,
                    "experiment": report["experiment"],
                    "clean": report["clean"],
                    "n_documents": report.get("n_documents"),
                    "embedding_dim": report.get("embedding_dim"),
                    "index_build_time_s": report.get("index_build_time_s"),
                    "elapsed_s": report.get("elapsed_s"),
                }
            )

    out_dir = ensure_dir(ROOT / "results" / args.out_name)
    save_json({"runs": runs}, out_dir / "comparison.json")

    lines = [
        "| base_config | ranking | MAP | MRR | P@1 | P@5 | R@1 | R@5 | R@10 | nDCG | latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        clean = run["clean"]
        lines.append(
            f"| {run['base_config']} | {run['ranking']} | {clean.get('MAP', 0.0):.4f} | {clean.get('MRR', 0.0):.4f} | "
            f"{clean.get('P@1', 0.0):.4f} | {clean.get('P@5', 0.0):.4f} | {clean.get('R@1', 0.0):.4f} | {clean.get('R@5', 0.0):.4f} | "
            f"{clean.get('R@10', 0.0):.4f} | {clean.get('nDCG', 0.0):.4f} | {_latency_ms(run)} |"
        )

    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n")
    print(table)
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())