"""Tiny shared launcher used by every exp*.py.

Each experiment file is a 3-line orchestrator:

    from experiments._runner import main
    if __name__ == "__main__":
        main(__file__)

The launcher resolves the matching `configs/<exp_stem>.yaml`, loads it,
and runs the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when running `python experiments/expN.py` directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402


def main(experiment_file: str | Path, config_override: Path | None = None) -> dict:
    exp_path = Path(experiment_file).resolve()
    cfg_path = config_override or (ROOT / "configs" / f"{exp_path.stem}.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path, base_dir=ROOT)
    return ExperimentRunner(cfg).run()
