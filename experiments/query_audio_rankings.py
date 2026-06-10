"""Query one audio file against one or more strategies and print rankings.

Example:
  python experiments/query_audio_rankings.py \
      --audio /path/to/audio.wav \
      --configs configs/strategy1_segments_torch.yaml \
                configs/strategy2_super_embedding_torch.yaml \
      --top-k 10

For each strategy this script:
  1. loads the configured corpus,
  2. builds the index,
  3. embeds the provided audio,
  4. prints the ranked species labels,
  5. includes representative source paths from the corpus for each label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import Record, load_records, stratified_split  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402


def _default_configs() -> list[str]:
    return [
        str(ROOT / "configs" / "strategy1_segments_torch.yaml"),
        str(ROOT / "configs" / "strategy2_super_embedding_torch.yaml"),
    ]


def _format_paths(paths: Iterable[Path], limit: int = 3) -> list[str]:
    out: list[str] = []
    for path in paths:
        text = str(path)
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _build_strategy(cfg_path: Path):
    cfg = load_config(cfg_path, base_dir=ROOT)
    runner = ExperimentRunner(cfg)

    records = load_records(cfg.data, seed=cfg.seed)
    train, _ = stratified_split(records, val_ratio=1.0 - cfg.data.train_ratio, seed=cfg.seed)

    train_emb = runner._embed_split(train, desc="train")
    runner.doc_store.build(train_emb)
    vectors = runner.doc_store.vectors
    if vectors.shape[0] == 0:
        raise RuntimeError(f"No documents built for strategy {cfg.name}")
    runner.index.build(vectors)

    retriever = Retriever(runner.index, runner.doc_store, runner.fusion, runner.ranker)
    record_by_id = {rec.item_id: rec for rec in train}
    return cfg, runner, retriever, record_by_id


def _rank_one_audio(cfg, runner, retriever, record_by_id: dict[int, Record], audio_path: Path, top_k: int):
    query_rec = Record(
        item_id=-1,
        primary_label="query",
        scientific_name="",
        common_name="",
        rating=0.0,
        rel_path=audio_path.name,
        local_path=audio_path,
    )
    query_emb = runner._embed_record(query_rec, use_cache=False)
    if isinstance(query_emb, Path):
        import numpy as np

        with np.load(query_emb) as z:
            query_emb = z["embeddings"].astype("float32")

    predictions = retriever.retrieve(query_emb, top_k=top_k)

    docs_by_label: dict[str, list] = defaultdict(list)
    for doc in runner.doc_store.documents:
        docs_by_label[doc.label].append(doc)

    ranked = []
    for rank, pred in enumerate(predictions, start=1):
        docs = docs_by_label.get(pred.label, [])
        source_paths: list[Path] = []
        source_ids: list[int] = []
        for doc in docs:
            for record_id in doc.source_record_ids:
                source_ids.append(record_id)
                record = record_by_id.get(record_id)
                if record is not None:
                    source_paths.append(record.local_path)
        ranked.append(
            {
                "rank": rank,
                "species": pred.label,
                "score": pred.score,
                "source_record_ids": sorted(set(source_ids)),
                "source_paths": _format_paths(source_paths),
            }
        )

    return {
        "strategy": cfg.name,
        "representation": cfg.representation.type,
        "ranking": ranked,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to the audio file to query.")
    ap.add_argument(
        "--configs",
        nargs="+",
        default=_default_configs(),
        help="One or more experiment YAML files to compare.",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--output",
        default=str(ROOT / "results" / "query_audio_rankings" / "ranking.json"),
        help="JSON output file.",
    )
    args = ap.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    results = []
    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str).expanduser().resolve()
        cfg, runner, retriever, record_by_id = _build_strategy(cfg_path)
        results.append(_rank_one_audio(cfg, runner, retriever, record_by_id, audio_path, args.top_k))

    out_path = Path(args.output).expanduser().resolve()
    ensure_dir(out_path.parent)
    save_json(
        {
            "audio": str(audio_path),
            "top_k": args.top_k,
            "results": results,
        },
        out_path,
    )

    print(json.dumps({"audio": str(audio_path), "results": results}, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())