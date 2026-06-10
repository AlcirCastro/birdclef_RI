"""Pick a random test audio and print the full ranking for all strategies.

This is a small demo / smoke test:
1. load the dataset split from a reference config,
2. pick a random record from the test split,
3. print its scientific and common names,
4. query the audio against every configured strategy,
5. print the complete ranking returned by each one.

The output is intended to help inspect whether the rankers are surfacing
biologically similar species, not just the exact top-1 label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_records, stratified_split  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402


DEFAULT_CONFIGS = [
    str(ROOT / "configs" / "strategy1_segments_torch.yaml"),
    str(ROOT / "configs" / "strategy2_super_embedding_torch.yaml"),
    str(ROOT / "configs" / "strategy1_segments_no_overlap_torch.yaml"),
    str(ROOT / "configs" / "strategy1_segments_no_overlap_noise_torch.yaml"),
    str(ROOT / "configs" / "strategy2_super_embedding_no_overlap_torch.yaml"),
    str(ROOT / "configs" / "strategy2_super_embedding_no_overlap_noise_torch.yaml"),
]


def _default_sample_config() -> str:
    return str(ROOT / "configs" / "strategy1_segments_torch.yaml")


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
    docs_by_label: dict[str, list] = {}
    for doc in runner.doc_store.documents:
        docs_by_label.setdefault(doc.label, []).append(doc)
    return cfg, runner, retriever, record_by_id, docs_by_label


def _label_meta(docs_by_label: dict[str, list], label: str) -> tuple[str, str]:
    docs = docs_by_label.get(label) or []
    if not docs:
        return "", ""
    doc = docs[0]
    return getattr(doc, "scientific_name", "") or "", getattr(doc, "common_name", "") or ""


def _rank_one_audio(cfg, runner, retriever, record_by_id, docs_by_label, audio_path: Path, top_k: int):
    query_rec = next((r for r in record_by_id.values() if r.local_path == audio_path), None)
    if query_rec is None:
        query_rec = next(iter(record_by_id.values()))
        query_rec = type(query_rec)(
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
        with np.load(query_emb) as z:
            query_emb = z["embeddings"].astype("float32")

    predictions = retriever.retrieve(query_emb, top_k=top_k)

    ranked = []
    for rank, pred in enumerate(predictions, start=1):
        scientific_name, common_name = _label_meta(docs_by_label, pred.label)
        ranked.append(
            {
                "rank": rank,
                "species": pred.label,
                "scientific_name": scientific_name,
                "common_name": common_name,
                "score": pred.score,
            }
        )

    return {
        "strategy": cfg.name,
        "representation": cfg.representation.type,
        "ranking": ranked,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-config",
        default=_default_sample_config(),
        help="Config used to load the dataset and pick a random test item.",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Experiment YAML files to compare.",
    )
    ap.add_argument("--top-k", type=int, default=20, help="How many results to print per strategy.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed used to pick the test audio.")
    ap.add_argument(
        "--output",
        default=str(ROOT / "results" / "random_test_audio_rankings" / "ranking.json"),
        help="JSON output file.",
    )
    args = ap.parse_args()

    sample_cfg = load_config(Path(args.sample_config), base_dir=ROOT)
    records = load_records(sample_cfg.data, seed=sample_cfg.seed)
    _, test = stratified_split(records, val_ratio=1.0 - sample_cfg.data.train_ratio, seed=sample_cfg.seed)
    if not test:
        raise RuntimeError("Test split is empty; cannot sample a random audio.")

    rng = np.random.default_rng(args.seed)
    query_rec = test[int(rng.integers(0, len(test)))]

    print("Random test sample")
    print(f"  item_id: {query_rec.item_id}")
    print(f"  scientific_name: {query_rec.scientific_name}")
    print(f"  common_name: {query_rec.common_name}")
    print(f"  primary_label: {query_rec.primary_label}")
    print(f"  audio_path: {query_rec.local_path}")
    print()

    results = []
    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str).expanduser().resolve()
        cfg, runner, retriever, record_by_id, docs_by_label = _build_strategy(cfg_path)
        query_results = _rank_one_audio(
            cfg,
            runner,
            retriever,
            record_by_id,
            docs_by_label,
            query_rec.local_path,
            args.top_k,
        )
        results.append(query_results)

        print(f"=== {cfg.name} ===")
        for item in query_results["ranking"]:
            common = f" | {item['common_name']}" if item.get("common_name") else ""
            print(
                f"{item['rank']:>2}. {item['species']} | {item['scientific_name']}{common}  "
                f"score={item['score']:.5f}"
            )
        print()

    print("Final query species")
    print(f"  scientific_name: {query_rec.scientific_name}")
    print(f"  common_name: {query_rec.common_name}")
    print(f"  primary_label: {query_rec.primary_label}")

    out_path = Path(args.output).expanduser().resolve()
    ensure_dir(out_path.parent)
    save_json(
        {
            "sample": {
                "item_id": query_rec.item_id,
                "scientific_name": query_rec.scientific_name,
                "common_name": query_rec.common_name,
                "primary_label": query_rec.primary_label,
                "audio_path": str(query_rec.local_path),
            },
            "top_k": args.top_k,
            "results": results,
        },
        out_path,
    )

    print(f"Saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())