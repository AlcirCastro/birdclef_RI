"""Advanced ranking suite.

This script is meant to answer a more practical question than plain MAP/MRR:
which reranking strategy brings biologically close species closer to the top?

It evaluates several rankers on the same config and also reports:
- genus-hit rate in the top-K
- common-name token overlap in the top-K

On top of the built-in rankers, it builds several weighted hybrid rerankers
that combine attention, softmax, RRF, Borda, and taxonomy boost.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.config.schema import StageConfig  # noqa: E402
from src.data import load_records, stratified_split  # noqa: E402
from src.evaluation import Evaluator, LatencyTimer, QueryResult, current_rss_bytes, peak_rss_bytes  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.ranking import RANKERS  # noqa: E402
from src.ranking.base import Hit, RankedResult  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402
from src.utils.taxonomy import genus_from_scientific_name, taxonomy_info  # noqa: E402


DEFAULT_CONFIGS = [
    str(ROOT / "configs" / "strategy1_segments_torch.yaml"),
    str(ROOT / "configs" / "strategy2_super_embedding_torch.yaml"),
]


def _normalize_scores(results: list[RankedResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = np.asarray([r.score for r in results], dtype=np.float64)
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v - min_v < 1e-12:
        return {r.label: 1.0 for r in results}
    return {r.label: float((r.score - min_v) / (max_v - min_v)) for r in results}


def _ranked_to_map(results: list[RankedResult]) -> dict[str, float]:
    return {r.label: float(r.score) for r in results}


def _combine_ranked_outputs(weighted_rankings: list[tuple[float, list[RankedResult]]], top_k: int) -> list[RankedResult]:
    combined: dict[str, float] = defaultdict(float)
    for weight, ranked in weighted_rankings:
        norm = _normalize_scores(ranked)
        for lbl, score in norm.items():
            combined[lbl] += weight * score
    return [RankedResult(label=lbl, score=score) for lbl, score in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def _build_per_query_hits(
    index,
    doc_store,
    fusion,
    query_embeddings: np.ndarray,
    top_k: int,
    candidate_factor: int = 10,
    max_candidates: int = 200,
) -> list[list[Hit]]:
    if query_embeddings.size == 0:
        return []

    queries = fusion.queries(query_embeddings)
    per_q_k = min(max_candidates, max(top_k, top_k * candidate_factor))
    per_q_k = min(per_q_k, index.size())
    scores, ids = index.search(queries, per_q_k)

    labels = doc_store.labels
    per_query_hits: list[list[Hit]] = []
    for q_scores, q_ids in zip(scores, ids):
        hits: list[Hit] = []
        rank = 0
        for s, doc_id in zip(q_scores, q_ids):
            if doc_id < 0 or doc_id >= len(labels):
                continue
            rank += 1
            hits.append(Hit(label=labels[doc_id], score=float(s), doc_id=int(doc_id), rank=rank))
        per_query_hits.append(hits)
    return per_query_hits


def _evaluate_ranker(
    runner: ExperimentRunner,
    test_emb,
    ranker_name: str,
    ranker_params: dict,
    top_k: int,
) -> dict:
    ranker_cls = RANKERS.get(ranker_name)
    ranker = ranker_cls(**ranker_params)
    retriever = Retriever(runner.index, runner.doc_store, runner.fusion, ranker)

    timer = LatencyTimer()
    results: list[QueryResult] = []
    genus_hit_at = {5: [], 10: []}
    common_overlap_at = {5: [], 10: []}
    record_meta = {doc.label: doc for doc in runner.doc_store.documents}

    for rec, emb_source in test_emb:
        if isinstance(emb_source, Path):
            with np.load(emb_source) as z:
                emb = z["embeddings"].astype(np.float32)
        else:
            emb = emb_source

        with timer.measure():
            preds = retriever.retrieve(emb, top_k=top_k)

        results.append(
            QueryResult(
                record_id=rec.item_id,
                true_label=rec.primary_label,
                predictions=preds,
                latency_ms=timer.samples_ms[-1],
            )
        )

        true_genus = genus_from_scientific_name(rec.scientific_name)
        true_tokens = {t for t in rec.common_name.lower().split() if t}
        for cutoff in genus_hit_at:
            head = preds[:cutoff]
            genus_hit_at[cutoff].append(
                1.0 if any(genus_from_scientific_name(record_meta[p.label].scientific_name) == true_genus for p in head if p.label in record_meta) else 0.0
            )
            common_overlap_at[cutoff].append(
                1.0 if any(true_tokens.intersection({t for t in record_meta[p.label].common_name.lower().split() if t}) for p in head if p.label in record_meta) else 0.0
            )

    evaluator = Evaluator(p_at=[1, 5], recall_at=[1, 5, 10], top_k=top_k)
    corpus_counts = defaultdict(int)
    for doc in runner.doc_store:
        corpus_counts[doc.label] += 1
    metrics = evaluator.evaluate(results, dict(corpus_counts))
    metrics["summary"]["latency"] = timer.stats()
    metrics["summary"]["genus@5"] = float(np.mean(genus_hit_at[5])) if genus_hit_at[5] else 0.0
    metrics["summary"]["genus@10"] = float(np.mean(genus_hit_at[10])) if genus_hit_at[10] else 0.0
    metrics["summary"]["common@5"] = float(np.mean(common_overlap_at[5])) if common_overlap_at[5] else 0.0
    metrics["summary"]["common@10"] = float(np.mean(common_overlap_at[10])) if common_overlap_at[10] else 0.0
    return {
        "ranker": ranker_name,
        "params": ranker_params,
        "experiment": runner.cfg.name,
        "clean": metrics["summary"],
    }


def _evaluate_ensemble(
    runner: ExperimentRunner,
    test_emb,
    top_k: int,
    ensemble_name: str,
    ensemble_specs: list[tuple[float, str, dict]],
) -> dict:
    built_rankers = []
    for weight, ranker_name, ranker_params in ensemble_specs:
        ranker = RANKERS.get(ranker_name)(**ranker_params)
        if hasattr(ranker, "set_label_meta"):
            ranker.set_label_meta({doc.label: taxonomy_info(doc.scientific_name, doc.common_name) for doc in runner.doc_store.documents})
        built_rankers.append((weight, ranker))

    timer = LatencyTimer()
    results: list[QueryResult] = []
    genus_hit_at = {5: [], 10: []}
    common_overlap_at = {5: [], 10: []}
    record_meta = {doc.label: doc for doc in runner.doc_store.documents}

    for rec, emb_source in test_emb:
        if isinstance(emb_source, Path):
            with np.load(emb_source) as z:
                emb = z["embeddings"].astype(np.float32)
        else:
            emb = emb_source

        per_query_hits = _build_per_query_hits(runner.index, runner.doc_store, runner.fusion, emb, top_k=top_k)
        with timer.measure():
            ranked_outputs = [(weight, ranker.rank(per_query_hits, top_k)) for weight, ranker in built_rankers]
            final = _combine_ranked_outputs(ranked_outputs, top_k)

        results.append(
            QueryResult(
                record_id=rec.item_id,
                true_label=rec.primary_label,
                predictions=final,
                latency_ms=timer.samples_ms[-1],
            )
        )

        true_genus = genus_from_scientific_name(rec.scientific_name)
        true_tokens = {t for t in rec.common_name.lower().split() if t}
        for cutoff in genus_hit_at:
            head = final[:cutoff]
            genus_hit_at[cutoff].append(
                1.0 if any(genus_from_scientific_name(record_meta[p.label].scientific_name) == true_genus for p in head if p.label in record_meta) else 0.0
            )
            common_overlap_at[cutoff].append(
                1.0 if any(true_tokens.intersection({t for t in record_meta[p.label].common_name.lower().split() if t}) for p in head if p.label in record_meta) else 0.0
            )

    evaluator = Evaluator(p_at=[1, 5], recall_at=[1, 5, 10], top_k=top_k)
    corpus_counts = defaultdict(int)
    for doc in runner.doc_store:
        corpus_counts[doc.label] += 1
    metrics = evaluator.evaluate(results, dict(corpus_counts))
    metrics["summary"]["latency"] = timer.stats()
    metrics["summary"]["genus@5"] = float(np.mean(genus_hit_at[5])) if genus_hit_at[5] else 0.0
    metrics["summary"]["genus@10"] = float(np.mean(genus_hit_at[10])) if genus_hit_at[10] else 0.0
    metrics["summary"]["common@5"] = float(np.mean(common_overlap_at[5])) if common_overlap_at[5] else 0.0
    metrics["summary"]["common@10"] = float(np.mean(common_overlap_at[10])) if common_overlap_at[10] else 0.0
    return {
        "ranker": ensemble_name,
        "params": {"weights": [{"weight": weight, "ranker": ranker_name, "params": ranker_params} for weight, ranker_name, ranker_params in ensemble_specs]},
        "experiment": runner.cfg.name,
        "clean": metrics["summary"],
    }


def _make_runner(cfg_path: Path) -> tuple[ExperimentRunner, list]:
    cfg = load_config(cfg_path, base_dir=ROOT)
    runner = ExperimentRunner(cfg)
    records = load_records(cfg.data, seed=cfg.seed)
    train, test = stratified_split(records, val_ratio=1.0 - cfg.data.train_ratio, seed=cfg.seed)
    train_emb = runner._embed_split(train, desc="train")
    runner.doc_store.build(train_emb)
    vectors = runner.doc_store.vectors
    if vectors.shape[0] == 0:
        raise RuntimeError("No documents built — empty corpus")
    runner.index.build(vectors)
    return runner, runner._embed_split(test, desc="test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Experiment YAML files to evaluate.",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-name", default="ranking_suite_advanced")
    args = ap.parse_args()

    rankers_to_test = [
        ("softmax", {"temperature": 0.1}),
        ("attention", {"temperature": 0.05, "weight_by_query_norm": True}),
        ("attention", {"temperature": 0.10, "weight_by_query_norm": True}),
        ("rrf", {"k_const": 60.0}),
        ("rrf", {"k_const": 30.0}),
        ("borda", {}),
        ("topk_mean", {"per_label_k": 3}),
        ("topk_mean", {"per_label_k": 5}),
        ("weighted_topk", {"per_label_k": 5}),
        ("weighted_topk", {"per_label_k": 7}),
        ("taxonomy_boost", {"genus_boost": 0.20, "common_name_boost": 0.10}),
        ("taxonomy_boost", {"genus_boost": 0.35, "common_name_boost": 0.15}),
        ("taxonomy_boost", {"genus_boost": 0.10, "common_name_boost": 0.05}),
    ]

    ensemble_to_test = [
        (
            "hybrid_attention_rrf_taxonomy",
            [
                (0.45, "attention", {"temperature": 0.05, "weight_by_query_norm": True}),
                (0.35, "rrf", {"k_const": 60.0}),
                (0.20, "taxonomy_boost", {"genus_boost": 0.20, "common_name_boost": 0.10}),
            ],
        ),
        (
            "hybrid_softmax_rrf_taxonomy",
            [
                (0.40, "softmax", {"temperature": 0.10}),
                (0.35, "rrf", {"k_const": 60.0}),
                (0.25, "taxonomy_boost", {"genus_boost": 0.20, "common_name_boost": 0.10}),
            ],
        ),
        (
            "hybrid_attention_softmax_taxonomy",
            [
                (0.40, "attention", {"temperature": 0.05, "weight_by_query_norm": True}),
                (0.35, "softmax", {"temperature": 0.10}),
                (0.25, "taxonomy_boost", {"genus_boost": 0.25, "common_name_boost": 0.10}),
            ],
        ),
        (
            "hybrid_rrf_borda_taxonomy",
            [
                (0.40, "rrf", {"k_const": 60.0}),
                (0.35, "borda", {}),
                (0.25, "taxonomy_boost", {"genus_boost": 0.25, "common_name_boost": 0.15}),
            ],
        ),
        (
            "consensus_ensemble",
            [
                (0.20, "attention", {"temperature": 0.05, "weight_by_query_norm": True}),
                (0.20, "softmax", {"temperature": 0.10}),
                (0.20, "rrf", {"k_const": 60.0}),
                (0.15, "borda", {}),
                (0.25, "taxonomy_boost", {"genus_boost": 0.30, "common_name_boost": 0.15}),
            ],
        ),
        (
            "taxonomy_boost_strong_ensemble",
            [
                (0.30, "attention", {"temperature": 0.05, "weight_by_query_norm": True}),
                (0.30, "rrf", {"k_const": 60.0}),
                (0.40, "taxonomy_boost", {"genus_boost": 0.40, "common_name_boost": 0.20}),
            ],
        ),
    ]

    runs: list[dict] = []
    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str).expanduser().resolve()
        runner, test_emb = _make_runner(cfg_path)
        for ranker_name, ranker_params in rankers_to_test:
            runs.append(_evaluate_ranker(runner, test_emb, ranker_name, ranker_params, args.top_k))
        for ensemble_name, ensemble_specs in ensemble_to_test:
            runs.append(_evaluate_ensemble(runner, test_emb, args.top_k, ensemble_name, ensemble_specs))

    out_dir = ensure_dir(ROOT / "results" / args.out_name)
    save_json({"runs": runs}, out_dir / "comparison.json")

    lines = [
        "| experiment | ranker | MAP | MRR | P@1 | P@5 | R@1 | R@5 | R@10 | nDCG | genus@5 | genus@10 | common@5 | common@10 | latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        clean = run["clean"]
        lines.append(
            f"| {run['experiment']} | {run['ranker']} | {clean.get('MAP', 0.0):.4f} | {clean.get('MRR', 0.0):.4f} | "
            f"{clean.get('P@1', 0.0):.4f} | {clean.get('P@5', 0.0):.4f} | {clean.get('R@1', 0.0):.4f} | {clean.get('R@5', 0.0):.4f} | {clean.get('R@10', 0.0):.4f} | {clean.get('nDCG', 0.0):.4f} | "
            f"{clean.get('genus@5', 0.0):.4f} | {clean.get('genus@10', 0.0):.4f} | {clean.get('common@5', 0.0):.4f} | {clean.get('common@10', 0.0):.4f} | "
            f"{clean.get('latency', {}).get('mean_ms', 0.0):.1f} |"
        )

    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n")

    report = {
        "runs": runs,
        "memory_peak_rss_bytes": peak_rss_bytes(),
        "memory_current_rss_bytes": current_rss_bytes(),
    }
    save_json(report, out_dir / "report.json")

    print(table)
    print(f"\nSaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())