"""Wilcoxon signed-rank analysis across models, strategies, and rerankers.

Mirrors ranking_suite_birdnet.py but collects per-query AP vectors for every
(backbone x strategy x reranker) combination and runs pairwise Wilcoxon tests
with Holm-Bonferroni correction.

Usage:
    python experiments/wilcoxon_ranking_analysis.py
    python experiments/wilcoxon_ranking_analysis.py --out-name my_wilcoxon
    python experiments/wilcoxon_ranking_analysis.py --top-k 10
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_records, stratified_split  # noqa: E402
from src.evaluation import Evaluator, LatencyTimer, QueryResult  # noqa: E402
from src.pipeline import ExperimentRunner  # noqa: E402
from src.ranking import RANKERS  # noqa: E402
from src.ranking.base import Hit, RankedResult  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402
from src.utils.taxonomy import genus_from_scientific_name, taxonomy_info  # noqa: E402


# ── Configs ───────────────────────────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "perch_v2": {
        "E1": ROOT / "configs" / "strategy1_segments_torch.yaml",
        "E2": ROOT / "configs" / "strategy2_super_embedding_torch.yaml",
    },
    "birdnet_v3": {
        "E1": ROOT / "configs" / "strategy1_segments_birdnet_v3.yaml",
        "E2": ROOT / "configs" / "strategy2_super_embedding_birdnet_v3.yaml",
    },
}

RANKERS_TO_TEST = [
    ("softmax",       {"temperature": 0.1}),
    ("attention",     {"temperature": 0.05, "weight_by_query_norm": True}),
    ("rrf",           {"k_const": 60.0}),
    ("borda",         {}),
    ("topk_mean",     {"per_label_k": 3}),
    ("taxonomy_boost",{"genus_boost": 0.20, "common_name_boost": 0.10}),
]

ENSEMBLES_TO_TEST = [
    (
        "hybrid_att_sm_tax",
        [
            (0.40, "attention",     {"temperature": 0.05, "weight_by_query_norm": True}),
            (0.35, "softmax",       {"temperature": 0.10}),
            (0.25, "taxonomy_boost",{"genus_boost": 0.25, "common_name_boost": 0.10}),
        ],
    ),
    (
        "hybrid_att_rrf_tax",
        [
            (0.45, "attention",     {"temperature": 0.05, "weight_by_query_norm": True}),
            (0.35, "rrf",           {"k_const": 60.0}),
            (0.20, "taxonomy_boost",{"genus_boost": 0.20, "common_name_boost": 0.10}),
        ],
    ),
]

# ── Helpers (copied from ranking_suite_birdnet.py) ────────────────────────────

def _normalize_scores(results: list[RankedResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = np.asarray([r.score for r in results], dtype=np.float64)
    min_v, max_v = float(scores.min()), float(scores.max())
    if max_v - min_v < 1e-12:
        return {r.label: 1.0 for r in results}
    return {r.label: float((r.score - min_v) / (max_v - min_v)) for r in results}


def _combine_ranked_outputs(
    weighted_rankings: list[tuple[float, list[RankedResult]]], top_k: int
) -> list[RankedResult]:
    combined: dict[str, float] = defaultdict(float)
    for weight, ranked in weighted_rankings:
        norm = _normalize_scores(ranked)
        for lbl, score in norm.items():
            combined[lbl] += weight * score
    return [
        RankedResult(label=lbl, score=score)
        for lbl, score in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ]


def _build_per_query_hits(index, doc_store, fusion, query_embeddings, top_k):
    if query_embeddings.size == 0:
        return []
    queries = fusion.queries(query_embeddings)
    per_q_k = min(200, max(top_k, top_k * 10), index.size())
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


def _make_runner(cfg_path: Path):
    cfg = load_config(cfg_path, base_dir=ROOT)
    runner = ExperimentRunner(cfg)
    records = load_records(cfg.data, seed=cfg.seed)
    train, test = stratified_split(records, val_ratio=1.0 - cfg.data.train_ratio, seed=cfg.seed)
    train_emb = runner._embed_split(train, desc="train")
    runner.doc_store.build(train_emb)
    vectors = runner.doc_store.vectors
    if vectors.shape[0] == 0:
        raise RuntimeError("Empty corpus")
    runner.index.build(vectors)
    return runner, runner._embed_split(test, desc="test")


# ── Per-query AP collection ───────────────────────────────────────────────────

def _collect_ranker(runner, test_emb, ranker_name, ranker_params, top_k) -> tuple[np.ndarray, dict]:
    """Returns (ap_per_query, summary_metrics)."""
    ranker_cls = RANKERS.get(ranker_name)
    ranker = ranker_cls(**ranker_params)
    if hasattr(ranker, "set_label_meta"):
        ranker.set_label_meta(
            {doc.label: taxonomy_info(doc.scientific_name, doc.common_name)
             for doc in runner.doc_store.documents}
        )
    retriever = Retriever(runner.index, runner.doc_store, runner.fusion, ranker)

    timer = LatencyTimer()
    results: list[QueryResult] = []

    for rec, emb_source in test_emb:
        if isinstance(emb_source, Path):
            with np.load(emb_source) as z:
                emb = z["embeddings"].astype(np.float32)
        else:
            emb = emb_source
        with timer.measure():
            preds = retriever.retrieve(emb, top_k=top_k)
        results.append(QueryResult(
            record_id=rec.item_id,
            true_label=rec.primary_label,
            predictions=preds,
            latency_ms=timer.samples_ms[-1],
        ))

    evaluator = Evaluator(p_at=[1, 5], recall_at=[1, 5, 10], top_k=top_k)
    corpus_counts = defaultdict(int)
    for doc in runner.doc_store:
        corpus_counts[doc.label] += 1
    metrics = evaluator.evaluate(results, dict(corpus_counts))

    ap_vec = np.array([row["AP"] for row in metrics["per_query"]])
    return ap_vec, metrics["summary"]


def _collect_ensemble(runner, test_emb, ensemble_name, ensemble_specs, top_k) -> tuple[np.ndarray, dict]:
    built_rankers = []
    for weight, ranker_name, ranker_params in ensemble_specs:
        ranker = RANKERS.get(ranker_name)(**ranker_params)
        if hasattr(ranker, "set_label_meta"):
            ranker.set_label_meta(
                {doc.label: taxonomy_info(doc.scientific_name, doc.common_name)
                 for doc in runner.doc_store.documents}
            )
        built_rankers.append((weight, ranker))

    timer = LatencyTimer()
    results: list[QueryResult] = []

    for rec, emb_source in test_emb:
        if isinstance(emb_source, Path):
            with np.load(emb_source) as z:
                emb = z["embeddings"].astype(np.float32)
        else:
            emb = emb_source
        per_query_hits = _build_per_query_hits(
            runner.index, runner.doc_store, runner.fusion, emb, top_k
        )
        with timer.measure():
            ranked_outputs = [(w, r.rank(per_query_hits, top_k)) for w, r in built_rankers]
            final = _combine_ranked_outputs(ranked_outputs, top_k)
        results.append(QueryResult(
            record_id=rec.item_id,
            true_label=rec.primary_label,
            predictions=final,
            latency_ms=timer.samples_ms[-1],
        ))

    evaluator = Evaluator(p_at=[1, 5], recall_at=[1, 5, 10], top_k=top_k)
    corpus_counts = defaultdict(int)
    for doc in runner.doc_store:
        corpus_counts[doc.label] += 1
    metrics = evaluator.evaluate(results, dict(corpus_counts))

    ap_vec = np.array([row["AP"] for row in metrics["per_query"]])
    return ap_vec, metrics["summary"]


# ── Wilcoxon with Holm-Bonferroni ─────────────────────────────────────────────

def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return adjusted p-values using Holm-Bonferroni step-down method."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    max_so_far = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = max(adj, max_so_far)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        max_so_far = adj
    return adjusted


def _run_wilcoxon_pairs(
    ap_vectors: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    alpha: float = 0.05,
) -> list[dict]:
    raw_results = []
    for key_a, key_b in pairs:
        a, b = ap_vectors[key_a], ap_vectors[key_b]
        diff = a - b
        if np.all(diff == 0):
            raw_results.append({"key_a": key_a, "key_b": key_b, "statistic": 0.0, "p_raw": 1.0,
                                 "map_a": float(a.mean()), "map_b": float(b.mean()),
                                 "delta_map": 0.0})
            continue
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        raw_results.append({
            "key_a": key_a,
            "key_b": key_b,
            "statistic": float(stat),
            "p_raw": float(p),
            "map_a": float(a.mean()),
            "map_b": float(b.mean()),
            "delta_map": float(a.mean() - b.mean()),
        })

    p_raws = [r["p_raw"] for r in raw_results]
    p_adj = _holm_bonferroni(p_raws)

    final = []
    for r, padj in zip(raw_results, p_adj):
        r["p_adjusted"] = padj
        r["significant"] = padj < alpha
        r["winner"] = r["key_a"] if r["delta_map"] > 0 else (r["key_b"] if r["delta_map"] < 0 else "tie")
        final.append(r)
    return final


# ── Report formatting ─────────────────────────────────────────────────────────

def _format_table(results: list[dict]) -> str:
    header = (
        "| A | B | MAP(A) | MAP(B) | ΔMAP | W-stat | p_raw | p_adj | sig | winner |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---|\n"
    )
    rows = []
    for r in results:
        sig = "✓" if r["significant"] else "✗"
        rows.append(
            f"| {r['key_a']} | {r['key_b']} "
            f"| {r['map_a']:.4f} | {r['map_b']:.4f} "
            f"| {r['delta_map']:+.4f} "
            f"| {r['statistic']:.1f} "
            f"| {r['p_raw']:.4e} | {r['p_adjusted']:.4e} "
            f"| {sig} | {r['winner']} |"
        )
    return header + "\n".join(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Wilcoxon analysis across backbones, strategies, rerankers")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-name", default="wilcoxon_analysis")
    ap.add_argument(
        "--backbones", nargs="+", default=list(BACKBONE_CONFIGS.keys()),
        help="Which backbones to run (perch_v2 birdnet_v3)"
    )
    ap.add_argument(
        "--strategies", nargs="+", default=["E1", "E2"],
        help="Which strategies to run (E1 E2)"
    )
    args = ap.parse_args()

    # ── Collect AP vectors ────────────────────────────────────────────────────
    # Key format: "{backbone}__{strategy}__{ranker}"
    ap_vectors: dict[str, np.ndarray] = {}
    summaries: list[dict] = []

    for backbone in args.backbones:
        for strategy in args.strategies:
            cfg_path = BACKBONE_CONFIGS[backbone][strategy]
            print(f"\n{'='*60}")
            print(f"  {backbone} / {strategy}  ({cfg_path.name})")
            print(f"{'='*60}")

            runner, test_emb = _make_runner(cfg_path)

            for ranker_name, ranker_params in RANKERS_TO_TEST:
                key = f"{backbone}__{strategy}__{ranker_name}"
                print(f"  ranker: {ranker_name} {ranker_params}")
                ap_vec, summary = _collect_ranker(
                    runner, test_emb, ranker_name, ranker_params, args.top_k
                )
                ap_vectors[key] = ap_vec
                summaries.append({"key": key, "backbone": backbone, "strategy": strategy,
                                   "ranker": ranker_name, "params": ranker_params,
                                   "MAP": float(ap_vec.mean()), **summary})

            for ensemble_name, ensemble_specs in ENSEMBLES_TO_TEST:
                key = f"{backbone}__{strategy}__{ensemble_name}"
                print(f"  ensemble: {ensemble_name}")
                ap_vec, summary = _collect_ensemble(
                    runner, test_emb, ensemble_name, ensemble_specs, args.top_k
                )
                ap_vectors[key] = ap_vec
                summaries.append({"key": key, "backbone": backbone, "strategy": strategy,
                                   "ranker": ensemble_name, "params": {},
                                   "MAP": float(ap_vec.mean()), **summary})

    # ── Build comparison pairs ────────────────────────────────────────────────
    all_keys = list(ap_vectors.keys())

    pairs: list[tuple[str, str]] = []
    pair_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # 1. Backbone comparison: BirdNET vs Perch, mesmo strategy + ranker
    if "perch_v2" in args.backbones and "birdnet_v3" in args.backbones:
        for strategy in args.strategies:
            for ranker_name, _ in RANKERS_TO_TEST:
                ka = f"birdnet_v3__{strategy}__{ranker_name}"
                kb = f"perch_v2__{strategy}__{ranker_name}"
                if ka in ap_vectors and kb in ap_vectors:
                    pairs.append((ka, kb))
                    pair_groups["backbone"].append((ka, kb))
            for ensemble_name, _ in ENSEMBLES_TO_TEST:
                ka = f"birdnet_v3__{strategy}__{ensemble_name}"
                kb = f"perch_v2__{strategy}__{ensemble_name}"
                if ka in ap_vectors and kb in ap_vectors:
                    pairs.append((ka, kb))
                    pair_groups["backbone"].append((ka, kb))

    # 2. Strategy comparison: E1 vs E2, mesmo backbone + ranker
    if "E1" in args.strategies and "E2" in args.strategies:
        for backbone in args.backbones:
            for ranker_name, _ in RANKERS_TO_TEST:
                ka = f"{backbone}__E1__{ranker_name}"
                kb = f"{backbone}__E2__{ranker_name}"
                if ka in ap_vectors and kb in ap_vectors:
                    pairs.append((ka, kb))
                    pair_groups["strategy"].append((ka, kb))
            for ensemble_name, _ in ENSEMBLES_TO_TEST:
                ka = f"{backbone}__E1__{ensemble_name}"
                kb = f"{backbone}__E2__{ensemble_name}"
                if ka in ap_vectors and kb in ap_vectors:
                    pairs.append((ka, kb))
                    pair_groups["strategy"].append((ka, kb))

    # 3. Reranker comparison: attention vs cada outro, por backbone+strategy
    for backbone in args.backbones:
        for strategy in args.strategies:
            anchor = f"{backbone}__{strategy}__attention"
            if anchor not in ap_vectors:
                continue
            for ranker_name, _ in RANKERS_TO_TEST:
                if ranker_name == "attention":
                    continue
                kb = f"{backbone}__{strategy}__{ranker_name}"
                if kb in ap_vectors:
                    pairs.append((anchor, kb))
                    pair_groups["reranker_vs_attention"].append((anchor, kb))
            for ensemble_name, _ in ENSEMBLES_TO_TEST:
                kb = f"{backbone}__{strategy}__{ensemble_name}"
                if kb in ap_vectors:
                    pairs.append((anchor, kb))
                    pair_groups["reranker_vs_attention"].append((anchor, kb))

    # 4. Score-based vs rank-fusion: attention vs rrf e borda, por backbone+strategy
    for backbone in args.backbones:
        for strategy in args.strategies:
            for score_r in ["attention", "softmax"]:
                for rank_r in ["rrf", "borda"]:
                    ka = f"{backbone}__{strategy}__{score_r}"
                    kb = f"{backbone}__{strategy}__{rank_r}"
                    if ka in ap_vectors and kb in ap_vectors:
                        if (ka, kb) not in pairs:
                            pairs.append((ka, kb))
                            pair_groups["score_vs_rank"].append((ka, kb))

    # Deduplicate preserving order
    seen: set[tuple[str, str]] = set()
    pairs_dedup = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            pairs_dedup.append(p)
    pairs = pairs_dedup

    print(f"\n{'='*60}")
    print(f"  Running {len(pairs)} Wilcoxon tests (alpha={args.alpha}, Holm-Bonferroni)")
    print(f"{'='*60}")

    wilcoxon_results = _run_wilcoxon_pairs(ap_vectors, pairs, alpha=args.alpha)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = ensure_dir(ROOT / "results" / args.out_name)

    # Save raw AP vectors as npz
    np.savez_compressed(
        out_dir / "ap_vectors.npz",
        **{k.replace("/", "_"): v for k, v in ap_vectors.items()}
    )

    # Save full JSON report
    report = {
        "alpha": args.alpha,
        "n_pairs": len(pairs),
        "n_queries": int(next(iter(ap_vectors.values())).shape[0]),
        "summaries": summaries,
        "wilcoxon": wilcoxon_results,
        "groups": {g: [{"key_a": a, "key_b": b} for a, b in ps]
                   for g, ps in pair_groups.items()},
    }
    save_json(report, out_dir / "report.json")

    # Save Markdown table per group
    md_lines = [f"# Wilcoxon Analysis (alpha={args.alpha}, Holm-Bonferroni)\n"]

    group_order = ["backbone", "strategy", "reranker_vs_attention", "score_vs_rank"]
    group_labels = {
        "backbone":               "## 1. BirdNET v3 vs Perch v2 (mesmo strategy + reranker)",
        "strategy":               "## 2. E1 (late fusion) vs E2 (early fusion) (mesmo backbone + reranker)",
        "reranker_vs_attention":  "## 3. Attention vs demais rerankers (por backbone + strategy)",
        "score_vs_rank":          "## 4. Score-based vs Rank-fusion",
    }

    res_by_pair = {(r["key_a"], r["key_b"]): r for r in wilcoxon_results}

    for group in group_order:
        if group not in pair_groups or not pair_groups[group]:
            continue
        md_lines.append(group_labels[group])
        group_results = [res_by_pair[(a, b)] for a, b in pair_groups[group] if (a, b) in res_by_pair]
        md_lines.append(_format_table(group_results))
        sig_count = sum(1 for r in group_results if r["significant"])
        md_lines.append(f"\n*{sig_count}/{len(group_results)} comparações significativas (p_adj < {args.alpha})*\n")

    md_lines.append("## Sumário de MAPs por configuração\n")
    md_lines.append("| Backbone | Strategy | Reranker | MAP |\n|---|---|---|---:|")
    for s in sorted(summaries, key=lambda x: -x["MAP"]):
        md_lines.append(f"| {s['backbone']} | {s['strategy']} | {s['ranker']} | {s['MAP']:.4f} |")

    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))
    print(f"\nSalvo em → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
