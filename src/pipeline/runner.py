"""End-to-end experiment runner.

Lifecycle:
  1. Configure logging, seed
  2. Load + split dataset
  3. Build all stage components from config
  4. Embed train + test (with on-disk cache)
  5. Build documents → build index
  6. Run queries → evaluate
  7. (Optional) sweep noise robustness over (noise_type × SNR)
  8. Save metrics, per-query rows, plots, embeddings sample

The runner is the only place that knows the lifecycle. Adding a new
component is a registry decoration; adding a new variable is a config
field; the runner's body never has to change.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config.schema import ExperimentConfig
from src.data import load_audio, load_records, stratified_split
from src.data.records import Record
from src.embeddings import EmbeddingCache
from src.evaluation import (
    Evaluator,
    LatencyTimer,
    QueryResult,
    current_rss_bytes,
    peak_rss_bytes,
)
from src.pipeline.builder import PipelineBuilder
from src.preprocessing.noise import mix_noise
from src.retrieval import Retriever
from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    save_csv,
    save_json,
    seed_everything,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_embedding_2d,
    plot_noise_robustness,
    top_confusions_table,
)


RecordEmbeddingEntry = Tuple[Record, Path | np.ndarray]


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.builder = PipelineBuilder(cfg)
        self.run_dir = ensure_dir(cfg.output.results_dir / cfg.name)
        configure_logging(cfg.name, cfg.output.logs_dir)
        self.log = get_logger("runner")
        seed_everything(cfg.seed)

        # Components built once and reused.
        self.preproc = self.builder.preprocessor()
        self.segmenter = self.builder.segmenter()
        self.embedder = self.builder.embedder()
        self.doc_store = self.builder.doc_store()
        self.index = self.builder.index()
        self.fusion = self.builder.fusion()
        self.ranker = self.builder.ranker()
        self.cache = EmbeddingCache(
            cache_dir=cfg.output.cache_dir / "embeddings",
            embedder=self.embedder,
            preproc_id=f"{cfg.preprocessing.type}:{sorted((cfg.preprocessing.params or {}).items())}",
            segmenter_id=f"{cfg.segmentation.type}:{sorted((cfg.segmentation.params or {}).items())}",
        )

    # ------------------------------------------------------------------ #
    def _embed_record(self, rec: Record, noise_inject: Optional[Tuple[str, float]] = None,
                      use_cache: bool = True) -> Path | np.ndarray:
        """Returns (n_segments, dim) for one record, with optional noise injection.

        Caching is bypassed when noise is injected — the cached vector is for
        the clean audio only.
        """
        if use_cache and noise_inject is None:
            cached_path = self.cache.path_for(rec.local_path)
            if cached_path.exists():
                return cached_path

        audio = load_audio(rec.local_path, self.cfg.data.sample_rate)
        if noise_inject is not None:
            noise_type, snr_db = noise_inject
            audio = mix_noise(
                audio, self.cfg.data.sample_rate, noise_type, snr_db,
                noise_dir=self.cfg.evaluation.noise.noise_dir,
                seed=self.cfg.seed + rec.item_id,
            )
        audio = self.preproc.process(audio, self.cfg.data.sample_rate)
        segments = self.segmenter.segment(audio, self.cfg.data.sample_rate)
        if not segments:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        emb = self.embedder.embed_segments(segments)

        if use_cache and noise_inject is None:
            return self.cache.save(rec.local_path, emb, segments)
        return emb

    def _embed_split(self, records: Sequence[Record], desc: str,
                      noise_inject: Optional[Tuple[str, float]] = None) -> List[RecordEmbeddingEntry]:
        out: list[RecordEmbeddingEntry] = []
        n = len(records)
        for i, rec in enumerate(records, start=1):
            emb = self._embed_record(rec, noise_inject=noise_inject)
            if isinstance(emb, Path):
                out.append((rec, emb))
            elif emb.shape[0] > 0:
                out.append((rec, emb))
            if i % 50 == 0 or i == n:
                self.log.info("[%s] embedded %d/%d", desc, i, n)
        return out

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        cfg = self.cfg
        self.log.info("Experiment %s starting", cfg.name)
        t0 = time.perf_counter()

        records = load_records(cfg.data, seed=cfg.seed)
        self.log.info("Loaded %d records", len(records))
        train, test = stratified_split(records, val_ratio=1.0 - cfg.data.train_ratio,
                                       seed=cfg.seed)
        self.log.info("Split: train=%d test=%d", len(train), len(test))

        # ---- index build ----
        train_emb = self._embed_split(train, desc="train")
        t_idx = time.perf_counter()
        self.doc_store.build(train_emb)
        vectors = self.doc_store.vectors
        if vectors.shape[0] == 0:
            raise RuntimeError("No documents built — empty corpus")
        self.index.build(vectors)
        index_time_s = time.perf_counter() - t_idx
        self.log.info("Index built: %d docs, dim=%d, %.1fs", vectors.shape[0], vectors.shape[1],
                      index_time_s)

        retriever = Retriever(self.index, self.doc_store, self.fusion, self.ranker)

        # ---- clean evaluation ----
        clean_metrics, clean_results = self._evaluate(test, retriever, label="clean")

        # ---- noise robustness sweep ----
        noise_rows: list[dict] = []
        if cfg.evaluation.noise.enabled:
            for nt in cfg.evaluation.noise.noise_types:
                for snr in cfg.evaluation.noise.snr_db:
                    label = f"noisy:{nt}@{snr}dB"
                    test_emb = self._embed_split(test, desc=label, noise_inject=(nt, snr))
                    # Reuse the same retriever (corpus is clean; only queries are noisy).
                    metrics, _ = self._evaluate_pre_embedded(
                        test_emb, retriever, label=label,
                    )
                    row = {"noise_type": nt, "snr_db": float(snr), **metrics["summary"]}
                    noise_rows.append(row)

        # ---- saving ----
        report = {
            "experiment": cfg.name,
            "notes": cfg.notes,
            "n_train": len(train),
            "n_test": len(test),
            "n_documents": int(vectors.shape[0]),
            "embedding_dim": int(vectors.shape[1]),
            "index_build_time_s": index_time_s,
            "memory_peak_rss_bytes": peak_rss_bytes(),
            "memory_current_rss_bytes": current_rss_bytes(),
            "config": _config_to_dict(cfg),
            "clean": clean_metrics["summary"],
            "noise": noise_rows,
            "elapsed_s": time.perf_counter() - t0,
        }
        save_json(report, self.run_dir / "report.json")
        save_csv(clean_metrics["per_query"], self.run_dir / "per_query_clean.csv")
        if noise_rows:
            save_csv(noise_rows, self.run_dir / "noise_robustness.csv")

        if cfg.output.save_rankings:
            save_json(
                [
                    {
                        "record_id": r.record_id,
                        "true_label": r.true_label,
                        "predictions": [asdict(p) for p in r.predictions],
                    }
                    for r in clean_results
                ],
                self.run_dir / "rankings_clean.json",
            )

        if cfg.output.plots:
            self._plots(clean_results, train_emb, noise_rows)

        self.log.info("Experiment %s done in %.1fs — MAP=%.4f MRR=%.4f",
                      cfg.name, report["elapsed_s"],
                      clean_metrics["summary"]["MAP"], clean_metrics["summary"]["MRR"])
        return report

    # ------------------------------------------------------------------ #
    def _evaluate(self, test: Sequence[Record], retriever: Retriever, label: str):
        test_emb = self._embed_split(test, desc=f"test:{label}")
        return self._evaluate_pre_embedded(test_emb, retriever, label)

    def _evaluate_pre_embedded(
        self,
        test_emb: Sequence[RecordEmbeddingEntry],
        retriever: Retriever,
        label: str,
    ):
        cfg = self.cfg
        timer = LatencyTimer()
        results: list[QueryResult] = []
        for rec, emb_source in test_emb:
            if isinstance(emb_source, Path):
                with np.load(emb_source) as z:
                    emb = z["embeddings"].astype(np.float32)
            else:
                emb = emb_source
            with timer.measure():
                preds = retriever.retrieve(emb, top_k=cfg.evaluation.top_k)
            results.append(
                QueryResult(
                    record_id=rec.item_id,
                    true_label=rec.primary_label,
                    predictions=preds,
                    latency_ms=timer.samples_ms[-1],
                )
            )

        evaluator = Evaluator(p_at=cfg.evaluation.p_at, recall_at=cfg.evaluation.recall_at,
                              top_k=cfg.evaluation.top_k)
        corpus_counts = Counter([d.label for d in self.doc_store])
        # For recall, we want per-label *count of relevant items*. With one
        # doc per species there is exactly 1 relevant; with segment docs, it
        # is the count in the corpus. Use whichever is meaningful.
        metrics = evaluator.evaluate(results, dict(corpus_counts))
        metrics["summary"]["latency"] = timer.stats()
        self.log.info("[%s] MAP=%.4f MRR=%.4f P@1=%.4f nDCG=%.4f n=%d",
                      label,
                      metrics["summary"]["MAP"],
                      metrics["summary"]["MRR"],
                      metrics["summary"].get("P@1", 0.0),
                      metrics["summary"]["nDCG"],
                      metrics["summary"]["n_queries"])
        return metrics, results

    # ------------------------------------------------------------------ #
    def _plots(self, clean_results, train_emb, noise_rows):
        # Confusion (top-1).
        true_labels = [r.true_label for r in clean_results]
        pred_labels = [r.predictions[0].label if r.predictions else "" for r in clean_results]
        if true_labels:
            plot_confusion_matrix(
                true_labels, pred_labels, self.run_dir / "confusion.png",
                max_classes=30,
            )
            save_csv(
                [{"true": t, "pred": p, "count": c}
                 for t, p, c in top_confusions_table(true_labels, pred_labels)],
                self.run_dir / "top_confusions.csv",
            )

        # Embedding scatter (sample of segment-level embeddings).
        if train_emb:
            embs, labels = [], []
            total_rows = 0
            target_rows = 2000
            for rec, emb_source in train_emb:
                if total_rows >= target_rows:
                    break
                if isinstance(emb_source, Path):
                    with np.load(emb_source) as z:
                        mat = z["embeddings"].astype(np.float32)
                else:
                    mat = emb_source
                if mat.shape[0] == 0:
                    continue
                take = min(mat.shape[0], target_rows - total_rows)
                sample = mat[:take]
                embs.append(sample)
                labels.extend([rec.primary_label] * sample.shape[0])
                total_rows += sample.shape[0]
            X = np.concatenate(embs, axis=0) if embs else None
            try:
                if X is not None:
                    plot_embedding_2d(X, labels, self.run_dir / "embeddings_2d.png")
            except Exception as exc:  # plotting failures are non-fatal
                self.log.warning("Embedding 2D plot failed: %s", exc)

        # Noise robustness curves.
        if noise_rows:
            plot_noise_robustness(noise_rows, "MAP", self.run_dir / "noise_robustness_MAP.png")
            plot_noise_robustness(noise_rows, "MRR", self.run_dir / "noise_robustness_MRR.png")


def _config_to_dict(cfg: ExperimentConfig) -> dict:
    """Manual asdict — Path objects need stringification, frozen dataclasses are nested."""
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "data": {**asdict(cfg.data), "dataset_zip": str(cfg.data.dataset_zip),
                 "audio_dir": str(cfg.data.audio_dir)},
        "preprocessing": asdict(cfg.preprocessing),
        "segmentation": asdict(cfg.segmentation),
        "embedding": asdict(cfg.embedding),
        "aggregation": asdict(cfg.aggregation),
        "representation": asdict(cfg.representation),
        "indexing": asdict(cfg.indexing),
        "fusion": asdict(cfg.fusion),
        "ranking": asdict(cfg.ranking),
        "similarity": cfg.similarity,
        "evaluation": {
            "top_k": cfg.evaluation.top_k,
            "p_at": cfg.evaluation.p_at,
            "recall_at": cfg.evaluation.recall_at,
            "noise": asdict(cfg.evaluation.noise),
        },
        "notes": cfg.notes,
    }
