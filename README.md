# Bird Audio Similarity Search — Research Framework

A modular experimental framework for evaluating how preprocessing,
segmentation, document representation, indexing, ranking, and fusion
choices affect bird audio similarity retrieval. Embedding model is
fixed: **Google Perch v2**.

The framework is config-driven: every experiment is one YAML file plus a
~3-line launcher in `experiments/`. Components are swapped by changing
registry keys, never by editing pipeline code.

## Layout

```
configs/             one YAML per experiment
experiments/         exp*.py — thin orchestrators
src/
  utils/             registry, logging, seeding, IO
  config/            schema + YAML loader
  data/              records, dataset (zip), stratified split
  preprocessing/     identity, spectral_gating, bandpass, noise (injection)
  segmentation/      fixed, overlapping, multi_scale, energy, event
  embeddings/        perch_v2 (the only embedder), on-disk cache
  aggregation/       mean / max / topk / median / attention pooling
  representation/    segment / audio / species / cluster / prototype docs
  indexing/          flat / ivf / hnsw / ivfpq (FAISS)
  ranking/           segment / mean / max / topk / hit / median / threshold /
                     weighted_topk / softmax / rrf / borda / attention
  fusion/            early (pool query first), late (per-segment queries)
  retrieval/         Retriever — wires fusion + index + ranker
  evaluation/        MAP, MRR, P@K, R@K, nDCG, latency, memory
  stats/             Wilcoxon, Friedman, bootstrap CI
  visualization/     2-D embeddings, confusion, latency-vs-metric, noise plots
  pipeline/          builder (config → components), runner (lifecycle)
results/             per-experiment outputs (one folder per `name`)
logs/                per-experiment log file
```

## Setup

```bash
# 1. Python deps (FAISS, TF, etc.)
pip install -r requirements.txt

# 2. Authenticate Kaggle so kagglehub can download Perch v2
#    (https://www.kaggle.com/docs/api — drop kaggle.json in ~/.kaggle/)
```

## Running an experiment

```bash
python experiments/exp1_baseline.py
python experiments/exp2_noise_robustness.py
python experiments/exp3_segmentation_strategy.py

# Optional experimental PyTorch-backed Perch v2 export path
python experiments/strategy_compare.py --backend torch
```

Each writes to `results/<name>/`:

- `report.json` — config snapshot + summary metrics + memory/timing
- `per_query_clean.csv` — one row per test query
- `noise_robustness.csv` — present when the noise sweep is enabled
- `rankings_clean.json` — full top-k predictions per query
- `confusion.png`, `embeddings_2d.png`, `noise_robustness_*.png`
- `top_confusions.csv`

## Adding a new variant

Pick the stage (e.g., a new ranker), inherit from its `Base*`, and decorate:

```python
# src/ranking/my_ranker.py
from src.ranking import RANKERS
from src.ranking.base import BaseRanker

@RANKERS.register("my_ranker")
class MyRanker(BaseRanker):
    def rank(self, per_query_hits, k):
        ...
```

Import the new module from `src/ranking/__init__.py` so the side-effect
registration runs, then point any config at `ranking.type: my_ranker`.

## Adding a new experiment

```bash
cp configs/exp1_baseline.yaml configs/exp4_my_thing.yaml
cp experiments/exp1_baseline.py experiments/exp4_my_thing.py
# edit the YAML; the .py file just needs the matching stem
```

## Statistical comparison

```python
from src.stats import wilcoxon_signed_rank, bootstrap_ci

# Per-query AP from two runs (same seed, same test set):
import csv
def col(p, k): return [float(r[k]) for r in csv.DictReader(open(p))]

a = col("results/exp1_baseline/per_query_clean.csv", "AP")
b = col("results/exp3_segmentation_strategy/per_query_clean.csv", "AP")
print(wilcoxon_signed_rank(a, b))
print(bootstrap_ci(a))
```

## Notes

- Perch v2 is loaded lazily on first call to `embed_segments` so unit-testing
  upstream modules doesn't require TensorFlow.
- Embeddings are cached on disk keyed by `(audio_path, mtime, embedder.name,
  preprocessor_id, segmenter_id)` — switching any of those forces re-embed.
  Noise-injected queries always bypass the cache.
- `perch_v2_torch` is an experimental backend that exports the official
  Perch v2 SavedModel to ONNX once and then runs inference with PyTorch when
  the graph is convertible. If the official graph hits unsupported ops during
  export, it falls back to the SavedModel runtime so the experiment still runs.
- The legacy `bird_search/` Flask + Elasticsearch demo from before this
  framework still works and is independent of `src/`.
