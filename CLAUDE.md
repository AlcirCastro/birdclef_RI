# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Two parallel codebases

- `src/` + `experiments/` + `configs/` — the **research framework**. Modular,
  config-driven, Perch v2 + FAISS. Read `README.md` for the user-facing tour.
- `bird_search/` — the **legacy Flask + Elasticsearch demo** (predates the
  framework). It uses log-mel statistics, not Perch v2. Independent: changes
  to one don't touch the other.

When the user is in framework territory, default new components into `src/`
and new experiments into `experiments/`. Only touch `bird_search/` if asked.

## Framework architecture (`src/`)

The whole framework hangs off the **registry pattern** in `src/utils/registry.py`.
Each stage package owns a `Registry` instance (`PREPROCESSORS`, `SEGMENTERS`,
`EMBEDDERS`, `AGGREGATORS`, `REPRESENTATIONS`, `INDEXES`, `FUSIONS`, `RANKERS`).
Components register themselves with `@<REGISTRY>.register("name")`. The
`PipelineBuilder` in `src/pipeline/builder.py` is the *only* place that maps
config strings to classes — there are no `if/elif` switches anywhere.

To register a new component: drop a module under the right package and
**add the import to that package's `__init__.py`** so the decorator fires.
That is the gotcha — registration is import-time, so a file that's never
imported never registers.

### Stage flow (in `ExperimentRunner.run`)

```
load → split → embed train → build doc_store → build index
                         ↘
                          embed test → fusion.queries → index.search → ranker.rank → evaluator
```

For each test record the runner: (1) preprocesses + segments + embeds,
(2) hands the query embedding matrix to `Retriever`, which (3) calls
`fusion.queries(...)` to decide whether to issue 1 query (early) or N
(late), (4) does kNN on the index, (5) calls `ranker.rank(...)` to fold
multiple result lists into a single per-label ranking.

### Document representations (`src/representation/`)

The doc store is what's actually in the index. Five flavors share the
`BaseDocumentStore` interface; the choice affects retrieval semantics:

- `segment` — every segment is a doc. Pairs naturally with late fusion.
- `audio` — one doc per file (segments aggregated). Smaller index.
- `species` — one doc per species. Fastest, drops within-species variability.
- `cluster` — k-means centroids per species. Captures multi-modal repertoires.
- `prototype` — single most-central segment per species.

The `aggregator` (mean/max/topk/median/attention) is consumed by `audio`,
`species`, `cluster`, `prototype`, and by `EarlyFusion` on the query side.
`segment` ignores it.

### Embedding cache

`EmbeddingCache` keys by `(path, size, mtime, embedder.name, preproc_id,
segmenter_id)` and stores .npz per audio. **Noise-injected queries
deliberately bypass the cache** — see `_embed_record` in the runner.

If you change the actual *behavior* of a registered component without
changing its registry key or params, you must invalidate manually
(`rm -rf birdclef_cache/embeddings/`) — the key only catches config
changes, not code changes inside a stage.

### Perch v2 specifics

`PerchV2Embedder` lazy-loads via `kagglehub` (preferred) or a custom
`tfhub_url`. Native input is **5 s of mono audio at 32 kHz** → ~1280-dim
vector. Segments shorter than 5 s are zero-padded; longer ones are
center-cropped. Window control belongs in the segmenter, not here.

The handle defaults to `google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier`;
override with `embedding.params.kaggle_model` / `kaggle_version` /
`tfhub_url` if Google publishes a new version.

Embeddings are L2-normalized inside the embedder, so cosine reduces to
inner product everywhere downstream.

## Running

```bash
# Framework
python experiments/exp1_baseline.py            # writes results/exp1_baseline/

# Legacy Flask demo
docker compose up -d                            # ES on :9200
python test.py                                  # serves Flask on :5000
```

`BIRDCLEF_MAX_RECORDS` env var controls subset size for the **legacy** app
only. The framework reads `data.max_records` from the YAML.

## Testing without dataset / TF

There is no test suite. To smoke-test individual modules without Perch v2 or
the 16 GB dataset:

- All registries are populated at package import time. `python -c "from
  src.ranking import RANKERS; print(list(RANKERS.keys()))"` is enough to
  catch import / registration breakage.
- Pipeline builder works without dataset; `ExperimentRunner.__init__` builds
  components but doesn't load audio. Audio is touched only when `run()` is
  called.

## Things to know before editing

- **Don't `from src.X import Y` inside a registered component module that's
  imported during `__init__.py` side-effect imports** — circular imports are
  easy. Pull the import inside the function body if needed.
- **`OutputConfig` paths are resolved against the YAML's grandparent**
  (`configs/foo.yaml` → repo root), see `src/config/loader.py`. New experiments
  with non-standard paths should pass an explicit `base_dir` to `load_config`.
- **`representation: segment` ignores the aggregator** in config but the
  builder still constructs one (cheap). Don't add error checks for "wrong"
  combinations — let the doc store decide.
- **Recall denominator** in `Evaluator` is the number of corpus docs sharing
  the query's label. With `species` representation that's 1, so recall
  collapses to top-1 hit-rate. With `segment` representation it's much
  larger. Compare like-with-like.
- **Latency** in `report.json` measures only the retrieval call (`Retriever.retrieve`),
  not embedding time. Index build time is reported separately as
  `index_build_time_s`.
