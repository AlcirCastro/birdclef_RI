import time
from pathlib import Path

from bird_search.dataset import load_records, stratified_split
from bird_search.embedding import Embedder
from bird_search.search import SearchIndex
from bird_search.settings import load_settings
from bird_search.web import create_app


def build_application(base_dir: Path):
    print("\n" + "=" * 60)
    print("BirdCLEF Search - Initializing Application")
    print("=" * 60)
    
    settings = load_settings(base_dir)
    search_index = SearchIndex(settings)
    search_index.ensure_connected()

    print(f"\n[1/5] Loading dataset...")
    records = load_records(settings)
    print(f"      ✓ Loaded {len(records)} records")

    if not records:
        raise RuntimeError("No records loaded from dataset")

    print(f"\n[2/5] Stratified split (train/test {settings.train_ratio:.0%}/{1-settings.train_ratio:.0%})...")
    train, test = stratified_split(
        records,
        val_ratio=1.0 - settings.train_ratio,
        seed=settings.random_seed,
    )
    print(f"      ✓ Train: {len(train)}, Test: {len(test)}")

    embedder = Embedder(settings)
    print(f"\n[3/5] Generating embeddings...")
    print(f"      Backend: {embedder.runtime_backend}")
    print(f"      (This will show progress below)")

    start = time.perf_counter()
    indexed, skipped = search_index.index_records(train, embedder=embedder, reset=True)
    elapsed = time.perf_counter() - start

    print(f"\n[4/5] Indexing summary")
    print(f"      ✓ Indexed: {indexed}, Skipped: {skipped}")
    print(f"      ✓ Time: {elapsed:.1f}s")

    print(f"\n[5/5] Creating Flask app...")
    app = create_app(
        settings=settings,
        embedder=embedder,
        search_index=search_index,
        train_records=train,
        test_records=test,
    )
    print(f"      ✓ Ready!")

    print("\n" + "=" * 60)
    print(f"Application ready at http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    return app
