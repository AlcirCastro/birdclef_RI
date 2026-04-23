#!/usr/bin/env python3
from bird_search.embedding import Embedder
from bird_search.settings import load_settings
from pathlib import Path

settings = load_settings(Path('.'))
embedder = Embedder(settings)
print(f'✓ Embedder com PANNs inicializado com sucesso')
print(f'  Cache dir: {settings.embedding_cache_dir}')
print(f'  Index name: {settings.es_index}')
print(f'  Embedding name: {settings.embedding_name}')
