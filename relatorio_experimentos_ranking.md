# Relatório dos Experimentos de Ranking

Este documento consolida os experimentos que aparecem nos logs do workspace, com foco em:

- qual modelo foi usado em cada execução;
- qual metodologia de ranking estava ativa;
- quais métricas foram reportadas;
- como cada tentativa foi estruturada para melhorar o ranking.

Os logs usados como referência são:

- [log_.txt](log_.txt)
- [log_datasetinteiro.txt](log_datasetinteiro.txt)
- [log_datasetinteiro2.txt](log_datasetinteiro2.txt)
- [log_dataset_3.txt](log_dataset_3.txt)
- [log_dataset_5.txt](log_dataset_5.txt)
- [log_dataset_6.txt](log_dataset_6.txt)
- [log_dataset_7.txt](log_dataset_7.txt)

## Leitura Rápida

O projeto compara duas arquiteturas-base e várias estratégias de reranking:

- **Strategy 1**: índice por segmento, fusão tardia, normalmente com `rrf`.
- **Strategy 2**: super-embedding por áudio, fusão precoce, normalmente com `max`.
- **Rankers avançados**: `softmax`, `attention`, `rrf`, `borda`, `topk_mean`, `weighted_topk`, `median`, `threshold`, `hit`, `taxonomy_boost` e híbridos.

Em termos práticos, as tentativas para melhorar o ranking seguem três ideias principais:

1. combinar melhor as evidências de vários segmentos do áudio;
2. suavizar ruído ou reforçar evidências fortes;
3. trazer espécies biologicamente próximas para mais perto do topo.

## 1. Comparação Base: `strategy_compare`

### O que esse conjunto mede

Esse conjunto compara duas formas de indexar e buscar o mesmo conjunto de áudios, sempre com o mesmo backbone `perch_v2_torch`:

- `strategy1_segments_torch`: usa documentos por segmento, segmentação sobreposta de 5 s com hop de 2,5 s, índice HNSW e fusão tardia com `rrf`.
- `strategy2_super_embedding_torch`: usa um documento por áudio, segmentação sobreposta de 5 s com hop de 2,5 s, índice flat e fusão precoce com `max`.

### `log_.txt`

Esse é o menor teste registrado. Ele foi feito com um subconjunto pequeno do corpus, por isso os números são mais baixos e a latência também é pequena.

**Modelo usado**: `perch_v2_torch`

**Metodologia de ranking**:

- `strategy1_segments_torch`: late fusion + `rrf`
- `strategy2_super_embedding_torch`: early fusion + `max`

| Métrica | strategy1_segments_torch | strategy2_super_embedding_torch |
|---|---:|---:|
| MAP | 0.6183 | 0.6335 |
| MRR | 0.6183 | 0.6335 |
| P@1 | 0.5359 | 0.5654 |
| P@5 | 0.1440 | 0.1464 |
| R@1 | 0.0086 | 0.1011 |
| R@5 | 0.0121 | 0.1299 |
| R@10 | 0.0137 | 0.1405 |
| nDCG | 0.6604 | 0.6705 |
| latency | 3.7 / p95 6.0 ms | 0.3 / p95 0.3 ms |

**Leitura**: aqui `strategy2_super_embedding_torch` ficou melhor em praticamente todas as métricas de ranking, e também foi muito mais rápido.

### `log_datasetinteiro.txt`

Essa execução já usa um subconjunto maior. Ela mantém a mesma comparação entre segmentação tardia e super-embedding.

**Modelo usado**: `perch_v2_torch`

**Metodologia de ranking**:

- `strategy1_segments_torch`: late fusion + `rrf`
- `strategy2_super_embedding_torch`: early fusion + `max`

| Métrica | strategy1_segments_torch | strategy2_super_embedding_torch |
|---|---:|---:|
| MAP | 0.8050 | 0.8008 |
| MRR | 0.8050 | 0.8008 |
| P@1 | 0.7574 | 0.7497 |
| P@5 | 0.2253 | 0.1730 |
| R@1 | 0.0011 | 0.0129 |
| R@5 | 0.0013 | 0.0150 |
| R@10 | 0.0014 | 0.0156 |
| nDCG | 0.8280 | 0.8245 |
| latency | 8.6 / p95 12.4 ms | 2.6 / p95 2.7 ms |

**Leitura**: nesse run, `strategy1_segments_torch` passou a ganhar em MAP, P@1, P@5 e nDCG, enquanto `strategy2_super_embedding_torch` continuou sendo mais rápido.

### `log_datasetinteiro2.txt`

Essa é a execução mais completa desse trio. Ela confirma o comportamento do mesmo par de estratégias em um conjunto maior ainda.

**Modelo usado**: `perch_v2_torch`

**Metodologia de ranking**:

- `strategy1_segments_torch`: late fusion + `rrf`
- `strategy2_super_embedding_torch`: early fusion + `max`

| Métrica | strategy1_segments_torch | strategy2_super_embedding_torch |
|---|---:|---:|
| MAP | 0.8191 | 0.8227 |
| MRR | 0.8191 | 0.8227 |
| P@1 | 0.7697 | 0.7729 |
| P@5 | 0.2518 | 0.2207 |
| R@1 | 0.0004 | 0.0054 |
| R@5 | 0.0005 | 0.0062 |
| R@10 | 0.0006 | 0.0065 |
| nDCG | 0.8417 | 0.8457 |
| latency | 22.6 / p95 33.3 ms | 6.3 / p95 6.4 ms |

**Leitura**: aqui `strategy2_super_embedding_torch` ficou levemente melhor em MAP, MRR, P@1 e nDCG, mas `strategy1_segments_torch` ainda manteve melhor P@5. Isso sugere que o super-embedding otimiza melhor a cabeça do ranking, enquanto a estratégia por segmento preserva melhor a lista expandida.

## 2. Estratégia, Ruído e Segmentação: `strategy_compare_giant`

### O que esse conjunto mede

O log [log_dataset_3.txt](log_dataset_3.txt) compara seis variantes do pipeline com o mesmo backbone `perch_v2_torch`, mudando três eixos ao mesmo tempo:

- segmentação com overlap ou sem overlap;
- presença ou ausência de `spectral_gating`;
- representação por segmento ou por áudio;
- late fusion com `rrf` ou early fusion com `max`.

### Metodologia de ranking

O ranking não é o fator que muda aqui. O que muda é a arquitetura do pipeline:

- `strategy1_*` usa `segment` + `rrf` em fusão tardia;
- `strategy2_*` usa `audio` + `max` em fusão precoce.

### Resultados completos

| Métrica | strategy1_segments_torch | strategy2_super_embedding_torch | strategy1_segments_no_overlap_torch | strategy1_segments_no_overlap_noise_torch | strategy2_super_embedding_no_overlap_torch | strategy2_super_embedding_no_overlap_noise_torch |
|---|---:|---:|---:|---:|---:|---:|
| MAP | 0.8194 | 0.8227 | 0.7980 | 0.7489 | 0.7929 | 0.7607 |
| MRR | 0.8194 | 0.8227 | 0.7980 | 0.7489 | 0.7929 | 0.7607 |
| P@1 | 0.7697 | 0.7729 | 0.7424 | 0.6865 | 0.7397 | 0.7067 |
| P@5 | 0.2509 | 0.2207 | 0.2065 | 0.1894 | 0.2120 | 0.1964 |
| R@1 | 0.0004 | 0.0054 | 0.0007 | 0.0007 | 0.0051 | 0.0049 |
| R@5 | 0.0005 | 0.0062 | 0.0009 | 0.0008 | 0.0061 | 0.0058 |
| R@10 | 0.0006 | 0.0065 | 0.0009 | 0.0009 | 0.0064 | 0.0060 |
| nDCG | 0.8422 | 0.8457 | 0.8240 | 0.7791 | 0.8183 | 0.7863 |
| latency | 21.9 / p95 28.5 ms | 6.2 / p95 6.4 ms | 14.3 / p95 17.8 ms | 14.4 / p95 20.7 ms | 6.3 / p95 6.4 ms | 6.3 / p95 6.5 ms |
| n_documents | 363917 | 27309 | 202228 | 202228 | 27309 | 27309 |
| index_build_time_s | 47.8 | 22.2 | 34.6 | 33.2 | 15.9 | 15.9 |
| embedding_dim | 1280 | 1280 | 1280 | 1280 | 1280 | 1280 |

### Leitura prática

Esse conjunto mostra que:

- remover overlap piora a qualidade em geral;
- aplicar `spectral_gating` reduz ainda mais o desempenho quando a configuração já está sensível ao ruído;
- o super-embedding continua mais rápido, mas nem sempre é o melhor em P@5;
- a estratégia por segmento preserva melhor a lista quando o ranking precisa de mais contexto.

## 3. Sweep de Rankers: `ranking_suite_advanced`

### O que esse conjunto mede

Os logs [log_dataset_5.txt](log_dataset_5.txt) e [log_dataset_6.txt](log_dataset_6.txt) vêm do mesmo tipo de experimento: um sweep de rankers sobre os mesmos configs base.

Os dois configs-base avaliados são:

- `strategy1_segments_torch`
- `strategy2_super_embedding_torch`

O script testa estratégias de reranking simples, híbridas e ensemble. Além das métricas usuais, ele mede:

- `genus@5` e `genus@10`;
- `common@5` e `common@10`.

Essas duas medidas são importantes porque o objetivo não é só acertar a espécie exata, mas também trazer candidatos taxonomicamente próximos ou com nome comum semelhante.

### Metodologia de ranking usada nesse sweep

Rankers simples testados:

- `softmax`
- `attention`
- `rrf`
- `borda`
- `topk_mean`
- `weighted_topk`
- `taxonomy_boost`

Híbridos e ensembles testados:

- `hybrid_attention_rrf_taxonomy`
- `hybrid_softmax_rrf_taxonomy`
- `hybrid_attention_softmax_taxonomy`
- `hybrid_rrf_borda_taxonomy`
- `consensus_ensemble`
- `taxonomy_boost_strong_ensemble`

Essas combinações misturam três ideias:

1. score-based reranking com `softmax` ou `attention`;
2. rank-based fusion com `rrf` ou `borda`;
3. reforço taxonômico com `taxonomy_boost`.

### Como interpretar os híbridos

- `hybrid_attention_rrf_taxonomy`: dá peso maior para a atenção, mas ainda mistura fusão por rank e bônus taxonômico.
- `hybrid_softmax_rrf_taxonomy`: troca a atenção por softmax, mantendo RRF e taxonomia.
- `hybrid_attention_softmax_taxonomy`: combina duas formas de suavização de score e soma um bônus taxonômico.
- `hybrid_rrf_borda_taxonomy`: privilegia fusão de rank, com uma segunda regra clássica de votação e reforço taxonômico.
- `consensus_ensemble`: junta vários rankers simples para tentar reduzir o viés de uma única regra.
- `taxonomy_boost_strong_ensemble`: mantém atenção, RRF e um reforço taxonômico mais agressivo.

### `log_dataset_5.txt`

Essa execução mostra o sweep completo com métricas de classificação, taxonomia e latência.

**Modelo usado**: `perch_v2_torch`

**Configuração-base**:

- `strategy1_segments_torch`: segmentação sobreposta + `segment` + late fusion + `rrf`
- `strategy2_super_embedding_torch`: segmentação sobreposta + `audio` + early fusion + `max`

| experiment | ranker | MAP | MRR | P@1 | P@5 | R@1 | R@5 | R@10 | nDCG | genus@5 | genus@10 | common@5 | common@10 | latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strategy1_segments_torch | softmax | 0.8693 | 0.8693 | 0.8257 | 0.2586 | 0.0005 | 0.0006 | 0.0006 | 0.8870 | 0.9268 | 0.9441 | 0.9328 | 0.9505 | 12.8 |
| strategy1_segments_torch | attention | 0.8709 | 0.8709 | 0.8293 | 0.2582 | 0.0005 | 0.0006 | 0.0006 | 0.8879 | 0.9249 | 0.9432 | 0.9314 | 0.9500 | 12.5 |
| strategy1_segments_torch | rrf | 0.8160 | 0.8160 | 0.7647 | 0.2499 | 0.0004 | 0.0005 | 0.0006 | 0.8389 | 0.8841 | 0.9138 | 0.8931 | 0.9238 | 11.5 |
| strategy1_segments_torch | borda | 0.8158 | 0.8158 | 0.7644 | 0.2501 | 0.0004 | 0.0005 | 0.0006 | 0.8391 | 0.8851 | 0.9156 | 0.8944 | 0.9251 | 13.6 |
| strategy1_segments_torch | topk_mean | 0.7646 | 0.7646 | 0.7089 | 0.2406 | 0.0004 | 0.0005 | 0.0005 | 0.7902 | 0.8379 | 0.8764 | 0.8546 | 0.8933 | 13.7 |
| strategy1_segments_torch | weighted_topk | 0.7673 | 0.7673 | 0.7125 | 0.2408 | 0.0004 | 0.0005 | 0.0005 | 0.7918 | 0.8386 | 0.8733 | 0.8540 | 0.8913 | 13.9 |
| strategy1_segments_torch | taxonomy_boost | 0.8637 | 0.8637 | 0.8184 | 0.2038 | 0.0004 | 0.0006 | 0.0006 | 0.8822 | 0.9239 | 0.9416 | 0.9301 | 0.9489 | 13.7 |
| strategy1_segments_torch | hybrid_attention_rrf_taxonomy | 0.8628 | 0.8628 | 0.8195 | 0.2027 | 0.0005 | 0.0006 | 0.0006 | 0.8806 | 0.9179 | 0.9384 | 0.9244 | 0.9450 | 0.9 |
| strategy2_super_embedding_torch | softmax | 0.8436 | 0.8436 | 0.7925 | 0.2257 | 0.0051 | 0.0062 | 0.0065 | 0.8656 | 0.9141 | 0.9375 | 0.9211 | 0.9443 | 8.4 |
| strategy2_super_embedding_torch | attention | 0.8494 | 0.8494 | 0.8041 | 0.2255 | 0.0053 | 0.0063 | 0.0065 | 0.8694 | 0.9131 | 0.9352 | 0.9217 | 0.9429 | 9.4 |
| strategy2_super_embedding_torch | rrf | 0.8227 | 0.8227 | 0.7729 | 0.2207 | 0.0054 | 0.0062 | 0.0065 | 0.8457 | 0.8907 | 0.9219 | 0.9015 | 0.9331 | 9.0 |
| strategy2_super_embedding_torch | borda | 0.8227 | 0.8227 | 0.7729 | 0.2207 | 0.0054 | 0.0062 | 0.0065 | 0.8457 | 0.8907 | 0.9219 | 0.9015 | 0.9331 | 5.4 |
| strategy2_super_embedding_torch | topk_mean | 0.8173 | 0.8173 | 0.7681 | 0.2196 | 0.0053 | 0.0062 | 0.0065 | 0.8406 | 0.8843 | 0.9188 | 0.8954 | 0.9307 | 5.5 |
| strategy2_super_embedding_torch | weighted_topk | 0.8051 | 0.8051 | 0.7532 | 0.2180 | 0.0052 | 0.0062 | 0.0064 | 0.8297 | 0.8762 | 0.9125 | 0.8890 | 0.9266 | 5.6 |
| strategy2_super_embedding_torch | taxonomy_boost | 0.8200 | 0.8200 | 0.7561 | 0.1906 | 0.0044 | 0.0060 | 0.0064 | 0.8467 | 0.9055 | 0.9331 | 0.9127 | 0.9405 | 5.6 |
| strategy2_super_embedding_torch | hybrid_attention_rrf_taxonomy | 0.8476 | 0.8476 | 0.8029 | 0.1907 | 0.0053 | 0.0063 | 0.0065 | 0.8672 | 0.9069 | 0.9324 | 0.9159 | 0.9410 | 0.3 |

**Leitura**:

- em `strategy1_segments_torch`, `attention` e `softmax` foram os melhores rankers puros em MAP/MRR;
- `taxonomy_boost` também ficou forte, principalmente quando a ideia é aproximar espécies do mesmo gênero;
- os híbridos reduziram o gap entre qualidade e robustez, com latência muito baixa nos casos mais leves;
- em `strategy2_super_embedding_torch`, `attention` e `hybrid_attention_rrf_taxonomy` ficaram entre os melhores;
- `topk_mean` e `weighted_topk` suavizam, mas tendem a perder para os melhores híbridos.

### `log_dataset_6.txt`

Esse log registra uma segunda execução do mesmo sweep, com números um pouco diferentes. A metodologia é a mesma, então ele serve como repetição útil para ver estabilidade.

**Modelo usado**: `perch_v2_torch`

**Configuração-base**:

- `strategy1_segments_torch`
- `strategy2_super_embedding_torch`

| experiment | ranker | MAP | MRR | P@1 | P@5 | R@1 | R@5 | R@10 | nDCG | genus@5 | genus@10 | common@5 | common@10 | latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strategy1_segments_torch | softmax | 0.8682 | 0.8682 | 0.8252 | 0.2588 | 0.0005 | 0.0006 | 0.0006 | 0.8856 | 0.9242 | 0.9419 | 0.9309 | 0.9486 | 12.6 |
| strategy1_segments_torch | attention | 0.8689 | 0.8689 | 0.8269 | 0.2584 | 0.0005 | 0.0006 | 0.0006 | 0.8862 | 0.9223 | 0.9424 | 0.9290 | 0.9492 | 12.5 |
| strategy1_segments_torch | attention | 0.8682 | 0.8682 | 0.8253 | 0.2588 | 0.0005 | 0.0006 | 0.0006 | 0.8855 | 0.9241 | 0.9415 | 0.9305 | 0.9485 | 11.9 |
| strategy1_segments_torch | rrf | 0.8179 | 0.8179 | 0.7691 | 0.2507 | 0.0004 | 0.0005 | 0.0005 | 0.8406 | 0.8844 | 0.9154 | 0.8935 | 0.9254 | 11.9 |
| strategy1_segments_torch | rrf | 0.8256 | 0.8256 | 0.7767 | 0.2522 | 0.0004 | 0.0005 | 0.0006 | 0.8480 | 0.8913 | 0.9217 | 0.8998 | 0.9312 | 11.8 |
| strategy1_segments_torch | borda | 0.8170 | 0.8170 | 0.7672 | 0.2504 | 0.0004 | 0.0005 | 0.0005 | 0.8403 | 0.8832 | 0.9175 | 0.8920 | 0.9271 | 11.9 |
| strategy1_segments_torch | topk_mean | 0.7632 | 0.7632 | 0.7071 | 0.2414 | 0.0004 | 0.0005 | 0.0005 | 0.7890 | 0.8375 | 0.8755 | 0.8541 | 0.8929 | 12.0 |
| strategy1_segments_torch | topk_mean | 0.7679 | 0.7679 | 0.7137 | 0.2419 | 0.0004 | 0.0005 | 0.0005 | 0.7921 | 0.8398 | 0.8730 | 0.8552 | 0.8903 | 12.0 |
| strategy1_segments_torch | weighted_topk | 0.7662 | 0.7662 | 0.7112 | 0.2417 | 0.0004 | 0.0005 | 0.0005 | 0.7908 | 0.8385 | 0.8729 | 0.8541 | 0.8906 | 12.1 |
| strategy1_segments_torch | weighted_topk | 0.7683 | 0.7683 | 0.7156 | 0.2415 | 0.0004 | 0.0005 | 0.0005 | 0.7921 | 0.8383 | 0.8708 | 0.8535 | 0.8887 | 12.1 |
| strategy1_segments_torch | taxonomy_boost | 0.8628 | 0.8628 | 0.8180 | 0.2033 | 0.0004 | 0.0006 | 0.0006 | 0.8813 | 0.9207 | 0.9409 | 0.9271 | 0.9482 | 12.0 |
| strategy1_segments_torch | taxonomy_boost | 0.8628 | 0.8628 | 0.8180 | 0.2033 | 0.0004 | 0.0006 | 0.0006 | 0.8813 | 0.9207 | 0.9409 | 0.9271 | 0.9482 | 12.0 |
| strategy1_segments_torch | taxonomy_boost | 0.8628 | 0.8628 | 0.8180 | 0.2033 | 0.0004 | 0.0006 | 0.0006 | 0.8813 | 0.9207 | 0.9410 | 0.9271 | 0.9482 | 12.0 |
| strategy1_segments_torch | hybrid_attention_rrf_taxonomy | 0.8622 | 0.8622 | 0.8202 | 0.2025 | 0.0005 | 0.0005 | 0.0006 | 0.8801 | 0.9163 | 0.9387 | 0.9236 | 0.9451 | 0.8 |
| strategy1_segments_torch | hybrid_softmax_rrf_taxonomy | 0.8615 | 0.8615 | 0.8187 | 0.2026 | 0.0004 | 0.0005 | 0.0006 | 0.8794 | 0.9169 | 0.9378 | 0.9244 | 0.9446 | 0.8 |
| strategy1_segments_torch | hybrid_attention_softmax_taxonomy | 0.8691 | 0.8691 | 0.8263 | 0.2041 | 0.0005 | 0.0006 | 0.0006 | 0.8865 | 0.9247 | 0.9431 | 0.9309 | 0.9498 | 1.0 |
| strategy1_segments_torch | hybrid_rrf_borda_taxonomy | 0.8383 | 0.8383 | 0.7940 | 0.1984 | 0.0004 | 0.0005 | 0.0006 | 0.8605 | 0.8964 | 0.9343 | 0.9049 | 0.9424 | 0.7 |
| strategy1_segments_torch | consensus_ensemble | 0.8629 | 0.8629 | 0.8211 | 0.2026 | 0.0005 | 0.0005 | 0.0006 | 0.8805 | 0.9172 | 0.9380 | 0.9247 | 0.9446 | 1.2 |
| strategy1_segments_torch | taxonomy_boost_strong_ensemble | 0.8633 | 0.8633 | 0.8211 | 0.2028 | 0.0005 | 0.0005 | 0.0006 | 0.8810 | 0.9178 | 0.9387 | 0.9255 | 0.9451 | 0.8 |
| strategy2_super_embedding_torch | softmax | 0.8436 | 0.8436 | 0.7925 | 0.2257 | 0.0051 | 0.0062 | 0.0065 | 0.8656 | 0.9141 | 0.9375 | 0.9211 | 0.9443 | 5.0 |
| strategy2_super_embedding_torch | attention | 0.8494 | 0.8494 | 0.8041 | 0.2255 | 0.0053 | 0.0063 | 0.0065 | 0.8694 | 0.9131 | 0.9352 | 0.9217 | 0.9429 | 5.1 |
| strategy2_super_embedding_torch | attention | 0.8436 | 0.8436 | 0.7925 | 0.2257 | 0.0051 | 0.0062 | 0.0065 | 0.8656 | 0.9141 | 0.9375 | 0.9211 | 0.9443 | 5.0 |
| strategy2_super_embedding_torch | rrf | 0.8227 | 0.8227 | 0.7729 | 0.2207 | 0.0054 | 0.0062 | 0.0065 | 0.8457 | 0.8907 | 0.9219 | 0.9015 | 0.9331 | 5.0 |
| strategy2_super_embedding_torch | rrf | 0.8227 | 0.8227 | 0.7729 | 0.2207 | 0.0054 | 0.0062 | 0.0065 | 0.8457 | 0.8907 | 0.9219 | 0.9015 | 0.9331 | 5.0 |
| strategy2_super_embedding_torch | borda | 0.8227 | 0.8227 | 0.7729 | 0.2207 | 0.0054 | 0.0062 | 0.0065 | 0.8457 | 0.8907 | 0.9219 | 0.9015 | 0.9331 | 5.0 |
| strategy2_super_embedding_torch | topk_mean | 0.8173 | 0.8173 | 0.7681 | 0.2196 | 0.0053 | 0.0062 | 0.0065 | 0.8406 | 0.8843 | 0.9188 | 0.8954 | 0.9307 | 5.1 |
| strategy2_super_embedding_torch | topk_mean | 0.7973 | 0.7973 | 0.7434 | 0.2172 | 0.0051 | 0.0062 | 0.0064 | 0.8227 | 0.8729 | 0.9074 | 0.8863 | 0.9223 | 5.2 |
| strategy2_super_embedding_torch | weighted_topk | 0.8051 | 0.8051 | 0.7532 | 0.2180 | 0.0052 | 0.0062 | 0.0064 | 0.8297 | 0.8762 | 0.9125 | 0.8890 | 0.9266 | 5.2 |
| strategy2_super_embedding_torch | weighted_topk | 0.7862 | 0.7862 | 0.7273 | 0.2165 | 0.0050 | 0.0062 | 0.0064 | 0.8139 | 0.8691 | 0.9058 | 0.8827 | 0.9206 | 5.2 |
| strategy2_super_embedding_torch | taxonomy_boost | 0.8200 | 0.8200 | 0.7561 | 0.1906 | 0.0044 | 0.0060 | 0.0064 | 0.8467 | 0.9055 | 0.9331 | 0.9127 | 0.9405 | 5.2 |
| strategy2_super_embedding_torch | taxonomy_boost | 0.8197 | 0.8197 | 0.7558 | 0.1906 | 0.0043 | 0.0060 | 0.0064 | 0.8465 | 0.9052 | 0.9330 | 0.9124 | 0.9403 | 5.3 |
| strategy2_super_embedding_torch | taxonomy_boost | 0.8200 | 0.8200 | 0.7561 | 0.1906 | 0.0044 | 0.0060 | 0.0064 | 0.8467 | 0.9056 | 0.9334 | 0.9129 | 0.9409 | 5.4 |
| strategy2_super_embedding_torch | hybrid_attention_rrf_taxonomy | 0.8476 | 0.8476 | 0.8029 | 0.1907 | 0.0053 | 0.0063 | 0.0065 | 0.8672 | 0.9069 | 0.9324 | 0.9159 | 0.9410 | 0.3 |
| strategy2_super_embedding_torch | hybrid_softmax_rrf_taxonomy | 0.8440 | 0.8440 | 0.7949 | 0.1917 | 0.0051 | 0.0063 | 0.0065 | 0.8648 | 0.9119 | 0.9330 | 0.9204 | 0.9413 | 0.3 |
| strategy2_super_embedding_torch | hybrid_attention_softmax_taxonomy | 0.8467 | 0.8467 | 0.7968 | 0.1926 | 0.0052 | 0.0063 | 0.0065 | 0.8679 | 0.9156 | 0.9371 | 0.9229 | 0.9440 | 0.3 |
| strategy2_super_embedding_torch | hybrid_rrf_borda_taxonomy | 0.8369 | 0.8369 | 0.7898 | 0.1889 | 0.0049 | 0.0063 | 0.0066 | 0.8586 | 0.8979 | 0.9309 | 0.9080 | 0.9400 | 0.3 |
| strategy2_super_embedding_torch | consensus_ensemble | 0.8462 | 0.8462 | 0.7987 | 0.1915 | 0.0052 | 0.0063 | 0.0065 | 0.8663 | 0.9106 | 0.9327 | 0.9189 | 0.9410 | 0.4 |
| strategy2_super_embedding_torch | taxonomy_boost_strong_ensemble | 0.8462 | 0.8462 | 0.7985 | 0.1917 | 0.0051 | 0.0063 | 0.0065 | 0.8666 | 0.9121 | 0.9330 | 0.9207 | 0.9413 | 0.3 |

**Leitura**:

- em `strategy1_segments_torch`, `attention` e `softmax` continuam muito fortes, com `taxonomy_boost` bem próximo;
- os híbridos com atenção + softmax + taxonomia formam o grupo mais consistente;
- em `strategy2_super_embedding_torch`, `attention`, `softmax`, `rrf` e `borda` ficam mais próximos entre si, mas `attention` ainda tende a liderar;
- `taxonomy_boost_strong_ensemble` e `consensus_ensemble` funcionam como tentativas de unir estabilidade e semântica biológica.

## 4. Sweep com BirdNET v3.0: `ranking_sweep_all`

### O que esse conjunto mede

O log [log_dataset_7.txt](log_dataset_7.txt) repete a mesma lógica de sweep, mas agora com `birdnet_v3` como backbone de embedding.

Os configs-base são:

- `strategy1_segments_birdnet_v3`
- `strategy2_super_embedding_birdnet_v3`

### Modelo usado

- `birdnet_v3`
- checkpoint baixado via URL no YAML
- janela nativa do embedder: 3 s a 32 kHz

### Metodologia de ranking usada nesse sweep

Os rankers testados são os mesmos da suíte avançada, mas sem os híbridos customizados. O foco aqui é ver como o BirdNET v3.0 se comporta com:

- `segment`
- `mean`
- `max`
- `topk_mean`
- `hit`
- `median`
- `threshold`
- `weighted_topk`
- `softmax`
- `rrf`
- `borda`
- `attention`
- `taxonomy_boost`

### Resultados completos

| base_config | ranking | MAP | MRR | P@1 | P@5 | R@1 | R@5 | R@10 | nDCG | latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strategy1_segments_birdnet_v3 | segment | 0.7326 | 0.7326 | 0.6585 | 0.1975 | 0.0004 | 0.0005 | 0.0005 | 0.7681 | 11.5 / 16.6 |
| strategy1_segments_birdnet_v3 | mean | 0.2520 | 0.2520 | 0.1671 | 0.1057 | 0.0001 | 0.0002 | 0.0003 | 0.2972 | 17.3 / 23.9 |
| strategy1_segments_birdnet_v3 | max | 0.7322 | 0.7322 | 0.6578 | 0.1977 | 0.0004 | 0.0005 | 0.0005 | 0.7677 | 16.5 / 22.5 |
| strategy1_segments_birdnet_v3 | topk_mean | 0.7707 | 0.7707 | 0.7083 | 0.2028 | 0.0004 | 0.0005 | 0.0006 | 0.8003 | 16.7 / 21.6 |
| strategy1_segments_birdnet_v3 | hit | 0.6743 | 0.6743 | 0.5649 | 0.1970 | 0.0003 | 0.0005 | 0.0005 | 0.7270 | 16.8 / 21.1 |
| strategy1_segments_birdnet_v3 | median | 0.2487 | 0.2487 | 0.1649 | 0.1041 | 0.0001 | 0.0002 | 0.0003 | 0.2934 | 17.8 / 22.7 |
| strategy1_segments_birdnet_v3 | threshold | 0.2532 | 0.2532 | 0.1694 | 0.1061 | 0.0001 | 0.0002 | 0.0003 | 0.2980 | 18.8 / 27.1 |
| strategy1_segments_birdnet_v3 | weighted_topk | 0.7743 | 0.7743 | 0.7146 | 0.2028 | 0.0004 | 0.0005 | 0.0005 | 0.8029 | 23.4 / 31.0 |
| strategy1_segments_birdnet_v3 | softmax | 0.8910 | 0.8910 | 0.8506 | 0.2207 | 0.0005 | 0.0006 | 0.0006 | 0.9078 | 22.4 / 27.6 |
| strategy1_segments_birdnet_v3 | rrf | 0.8312 | 0.8312 | 0.7775 | 0.2127 | 0.0005 | 0.0005 | 0.0006 | 0.8555 | 20.3 / 25.4 |
| strategy1_segments_birdnet_v3 | borda | 0.8280 | 0.8280 | 0.7722 | 0.2128 | 0.0004 | 0.0005 | 0.0006 | 0.8535 | 19.4 / 24.0 |
| strategy1_segments_birdnet_v3 | attention | 0.8944 | 0.8944 | 0.8549 | 0.2209 | 0.0005 | 0.0006 | 0.0006 | 0.9105 | 18.7 / 23.9 |
| strategy1_segments_birdnet_v3 | taxonomy_boost | 0.8894 | 0.8894 | 0.8478 | 0.1960 | 0.0005 | 0.0006 | 0.0006 | 0.9064 | 18.5 / 23.3 |
| strategy2_super_embedding_birdnet_v3 | segment | 0.8108 | 0.8108 | 0.7557 | 0.2501 | 0.0053 | 0.0063 | 0.0065 | 0.8371 | 6.1 / 6.2 |
| strategy2_super_embedding_birdnet_v3 | mean | 0.5163 | 0.5163 | 0.3557 | 0.2206 | 0.0029 | 0.0054 | 0.0061 | 0.5973 | 6.0 / 6.2 |
| strategy2_super_embedding_birdnet_v3 | max | 0.8108 | 0.8108 | 0.7557 | 0.2501 | 0.0053 | 0.0063 | 0.0065 | 0.8371 | 6.0 / 6.2 |
| strategy2_super_embedding_birdnet_v3 | topk_mean | 0.8076 | 0.8076 | 0.7539 | 0.2494 | 0.0052 | 0.0062 | 0.0064 | 0.8332 | 6.1 / 6.3 |
| strategy2_super_embedding_birdnet_v3 | hit | 0.2526 | 0.2526 | 0.1504 | 0.1481 | 0.0006 | 0.0021 | 0.0034 | 0.3217 | 6.9 / 11.8 |
| strategy2_super_embedding_birdnet_v3 | median | 0.4560 | 0.4560 | 0.3050 | 0.2055 | 0.0026 | 0.0049 | 0.0058 | 0.5382 | 6.5 / 6.9 |
| strategy2_super_embedding_birdnet_v3 | threshold | 0.5163 | 0.5163 | 0.3557 | 0.2206 | 0.0029 | 0.0054 | 0.0061 | 0.5973 | 6.2 / 6.6 |
| strategy2_super_embedding_birdnet_v3 | weighted_topk | 0.7916 | 0.7916 | 0.7358 | 0.2466 | 0.0051 | 0.0061 | 0.0064 | 0.8193 | 6.2 / 6.4 |
| strategy2_super_embedding_birdnet_v3 | softmax | 0.8298 | 0.8298 | 0.7786 | 0.2531 | 0.0046 | 0.0060 | 0.0064 | 0.8540 | 6.0 / 6.4 |
| strategy2_super_embedding_birdnet_v3 | rrf | 0.8108 | 0.8108 | 0.7557 | 0.2501 | 0.0053 | 0.0063 | 0.0065 | 0.8371 | 5.9 / 6.1 |
| strategy2_super_embedding_birdnet_v3 | borda | 0.8108 | 0.8108 | 0.7557 | 0.2501 | 0.0053 | 0.0063 | 0.0065 | 0.8371 | 6.3 / 7.4 |
| strategy2_super_embedding_birdnet_v3 | attention | 0.8373 | 0.8373 | 0.7881 | 0.2542 | 0.0048 | 0.0062 | 0.0065 | 0.8604 | 6.2 / 6.7 |
| strategy2_super_embedding_birdnet_v3 | taxonomy_boost | 0.8240 | 0.8240 | 0.7700 | 0.1965 | 0.0043 | 0.0058 | 0.0063 | 0.8493 | 6.2 / 6.7 |

### Leitura prática

Em BirdNET v3.0, o comportamento é bem claro:

- em `strategy1_segments_birdnet_v3`, `attention` e `softmax` são os melhores rankers em MAP/MRR;
- `taxonomy_boost` também fica muito forte, com apelo biológico maior na cauda da lista;
- em `strategy2_super_embedding_birdnet_v3`, `attention` lidera, seguido de `softmax` e `rrf`;
- `mean`, `median` e `threshold` são mais conservadores e tendem a perder qualidade;
- `hit` é útil como votação, mas aqui ficou bem abaixo das melhores opções.

## Conclusão Geral

O conjunto de logs mostra um padrão consistente:

- **segmentação + fusão tardia** funciona bem quando se quer explorar evidências de vários trechos do áudio;
- **super-embedding + fusão precoce** é mais rápido e, em alguns casos, melhora a cabeça do ranking;
- **attention** e **softmax** costumam ser as estratégias mais fortes para reranking;
- **rrf** e **borda** são bons quando a concordância entre listas importa mais do que o score bruto;
- **taxonomy_boost** é a melhor tentativa quando a meta é aproximar espécies biologicamente relacionadas;
- **hybrid** e **ensemble** tentam capturar o melhor dos mundos: score, rank e taxonomia ao mesmo tempo.

Se o objetivo for escolher uma direção para próximos testes, os melhores candidatos práticos, pelos logs, são:

1. `attention`
2. `softmax`
3. `taxonomy_boost`
4. `hybrid_attention_softmax_taxonomy`
5. `hybrid_attention_rrf_taxonomy`
