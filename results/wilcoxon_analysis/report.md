# Wilcoxon Analysis (alpha=0.05, Holm-Bonferroni)

## 1. BirdNET v3 vs Perch v2 (mesmo strategy + reranker)
| A | B | MAP(A) | MAP(B) | ΔMAP | W-stat | p_raw | p_adj | sig | winner |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| birdnet_v3__E1__softmax | perch_v2__E1__softmax | 0.8913 | 0.8691 | +0.0221 | 331149.0 | 2.0347e-11 | 5.9006e-10 | ✓ | birdnet_v3__E1__softmax |
| birdnet_v3__E1__attention | perch_v2__E1__attention | 0.8932 | 0.8695 | +0.0237 | 318492.0 | 7.7579e-13 | 2.4049e-11 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__rrf | perch_v2__E1__rrf | 0.8311 | 0.8187 | +0.0124 | 764602.0 | 1.9733e-03 | 2.7626e-02 | ✓ | birdnet_v3__E1__rrf |
| birdnet_v3__E1__borda | perch_v2__E1__borda | 0.8284 | 0.8180 | +0.0104 | 789318.5 | 1.0077e-02 | 1.0077e-01 | ✗ | birdnet_v3__E1__borda |
| birdnet_v3__E1__topk_mean | perch_v2__E1__topk_mean | 0.7711 | 0.7628 | +0.0083 | 1406639.0 | 4.5106e-02 | 3.1986e-01 | ✗ | birdnet_v3__E1__topk_mean |
| birdnet_v3__E1__taxonomy_boost | perch_v2__E1__taxonomy_boost | 0.8890 | 0.8635 | +0.0255 | 337787.0 | 1.4672e-14 | 4.9884e-13 | ✓ | birdnet_v3__E1__taxonomy_boost |
| birdnet_v3__E1__hybrid_att_sm_tax | perch_v2__E1__hybrid_att_sm_tax | 0.8913 | 0.8697 | +0.0216 | 334451.5 | 4.7427e-11 | 1.3280e-09 | ✓ | birdnet_v3__E1__hybrid_att_sm_tax |
| birdnet_v3__E1__hybrid_att_rrf_tax | perch_v2__E1__hybrid_att_rrf_tax | 0.8880 | 0.8623 | +0.0257 | 333328.5 | 2.8091e-14 | 9.2699e-13 | ✓ | birdnet_v3__E1__hybrid_att_rrf_tax |
| birdnet_v3__E2__softmax | perch_v2__E2__softmax | 0.8298 | 0.8436 | -0.0137 | 665242.0 | 2.2807e-04 | 3.6490e-03 | ✓ | perch_v2__E2__softmax |
| birdnet_v3__E2__attention | perch_v2__E2__attention | 0.8373 | 0.8494 | -0.0121 | 601923.0 | 1.0599e-03 | 1.5899e-02 | ✓ | perch_v2__E2__attention |
| birdnet_v3__E2__rrf | perch_v2__E2__rrf | 0.8108 | 0.8227 | -0.0119 | 807249.5 | 2.5800e-03 | 3.0960e-02 | ✓ | perch_v2__E2__rrf |
| birdnet_v3__E2__borda | perch_v2__E2__borda | 0.8108 | 0.8227 | -0.0119 | 807249.5 | 2.5800e-03 | 3.0960e-02 | ✓ | perch_v2__E2__borda |
| birdnet_v3__E2__topk_mean | perch_v2__E2__topk_mean | 0.8076 | 0.8173 | -0.0097 | 838151.5 | 1.4039e-02 | 1.2635e-01 | ✗ | perch_v2__E2__topk_mean |
| birdnet_v3__E2__taxonomy_boost | perch_v2__E2__taxonomy_boost | 0.8240 | 0.8200 | +0.0040 | 828367.5 | 3.1029e-01 | 9.7114e-01 | ✗ | birdnet_v3__E2__taxonomy_boost |
| birdnet_v3__E2__hybrid_att_sm_tax | perch_v2__E2__hybrid_att_sm_tax | 0.8320 | 0.8467 | -0.0147 | 628161.0 | 6.0387e-05 | 1.2077e-03 | ✓ | perch_v2__E2__hybrid_att_sm_tax |
| birdnet_v3__E2__hybrid_att_rrf_tax | perch_v2__E2__hybrid_att_rrf_tax | 0.8404 | 0.8476 | -0.0072 | 607990.0 | 3.9983e-02 | 3.1986e-01 | ✗ | perch_v2__E2__hybrid_att_rrf_tax |

*11/16 comparações significativas (p_adj < 0.05)*

## 2. E1 (late fusion) vs E2 (early fusion) (mesmo backbone + reranker)
| A | B | MAP(A) | MAP(B) | ΔMAP | W-stat | p_raw | p_adj | sig | winner |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| perch_v2__E1__softmax | perch_v2__E2__softmax | 0.8691 | 0.8436 | +0.0255 | 198842.0 | 5.3500e-23 | 2.3540e-21 | ✓ | perch_v2__E1__softmax |
| perch_v2__E1__attention | perch_v2__E2__attention | 0.8695 | 0.8494 | +0.0200 | 185843.5 | 1.7741e-15 | 6.2094e-14 | ✓ | perch_v2__E1__attention |
| perch_v2__E1__rrf | perch_v2__E2__rrf | 0.8187 | 0.8227 | -0.0040 | 423677.5 | 2.4278e-01 | 9.7114e-01 | ✗ | perch_v2__E2__rrf |
| perch_v2__E1__borda | perch_v2__E2__borda | 0.8180 | 0.8227 | -0.0048 | 424071.0 | 1.5636e-01 | 7.8179e-01 | ✗ | perch_v2__E2__borda |
| perch_v2__E1__topk_mean | perch_v2__E2__topk_mean | 0.7628 | 0.8173 | -0.0545 | 370079.0 | 9.2496e-56 | 4.8098e-54 | ✓ | perch_v2__E2__topk_mean |
| perch_v2__E1__taxonomy_boost | perch_v2__E2__taxonomy_boost | 0.8635 | 0.8200 | +0.0436 | 214460.5 | 4.9152e-55 | 2.5067e-53 | ✓ | perch_v2__E1__taxonomy_boost |
| perch_v2__E1__hybrid_att_sm_tax | perch_v2__E2__hybrid_att_sm_tax | 0.8697 | 0.8467 | +0.0230 | 188869.0 | 8.7263e-20 | 3.4033e-18 | ✓ | perch_v2__E1__hybrid_att_sm_tax |
| perch_v2__E1__hybrid_att_rrf_tax | perch_v2__E2__hybrid_att_rrf_tax | 0.8623 | 0.8476 | +0.0147 | 196254.5 | 6.7720e-10 | 1.6930e-08 | ✓ | perch_v2__E1__hybrid_att_rrf_tax |
| birdnet_v3__E1__softmax | birdnet_v3__E2__softmax | 0.8913 | 0.8298 | +0.0614 | 121771.5 | 2.0241e-99 | 1.2347e-97 | ✓ | birdnet_v3__E1__softmax |
| birdnet_v3__E1__attention | birdnet_v3__E2__attention | 0.8932 | 0.8373 | +0.0558 | 122468.0 | 9.0579e-88 | 5.1630e-86 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__rrf | birdnet_v3__E2__rrf | 0.8311 | 0.8108 | +0.0202 | 440016.0 | 1.3254e-12 | 3.9763e-11 | ✓ | birdnet_v3__E1__rrf |
| birdnet_v3__E1__borda | birdnet_v3__E2__borda | 0.8284 | 0.8108 | +0.0175 | 469892.5 | 9.7508e-10 | 2.2427e-08 | ✓ | birdnet_v3__E1__borda |
| birdnet_v3__E1__topk_mean | birdnet_v3__E2__topk_mean | 0.7711 | 0.8076 | -0.0366 | 717496.0 | 2.3991e-22 | 1.0316e-20 | ✓ | birdnet_v3__E2__topk_mean |
| birdnet_v3__E1__taxonomy_boost | birdnet_v3__E2__taxonomy_boost | 0.8890 | 0.8240 | +0.0651 | 132972.0 | 2.0359e-106 | 1.3030e-104 | ✓ | birdnet_v3__E1__taxonomy_boost |
| birdnet_v3__E1__hybrid_att_sm_tax | birdnet_v3__E2__hybrid_att_sm_tax | 0.8913 | 0.8320 | +0.0593 | 124725.0 | 2.5428e-95 | 1.5257e-93 | ✓ | birdnet_v3__E1__hybrid_att_sm_tax |
| birdnet_v3__E1__hybrid_att_rrf_tax | birdnet_v3__E2__hybrid_att_rrf_tax | 0.8880 | 0.8404 | +0.0476 | 130884.0 | 7.3087e-75 | 3.9467e-73 | ✓ | birdnet_v3__E1__hybrid_att_rrf_tax |

*14/16 comparações significativas (p_adj < 0.05)*

## 3. Attention vs demais rerankers (por backbone + strategy)
| A | B | MAP(A) | MAP(B) | ΔMAP | W-stat | p_raw | p_adj | sig | winner |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| perch_v2__E1__attention | perch_v2__E1__softmax | 0.8695 | 0.8691 | +0.0003 | 29638.5 | 6.9811e-01 | 9.7114e-01 | ✗ | perch_v2__E1__attention |
| perch_v2__E1__attention | perch_v2__E1__rrf | 0.8695 | 0.8187 | +0.0508 | 89098.5 | 2.1527e-89 | 1.2701e-87 | ✓ | perch_v2__E1__attention |
| perch_v2__E1__attention | perch_v2__E1__borda | 0.8695 | 0.8180 | +0.0515 | 93503.5 | 5.2346e-89 | 3.0361e-87 | ✓ | perch_v2__E1__attention |
| perch_v2__E1__attention | perch_v2__E1__topk_mean | 0.8695 | 0.7628 | +0.1067 | 87739.0 | 3.9263e-182 | 2.6306e-180 | ✓ | perch_v2__E1__attention |
| perch_v2__E1__attention | perch_v2__E1__taxonomy_boost | 0.8695 | 0.8635 | +0.0059 | 49280.0 | 4.3328e-05 | 9.0989e-04 | ✓ | perch_v2__E1__attention |
| perch_v2__E1__attention | perch_v2__E1__hybrid_att_sm_tax | 0.8695 | 0.8697 | -0.0002 | 15558.5 | 3.3610e-01 | 9.7114e-01 | ✗ | perch_v2__E1__hybrid_att_sm_tax |
| perch_v2__E1__attention | perch_v2__E1__hybrid_att_rrf_tax | 0.8695 | 0.8623 | +0.0072 | 24361.0 | 8.1054e-19 | 3.0801e-17 | ✓ | perch_v2__E1__attention |
| perch_v2__E2__attention | perch_v2__E2__softmax | 0.8494 | 0.8436 | +0.0058 | 87607.0 | 1.3864e-04 | 2.4956e-03 | ✓ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__rrf | 0.8494 | 0.8227 | +0.0267 | 140554.5 | 2.8148e-39 | 1.3511e-37 | ✓ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__borda | 0.8494 | 0.8227 | +0.0267 | 140554.5 | 2.8148e-39 | 1.3511e-37 | ✓ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__topk_mean | 0.8494 | 0.8173 | +0.0321 | 120364.5 | 1.2746e-59 | 6.7555e-58 | ✓ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__taxonomy_boost | 0.8494 | 0.8200 | +0.0295 | 146084.0 | 7.6333e-41 | 3.8167e-39 | ✓ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__hybrid_att_sm_tax | 0.8494 | 0.8467 | +0.0027 | 60891.5 | 8.5637e-02 | 5.1382e-01 | ✗ | perch_v2__E2__attention |
| perch_v2__E2__attention | perch_v2__E2__hybrid_att_rrf_tax | 0.8494 | 0.8476 | +0.0019 | 31798.5 | 1.3954e-04 | 2.4956e-03 | ✓ | perch_v2__E2__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__softmax | 0.8932 | 0.8913 | +0.0019 | 3937.5 | 1.0221e-04 | 1.9420e-03 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__rrf | 0.8932 | 0.8311 | +0.0621 | 97389.5 | 9.8300e-109 | 6.3895e-107 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__borda | 0.8932 | 0.8284 | +0.0648 | 99751.5 | 1.8692e-114 | 1.2337e-112 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__topk_mean | 0.8932 | 0.7711 | +0.1221 | 135635.0 | 9.5753e-202 | 6.5112e-200 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__taxonomy_boost | 0.8932 | 0.8890 | +0.0042 | 7472.5 | 7.3706e-10 | 1.7690e-08 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__hybrid_att_sm_tax | 0.8932 | 0.8913 | +0.0019 | 1824.0 | 3.6082e-05 | 7.9379e-04 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E1__attention | birdnet_v3__E1__hybrid_att_rrf_tax | 0.8932 | 0.8880 | +0.0052 | 25771.5 | 4.4010e-13 | 1.4083e-11 | ✓ | birdnet_v3__E1__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__softmax | 0.8373 | 0.8298 | +0.0075 | 7589.5 | 9.6582e-26 | 4.3462e-24 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__rrf | 0.8373 | 0.8108 | +0.0265 | 371606.0 | 2.5455e-21 | 1.0691e-19 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__borda | 0.8373 | 0.8108 | +0.0265 | 371606.0 | 2.5455e-21 | 1.0691e-19 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__topk_mean | 0.8373 | 0.8076 | +0.0297 | 276863.0 | 4.0202e-34 | 1.8493e-32 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__taxonomy_boost | 0.8373 | 0.8240 | +0.0134 | 12107.0 | 2.6298e-40 | 1.2886e-38 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__hybrid_att_sm_tax | 0.8373 | 0.8320 | +0.0054 | 4763.0 | 6.1121e-20 | 2.4448e-18 | ✓ | birdnet_v3__E2__attention |
| birdnet_v3__E2__attention | birdnet_v3__E2__hybrid_att_rrf_tax | 0.8373 | 0.8404 | -0.0031 | 81968.0 | 2.0693e-03 | 2.7626e-02 | ✓ | birdnet_v3__E2__hybrid_att_rrf_tax |

*25/28 comparações significativas (p_adj < 0.05)*

## 4. Score-based vs Rank-fusion
| A | B | MAP(A) | MAP(B) | ΔMAP | W-stat | p_raw | p_adj | sig | winner |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| perch_v2__E1__softmax | perch_v2__E1__rrf | 0.8691 | 0.8187 | +0.0505 | 112175.5 | 5.5770e-82 | 3.1231e-80 | ✓ | perch_v2__E1__softmax |
| perch_v2__E1__softmax | perch_v2__E1__borda | 0.8691 | 0.8180 | +0.0512 | 115970.5 | 8.7472e-82 | 4.8110e-80 | ✓ | perch_v2__E1__softmax |
| perch_v2__E2__softmax | perch_v2__E2__rrf | 0.8436 | 0.8227 | +0.0209 | 294123.5 | 5.2306e-16 | 1.9353e-14 | ✓ | perch_v2__E2__softmax |
| perch_v2__E2__softmax | perch_v2__E2__borda | 0.8436 | 0.8227 | +0.0209 | 294123.5 | 5.2306e-16 | 1.9353e-14 | ✓ | perch_v2__E2__softmax |
| birdnet_v3__E1__softmax | birdnet_v3__E1__rrf | 0.8913 | 0.8311 | +0.0602 | 118089.0 | 1.2533e-99 | 7.7702e-98 | ✓ | birdnet_v3__E1__softmax |
| birdnet_v3__E1__softmax | birdnet_v3__E1__borda | 0.8913 | 0.8284 | +0.0629 | 119353.5 | 1.7988e-105 | 1.1333e-103 | ✓ | birdnet_v3__E1__softmax |
| birdnet_v3__E2__softmax | birdnet_v3__E2__rrf | 0.8298 | 0.8108 | +0.0190 | 472208.5 | 1.1872e-10 | 3.2054e-09 | ✓ | birdnet_v3__E2__softmax |
| birdnet_v3__E2__softmax | birdnet_v3__E2__borda | 0.8298 | 0.8108 | +0.0190 | 472208.5 | 1.1872e-10 | 3.2054e-09 | ✓ | birdnet_v3__E2__softmax |

*8/8 comparações significativas (p_adj < 0.05)*

## Sumário de MAPs por configuração

| Backbone | Strategy | Reranker | MAP |
|---|---|---|---:|
| birdnet_v3 | E1 | attention | 0.8932 |
| birdnet_v3 | E1 | softmax | 0.8913 |
| birdnet_v3 | E1 | hybrid_att_sm_tax | 0.8913 |
| birdnet_v3 | E1 | taxonomy_boost | 0.8890 |
| birdnet_v3 | E1 | hybrid_att_rrf_tax | 0.8880 |
| perch_v2 | E1 | hybrid_att_sm_tax | 0.8697 |
| perch_v2 | E1 | attention | 0.8695 |
| perch_v2 | E1 | softmax | 0.8691 |
| perch_v2 | E1 | taxonomy_boost | 0.8635 |
| perch_v2 | E1 | hybrid_att_rrf_tax | 0.8623 |
| perch_v2 | E2 | attention | 0.8494 |
| perch_v2 | E2 | hybrid_att_rrf_tax | 0.8476 |
| perch_v2 | E2 | hybrid_att_sm_tax | 0.8467 |
| perch_v2 | E2 | softmax | 0.8436 |
| birdnet_v3 | E2 | hybrid_att_rrf_tax | 0.8404 |
| birdnet_v3 | E2 | attention | 0.8373 |
| birdnet_v3 | E2 | hybrid_att_sm_tax | 0.8320 |
| birdnet_v3 | E1 | rrf | 0.8311 |
| birdnet_v3 | E2 | softmax | 0.8298 |
| birdnet_v3 | E1 | borda | 0.8284 |
| birdnet_v3 | E2 | taxonomy_boost | 0.8240 |
| perch_v2 | E2 | rrf | 0.8227 |
| perch_v2 | E2 | borda | 0.8227 |
| perch_v2 | E2 | taxonomy_boost | 0.8200 |
| perch_v2 | E1 | rrf | 0.8187 |
| perch_v2 | E1 | borda | 0.8180 |
| perch_v2 | E2 | topk_mean | 0.8173 |
| birdnet_v3 | E2 | rrf | 0.8108 |
| birdnet_v3 | E2 | borda | 0.8108 |
| birdnet_v3 | E2 | topk_mean | 0.8076 |
| birdnet_v3 | E1 | topk_mean | 0.7711 |
| perch_v2 | E1 | topk_mean | 0.7628 |
