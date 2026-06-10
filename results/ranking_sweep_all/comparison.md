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
