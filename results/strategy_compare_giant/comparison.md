| Metric | strategy1_segments_torch | strategy2_super_embedding_torch | strategy1_segments_no_overlap_torch | strategy1_segments_no_overlap_noise_torch | strategy2_super_embedding_no_overlap_torch | strategy2_super_embedding_no_overlap_noise_torch |
|---|---|---|---|---|---|---|
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
