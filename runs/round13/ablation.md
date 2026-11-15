# Design-choice ablation (d=128, equal tokens, rounds 8-11 runs)

BPB at the 1,200M-stable-token branch (600M in brackets). Δ = this row minus the row in 'vs' (two-seed means where two seeds exist). Equal tokens, not equal FLOPs: the flat row costs ~2.4-4x the two-level rows' FLOPs.

| row | change | vs | seeds | BPB 1200M (600M) | Δ vs 1200M (600M) | FLOPs 1200M |
|---|---|---|---|---|---|---|
| flat | flat tied softmax over all 50,257 tokens | — | 2 | 1.2835 (1.3022) | — | 6.66e+16 |
| random | two-level, 1,024 random clusters, no frequent list, free cluster vectors | — | 1 | 1.3694 (1.3911) | — | 1.67e+16 |
| kmeans | clusters by balanced k-means on a trained embedding | random | 1 | 1.3122 (1.3328) | -0.0572 (-0.0583) | 1.67e+16 |
| kmeans+tied | cluster logit tied to the members' mean embedding | kmeans | 1 | 1.3125 (1.3307) | +0.0004 (-0.0021) | 1.67e+16 |
| kmeans+freq4k | 4,096 most frequent tokens as top-level singletons | kmeans | 1 | 1.3015 (1.3205) | -0.0107 (-0.0123) | 2.08e+16 |
| freq4k+tied | both: frequent singletons and tied-mean cluster logits | kmeans+freq4k | 1 | 1.3000 (1.3180) | -0.0014 (-0.0025) | 2.08e+16 |
| freq8k+tied | 8,192 frequent singletons | freq4k+tied | 2 | 1.2957 (1.3144) | -0.0043 (-0.0036) | 2.50e+16 |
| freq4k+tied+4kcl | 4,096 tail clusters instead of 1,024 | freq4k+tied | 1 | 1.2999 (1.3185) | -0.0001 (+0.0005) | 2.39e+16 |
| final | 8,192 singletons and 4,096 tail clusters (the head used at d=384) | freq8k+tied | 2 | 1.2938 (1.3130) | -0.0018 (-0.0013) | 2.80e+16 |
| final+rescore16 | exact re-scoring of the 16 top clusters | final | 2 | 1.2923 (1.3109) | -0.0015 (-0.0022) | 2.82e+16 |

Seed spread where two seeds exist: 0.0002-0.0009 BPB, so a one-seed difference under ~0.002 is not distinguishable.

Runs: flat = wsd_B_r8_s0, wsd_B_s1; random = wsd_H2r_s0; kmeans = wsd_H2_s0; kmeans+tied = wsd_H2tm_s0; kmeans+freq4k = wsd_H2a_s0; freq4k+tied = wsd_H2atm_s0; freq8k+tied = wsd_H3k8_s0, wsd_H3k8_s1; freq4k+tied+4kcl = wsd_H3c4_s0; final = wsd_H3k8c4_s0, wsd_H3k8c4_s1; final+rescore16 = wsd_H3rs16_s0, wsd_H3rs16_s1
