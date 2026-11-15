# Round 8 decode benchmark (batch 1, random bf16 weights)

KV-cache decode of 256 tokens from position 512; ms/token = median of 5 interleaved repeats per process; mean over 3 processes. head ms = variant - backbone-only ('none'). `python round8_decode.py run <proc>` (proc 0, 1, 2), then `python round8_decode.py md`.

## tiny: 6 x d128, 2/2 heads, gelu 512, V 50304, C 1024

| head | proc 0 | proc 1 | proc 2 | mean ms/tok | head ms/tok | vs flat_greedy (whole) | fallback-3 / unresolved (of 256) |
|---|---|---|---|---|---|---|---|
| none | 0.1398 | 0.1369 | 0.1323 | 0.1363 | +0.0000 | -4.8% |  |
| flat_greedy | 0.1458 | 0.1440 | 0.1399 | 0.1432 | +0.0069 | +0.0% |  |
| flat_sample | 0.1488 | 0.1469 | 0.1426 | 0.1461 | +0.0098 | +2.0% |  |
| hier_sample | 0.1498 | 0.1454 | 0.1479 | 0.1477 | +0.0113 | +3.1% |  |
| hier_greedy | 0.2836 | 0.2579 | 0.2350 | 0.2588 | +0.1225 | +80.7% | 256 / 256 |

## big: 28 x d1280, 20/4 heads, swiglu 3456, V 65536, C 1024

| head | proc 0 | proc 1 | proc 2 | mean ms/tok | head ms/tok | vs flat_greedy (whole) | fallback-3 / unresolved (of 256) |
|---|---|---|---|---|---|---|---|
| none | 2.2970 | 2.2025 | 2.2368 | 2.2454 | +0.0000 | -10.0% |  |
| flat_greedy | 2.5568 | 2.4369 | 2.4927 | 2.4955 | +0.2500 | +0.0% |  |
| flat_sample | 2.5076 | 2.4377 | 2.4801 | 2.4751 | +0.2297 | -0.8% |  |
| hier_sample | 2.3317 | 2.2189 | 2.2526 | 2.2677 | +0.0223 | -9.1% |  |
| hier_greedy | 2.5763 | 2.3587 | 2.3938 | 2.4429 | +0.1975 | -2.1% | 256 / 256 |

GPU (util, mem, SM clock) before/after: proc 0: 4 %, 3055 MiB, 2602 MHz / 95 %, 7118 MiB, 2872 MHz; proc 1: 3 %, 3141 MiB, 2872 MHz / 96 %, 6680 MiB, 2872 MHz; proc 2: 2 %, 2955 MiB, 2872 MHz / 95 %, 6651 MiB, 2872 MHz

Trained H2 (val contexts): {"ckpt": "wsd_H2_s0", "positions": 4096, "fallback3_rate": 0.171630859375, "unresolved_rate": 0.115966796875, "jit_greedy_equals_exact_argmax": 0.9736328125, "jit_greedy_equals_exact_when_resolved": 1.0, "host_greedy_classes_expanded_mean_256": 1.81640625, "host_greedy_classes_expanded_max_256": 21}
