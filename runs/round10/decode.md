# Round 10 decode: batch 1, random bf16 weights, round-8 tiny protocol

K = 8192 singletons + C = 4096 tail clusters (H3*'s shape). ms/token = median of 5 repeats per process; mean (min-max) over 3 processes.

| arm | applications x d | head | ms/token | min-max | vs B (flat_sample) |
|---|---|---|---|---|---|
| B | 6 x 128 | flat_sample | 0.1542 | 0.1371-0.1701 | 1.00x |
| B_greedy | 6 x 128 | flat_greedy | 0.1549 | 0.1353-0.1668 | 1.01x |
| H3s | 6 x 128 | hier9_sample | 0.1578 | 0.1362-0.1735 | 1.02x |
| Rloop | 24 x 128 (6 weight sets) | hier9_sample | 0.5522 | 0.4964-0.5889 | 3.58x |
| Rdeep | 24 x 128 | hier9_sample | 0.5342 | 0.4759-0.6128 | 3.47x |
| Rwide | 6 x 256 | hier9_sample | 0.1619 | 0.1552-0.1674 | 1.05x |
| D1 | 6 x 128, + 2-gram | hier9_sample | 0.1546 | 0.1402-0.1724 | 1.00x |
| D2 | 6 x 128, + 3-gram | hier9_sample | 0.1608 | 0.1383-0.1744 | 1.04x |
