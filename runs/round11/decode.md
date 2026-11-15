# Round 11 decode: batch 1, random bf16 weights, round-8 tiny protocol

K = 8192 singletons + C = 4096 tail clusters. ms/token = median of 5 repeats per process; mean (min-max) over 3 processes. ms/byte = ms/token / val bytes per token (GPT-2 4.633, BPE-8k 3.933, BPE-4k 3.473).

| arm | d | head | vocab | ms/token | min-max | ms/byte | vs flat GPT-2 of the same width |
|---|---|---|---|---|---|---|---|
| B | 128 | flat_sample | 50,304 | 0.1434 | 0.1419-0.1446 | 0.0309 | 1.00x |
| H3s | 128 | hier9_sample | 50,304 | 0.1434 | 0.1412-0.1465 | 0.0310 | 1.00x |
| H3rs16 | 128 | hier9_sample + top-16 re-scoring | 50,304 | 0.1779 | 0.1768-0.1792 | 0.0384 | 1.24x |
| H3rs64 | 128 | hier9_sample + top-64 re-scoring | 50,304 | 0.1770 | 0.1756-0.1786 | 0.0382 | 1.23x |
| B384 | 384 | flat_sample | 50,304 | 0.2738 | 0.2700-0.2768 | 0.0591 | 1.00x |
| H3s384 | 384 | hier9_sample | 50,304 | 0.1981 | 0.1918-0.2029 | 0.0428 | 0.72x |
| H3rs16_384 | 384 | hier9_sample + top-16 re-scoring | 50,304 | 0.2327 | 0.2277-0.2367 | 0.0502 | 0.85x |
| F8k384 | 384 | flat_sample | 8,192 | 0.1912 | 0.1862-0.1946 | 0.0486 | 0.70x |
| F4k384 | 384 | flat_sample | 4,096 | 0.1924 | 0.1890-0.1958 | 0.0554 | 0.70x |
| H3s512 | 512 | hier9_sample | 50,304 | 0.2757 | 0.2752-0.2763 | 0.0595 | 1.01x |
