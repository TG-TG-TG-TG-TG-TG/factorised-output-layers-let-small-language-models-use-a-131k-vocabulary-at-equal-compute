# Round 9 Part F: exact jitted greedy (batch 1, random bf16 weights, round-8 protocol)

Head shape: K = 4096 singletons + C = 1024 clusters. ms/token = median of 5 repeats per process, mean over 3 processes. 'expected' = nofb + r·(fb − nofb) with the trained fallback rate r = 0.0054 (H2atm, 4096 val positions; exact match 1.0000).

## tiny

| head | ms/token | vs flat_greedy |
|---|---|---|
| none | 0.1396 | -11.7% |
| flat_greedy | 0.1581 | +0.0% |
| hier_sample | 0.1488 | -5.9% |
| exact_greedy_nofb | 0.1670 | +5.6% |
| exact_greedy_fb | 0.1870 | +18.3% |
| exact_greedy, expected at r = 0.005 | 0.1671 | +5.7% |

## big

| head | ms/token | vs flat_greedy |
|---|---|---|
| none | 2.1658 | -9.1% |
| flat_greedy | 2.3833 | +0.0% |
| hier_sample | 2.1665 | -9.1% |
| exact_greedy_nofb | 2.1984 | -7.8% |
| exact_greedy_fb | 2.4552 | +3.0% |
| exact_greedy, expected at r = 0.005 | 2.1998 | -7.7% |

