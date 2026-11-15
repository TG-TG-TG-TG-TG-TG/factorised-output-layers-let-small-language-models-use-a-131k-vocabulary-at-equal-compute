# Round 12 decode: random bf16 weights, round-8 protocol, batch 1 and 32

ms/step = median of 5 repeats per process (mean (min-max) over processes); ms/token = ms/step / batch; ms/byte = ms/token / val bytes per token.

| arm | head | vocab | batch | ms/step | min-max | ms/token | ms/byte | vs B384 (same batch) |
|---|---|---|---|---|---|---|---|---|
| B384 | flat_sample | 50,257 | 1 | 0.2789 | 0.2730-0.2830 | 0.2789 | 0.06021 | 1.00x |
| H3s384 | hier9_sample | 50,257 | 1 | 0.2029 | 0.1982-0.2073 | 0.2029 | 0.04380 | 0.73x |
| F8k384 | flat_sample | 8,192 | 1 | 0.1975 | 0.1934-0.2024 | 0.1975 | 0.05021 | 0.71x |
| F128k384 | flat_sample | 131,072 | 1 | 0.3664 | 0.3642-0.3696 | 0.3664 | 0.07411 | 1.31x |
| H128k384 | hier9_sample | 131,072 | 1 | 0.2013 | 0.1949-0.2055 | 0.2013 | 0.04071 | 0.72x |
| B384 | flat_sample | 50,257 | 32 | 1.9815 | 1.9672-1.9953 | 0.0619 | 0.01337 | 1.00x |
| H3s384 | hier9_sample | 50,257 | 32 | 1.9380 | 1.9232-1.9560 | 0.0606 | 0.01307 | 0.98x |
| F8k384 | flat_sample | 8,192 | 32 | 1.9209 | 1.9175-1.9276 | 0.0600 | 0.01526 | 0.97x |
| F128k384 | flat_sample | 131,072 | 32 | 2.1003 | 2.0914-2.1056 | 0.0656 | 0.01328 | 1.06x |
| H128k384 | hier9_sample | 131,072 | 32 | 1.9210 | 1.9134-1.9357 | 0.0600 | 0.01214 | 0.97x |
