"""Round 8, step 0.3: UTF-8 bytes of the scored val targets, for bits/byte.

The val set is data.val_rows(1024): 4882 rows of 1025 tokens (shard 013, first 5M tokens),
and lm_loss scores row[1:] of each row, so the bytes are summed over exactly those targets.
Bytes per token = len(enc.decode_single_token_bytes(t)); concatenated they are the UTF-8 text.
EOT (50256) counts 0 bytes (it is a separator, not text). BPB = loss * tokens / (bytes * ln 2).

python round8_valbytes.py  ->  runs/round8/valbytes.json
"""
import json
import os

import numpy as np
import tiktoken

from data import EOT, val_rows

v = val_rows(1024)
tgt = v[:, 1:].ravel()
enc = tiktoken.get_encoding("gpt2")
nb = np.zeros(enc.n_vocab, np.int64)
for t in range(enc.n_vocab):
    nb[t] = 0 if t == EOT else len(enc.decode_single_token_bytes(t))
bytes_ = int(nb[tgt].sum())
out = {"rows": int(v.shape[0]), "tokens": int(tgt.size), "bytes": bytes_, "eot_tokens": int((tgt == EOT).sum()),
       "bytes_per_token": bytes_ / tgt.size, "eot_bytes": 0,
       "bpb_per_nat": tgt.size / (bytes_ * np.log(2))}
os.makedirs("runs/round8", exist_ok=True)
json.dump(out, open("runs/round8/valbytes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
