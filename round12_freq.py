"""Round 12: bpe131072 training-set target counts for H128k-384's frequent list (round9_freq.py's definition):
target occurrences (row[:, 1:]) in exactly the rows the H128k run consumes = the first B batches x 64 rows of
data.token_rows(TRAIN_FILES, 1024, seed=0, source="bpe131072"), B = stable steps + final decay steps (plan.json).

python round12_freq.py  -> runs/round12/v131072_train_counts.npy, v131072_freq_report.json
"""
import json
import os
import time

import numpy as np

from data import TRAIN_FILES, token_rows

OUT = "runs/round12"
V = 131072
plan = json.load(open(os.path.join(OUT, "plan.json")))["H128k384"]["match_575"]
S = plan["stable_steps"]
B = S + round(0.1 * S)
need_rows = B * 64
counts = np.zeros(V, np.int64)
rows_seen, docs, t0 = 0, 0, time.time()
for rows, docs in token_rows(TRAIN_FILES, 1024, seed=0, source=f"bpe{V}", log=print):
    take = rows[: need_rows - rows_seen]
    counts += np.bincount(take[:, 1:].ravel().astype(np.int64), minlength=V)
    rows_seen += len(take)
    if rows_seen % (64 * 1000) < len(rows):
        print(f"{rows_seen / 64:.0f} / {B} batches, {time.time() - t0:.0f} s", flush=True)
    if rows_seen >= need_rows:
        break
np.save(os.path.join(OUT, f"v{V}_train_counts.npy"), counts)
top = np.sort(counts)[::-1]
cum = np.cumsum(top) / top.sum()
rep = {"batches": B, "stable_steps": S, "rows": rows_seen, "targets_counted": int(counts.sum()), "docs_consumed": int(docs),
       "count_s": time.time() - t0, "nonzero_vocab": int((counts > 0).sum()),
       "top_share": {str(k): float(cum[k - 1]) for k in (1024, 4096, 8192, 16384, 65536)}}
json.dump(rep, open(os.path.join(OUT, f"v{V}_freq_report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
