"""Round 9: training-set token counts and the unique-token budget check for Part D.

Counts = target occurrences (row[:, 1:]) in exactly the rows the 1200M WSD runs consume:
the first 20,142 batches x 64 rows of data.token_rows(TRAIN_FILES, 1024, seed=0), i.e. the stable
phase (18,311 steps) plus the final 1,831-step decay; earlier branches replay their batches.
(data.Batches cuts batches from this same row stream in order, so these are the same rows.)

Unique-token check: row counts of the train shards from their parquet footers (HfFileSystem reads only
the footer), times the mean tokens per document of this count.

python round9_freq.py [--batches 20142]  -> runs/round9/train_counts.npy, freq_report.json
"""
import argparse
import json
import os
import time

import numpy as np

from data import REPO, TRAIN_FILES, token_rows

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=20142)
a = ap.parse_args()
OUT = "runs/round9"
os.makedirs(OUT, exist_ok=True)

need_rows = a.batches * 64
counts = np.zeros(50304, np.int64)
rows_seen, docs, t0 = 0, 0, time.time()
for rows, docs in token_rows(TRAIN_FILES, 1024, seed=0, log=print):
    take = rows[: need_rows - rows_seen]
    counts += np.bincount(take[:, 1:].ravel(), minlength=50304)
    rows_seen += len(take)
    if rows_seen and rows_seen % (64 * 2000) < len(rows):
        print(f"{rows_seen / 64:.0f} batches, {time.time() - t0:.0f} s", flush=True)
    if rows_seen >= need_rows:
        break
dt = time.time() - t0
np.save(os.path.join(OUT, "train_counts.npy"), counts)
tok_streamed = rows_seen * 1025
tok_per_doc = tok_streamed / docs

shard_rows = {}
try:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    for f in TRAIN_FILES:
        with fs.open(f"datasets/{REPO}/{f}", "rb") as fh:
            shard_rows[f] = pq.ParquetFile(fh).metadata.num_rows
except Exception as e:  # report, don't guess
    shard_rows = {"error": f"{type(e).__name__}: {e}"}
n_docs = sum(shard_rows.values()) if "error" not in shard_rows else None
rep = {"batches": a.batches, "rows": rows_seen, "targets_counted": int(counts.sum()), "docs_consumed": int(docs),
       "tokens_per_doc_incl_eot": tok_per_doc, "count_s": dt,
       "nonzero_vocab": int((counts[:50257] > 0).sum()), "pad_rows_counted": int(counts[50257:].sum()),
       "train_shard_docs": shard_rows, "train_docs_total": n_docs,
       "est_unique_train_tokens": n_docs * tok_per_doc if n_docs else None,
       "partD_tokens_needed": (4800e6 * 1.1) // 65536 * 65536}
top = np.sort(counts[:50257])[::-1]
cum = np.cumsum(top) / top.sum()
rep["top_share"] = {str(k): float(cum[k - 1]) for k in (256, 1024, 4096, 16384)}
json.dump(rep, open(os.path.join(OUT, "freq_report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
