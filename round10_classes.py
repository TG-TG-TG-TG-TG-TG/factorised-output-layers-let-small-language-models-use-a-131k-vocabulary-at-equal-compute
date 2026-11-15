"""Round 10 (2026-11-07): adaptive splits for the Part-A tail fixes, built exactly as round9_classes.py.

frequent = the top K tokens by training-set target count (runs/round9/train_counts.npy, round9_freq.py),
           ties broken by token id (stable sort)
tail     = the other 50,257 - K tokens, in C' clusters by heads.balanced_kmeans on B_r8's trained U rows
           of those tokens (capacity ceil(n_tail / C'))
K = 4096 reuses round 9's frequent list unchanged (asserted identical).

python round10_classes.py 8192:1024 4096:4096 8192:4096
   -> runs/round10/adapt_freq<K>.npy, adapt_tail<K>_<C>.npy, adapt_report_<K>_<C>.json
"""
import json
import os
import sys
import time

import numpy as np

from arms import ARMS
from round8_heads import V_REAL, H, load_params

OUT, R9 = "runs/round10", "runs/round9"
counts = np.load(os.path.join(R9, "train_counts.npy"))[:V_REAL]
order = np.argsort(-counts, kind="stable")
p = load_params(ARMS["tB"], "wsd_B_r8_s0")
U = np.asarray(p["emb"])[:V_REAL]
assert np.array_equal(order[:4096], np.load(os.path.join(R9, "adapt_freq4096.npy"))), "round-9 frequent list differs"
for spec in sys.argv[1:]:
    K, C = map(int, spec.split(":"))
    freq = order[:K].astype(np.int32)
    tail_ids = np.sort(order[K:])
    t = time.time()
    cl = H.balanced_kmeans(U[tail_ids], C).astype(np.int32)
    dt = time.time() - t
    cls = np.full(V_REAL, -1, np.int32)
    cls[tail_ids] = cl
    sizes = np.bincount(cl, minlength=C)
    assert len(np.intersect1d(freq, tail_ids)) == 0 and len(freq) + len(tail_ids) == V_REAL
    np.save(os.path.join(OUT, f"adapt_freq{K}.npy"), freq)
    np.save(os.path.join(OUT, f"adapt_tail{K}_{C}.npy"), cls)
    tot = counts.sum()
    rep = {"K": K, "C": C, "n_tail": int(len(tail_ids)), "capacity": int(-(-len(tail_ids) // C)), "kmeans_s": dt,
           "sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                     "n_full": int((sizes == sizes.max()).sum()), "n_empty": int((sizes == 0).sum())},
           "frequent_share_of_train_targets": float(counts[freq].sum() / tot),
           "count_at_rank_K": int(counts[order[K - 1]]), "disjoint_and_cover": True}
    json.dump(rep, open(os.path.join(OUT, f"adapt_report_{K}_{C}.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)
