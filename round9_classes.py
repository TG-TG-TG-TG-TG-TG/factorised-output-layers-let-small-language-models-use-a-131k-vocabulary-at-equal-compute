"""Round 9: the adaptive split for tH2a / tH2atm (and the frequency buckets of Part A).

frequent = the top K = 4096 of the 50,257 real tokens by training-set target count (runs/round9/train_counts.npy,
           round9_freq.py: the 1,320M-token stream the 1200M runs see), ties broken by token id (stable sort)
tail     = the other 46,161 tokens, in C' = 1024 clusters by heads.balanced_kmeans on B_r8's trained U rows
           of those tokens (capacity ceil(46161/1024) = 46). Like round 8's classes, a bootstrap.
rank     = 0-based rank of every token by count (the A1 buckets)

python round9_classes.py  -> runs/round9/adapt_freq4096.npy, adapt_tail4096_1024.npy, token_rank.npy, adapt_report.json
"""
import json
import os
import time

import numpy as np

from arms import ARMS
from round8_heads import V_REAL, H, load_params

K, C, OUT = 4096, 1024, "runs/round9"
counts = np.load(os.path.join(OUT, "train_counts.npy"))[:V_REAL]
order = np.argsort(-counts, kind="stable")
rank = np.empty(V_REAL, np.int64)
rank[order] = np.arange(V_REAL)
freq = order[:K].astype(np.int32)
tail_ids = np.sort(order[K:])
p = load_params(ARMS["tB"], "wsd_B_r8_s0")
U = np.asarray(p["emb"])[:V_REAL]
t = time.time()
cl = H.balanced_kmeans(U[tail_ids], C).astype(np.int32)
dt = time.time() - t
cls = np.full(V_REAL, -1, np.int32)
cls[tail_ids] = cl
sizes = np.bincount(cl, minlength=C)
assert len(np.intersect1d(freq, tail_ids)) == 0 and len(freq) + len(tail_ids) == V_REAL
np.save(os.path.join(OUT, f"adapt_freq{K}.npy"), freq)
np.save(os.path.join(OUT, f"adapt_tail{K}_{C}.npy"), cls)
np.save(os.path.join(OUT, "token_rank.npy"), rank)
tot = counts.sum()
rep = {"K": K, "C": C, "n_tail": int(len(tail_ids)), "capacity": int(-(-len(tail_ids) // C)), "kmeans_s": dt,
       "sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                 "n_full": int((sizes == sizes.max()).sum()), "n_empty": int((sizes == 0).sum()),
                 "histogram": {str(int(s)): int((sizes == s).sum()) for s in np.unique(sizes)}},
       "frequent_share_of_train_targets": float(counts[freq].sum() / tot),
       "count_at_rank_4096": int(counts[order[K - 1]]), "zero_count_tokens": int((counts == 0).sum()),
       "disjoint_and_cover": True}
json.dump(rep, open(os.path.join(OUT, "adapt_report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
