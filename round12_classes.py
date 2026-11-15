"""Round 12: H128k-384's two-level split, built as round10_classes.py.

frequent = the top K = 8,192 ids by target count over the rows the H128k run consumes (v131072_train_counts.npy,
           round12_freq.py), ties broken by id (stable sort)
tail     = the other 122,880 ids in C' = 4,096 clusters by heads.balanced_kmeans on the trained embedding rows of
           the d=128 flat proxy wsd_tF128k_proxy_50M_s0 (arm tF128k = tB on bpe131072, 50M stable + 10% decay)
           (capacity ceil(122,880 / 4,096) = 30)

python round12_classes.py  -> runs/round12/v131072_adapt_freq8192.npy, v131072_adapt_tail8192_4096.npy, v131072_adapt_report.json
"""
import json
import os
import time

import numpy as np

from arms import ARMS
from round8_heads import H, load_params

OUT, V, K, C = "runs/round12", 131072, 8192, 4096
counts = np.load(os.path.join(OUT, f"v{V}_train_counts.npy"))
order = np.argsort(-counts, kind="stable")
p = load_params(ARMS["tF128k"], "wsd_tF128k_proxy_50M_s0")
U = np.asarray(p["emb"], np.float32)
assert U.shape == (V, 128), U.shape
freq = order[:K].astype(np.int32)
tail_ids = np.sort(order[K:])
t = time.time()
cl = H.balanced_kmeans(U[tail_ids], C).astype(np.int32)
dt = time.time() - t
cls = np.full(V, -1, np.int32)
cls[tail_ids] = cl
sizes = np.bincount(cl, minlength=C)
assert len(np.intersect1d(freq, tail_ids)) == 0 and len(freq) + len(tail_ids) == V
assert V - 1 in set(freq.tolist()), "EOT should be frequent"
np.save(os.path.join(OUT, f"v{V}_adapt_freq{K}.npy"), freq)
np.save(os.path.join(OUT, f"v{V}_adapt_tail{K}_{C}.npy"), cls)
tot = counts.sum()
rep = {"V": V, "K": K, "C": C, "n_tail": int(len(tail_ids)), "capacity": int(-(-len(tail_ids) // C)), "kmeans_s": dt,
       "sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                 "n_full": int((sizes == sizes.max()).sum()), "n_empty": int((sizes == 0).sum())},
       "frequent_share_of_train_targets": float(counts[freq].sum() / tot),
       "count_at_rank_K": int(counts[order[K - 1]]), "never_seen_ids": int((counts == 0).sum()), "disjoint_and_cover": True}
json.dump(rep, open(os.path.join(OUT, f"v{V}_adapt_report.json"), "w"), indent=1)
print(json.dumps(rep), flush=True)
