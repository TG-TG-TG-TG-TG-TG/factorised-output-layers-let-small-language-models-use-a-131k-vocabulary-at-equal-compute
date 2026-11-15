"""Round 13, experiment 1 (2026-11-11): the H3*-384 split (K = 8,192 singletons + C' = 4,096 tail clusters) rebuilt
exactly as round10_classes.py, except that the tail clusters come from a cheap proxy whose FLOPs are charged:

frequent = the top K = 8,192 tokens by training-set target count (runs/round9/train_counts.npy), ties by id (stable
           sort) -- asserted identical to runs/round10/adapt_freq8192.npy (counting tokens costs no model FLOPs)
tail     = the other 42,065 tokens in C' = 4,096 clusters by heads.balanced_kmeans on the trained embedding rows of
           wsd_tBproxy_50M_s0 (arm tB = the d=128 flat recipe of B_r8, 50M stable tokens + 10% decay, seed 0), the
           50k twin of round 12's 131k proxy wsd_tF128k_proxy_50M_s0 (capacity ceil(42,065 / 4,096) = 11)

python round13_classes.py  -> runs/round13/adapt_freq8192.npy, adapt_tail8192_4096.npy, adapt_report_8192_4096.json
"""
import json
import os
import time

import numpy as np

from arms import ARMS
from round8_heads import V_REAL, H, load_params

OUT, R9, R10, K, C = "runs/round13", "runs/round9", "runs/round10", 8192, 4096
counts = np.load(os.path.join(R9, "train_counts.npy"))[:V_REAL]
order = np.argsort(-counts, kind="stable")
freq = order[:K].astype(np.int32)
assert np.array_equal(freq, np.load(os.path.join(R10, f"adapt_freq{K}.npy"))), "round-10 frequent list differs"
p = load_params(ARMS["tB"], "wsd_tBproxy_50M_s0")
U = np.asarray(p["emb"], np.float32)[:V_REAL]
tail_ids = np.sort(order[K:])
t = time.time()
cl = H.balanced_kmeans(U[tail_ids], C).astype(np.int32)
dt = time.time() - t
cls = np.full(V_REAL, -1, np.int32)
cls[tail_ids] = cl
sizes = np.bincount(cl, minlength=C)
assert len(np.intersect1d(freq, tail_ids)) == 0 and len(freq) + len(tail_ids) == V_REAL
old = np.load(os.path.join(R10, f"adapt_tail{K}_{C}.npy"))
np.save(os.path.join(OUT, f"adapt_freq{K}.npy"), freq)
np.save(os.path.join(OUT, f"adapt_tail{K}_{C}.npy"), cls)


def pair_agree(a, b, ids, n=200000, seed=0):
    """fraction of random tail-token pairs on which the two clusterings agree (same cluster in both, or in neither)"""
    r = np.random.default_rng(seed)
    i, j = r.choice(ids, n), r.choice(ids, n)
    return float(np.mean((a[i] == a[j]) == (b[i] == b[j]))), float(np.mean(a[i] == a[j]))


agree, same = pair_agree(cls, old, tail_ids)
rep = {"K": K, "C": C, "proxy": "wsd_tBproxy_50M_s0", "n_tail": int(len(tail_ids)), "capacity": int(-(-len(tail_ids) // C)),
       "kmeans_s": dt, "sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                                 "n_full": int((sizes == sizes.max()).sum()), "n_empty": int((sizes == 0).sum())},
       "pair_agreement_vs_round10": agree, "same_cluster_rate_random_pairs": same,
       "identical_to_round10": bool(np.array_equal(cls, old)), "disjoint_and_cover": True}
json.dump(rep, open(os.path.join(OUT, f"adapt_report_{K}_{C}.json"), "w"), indent=1)
print(json.dumps(rep), flush=True)
