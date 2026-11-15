"""Round 8, step 2: C = 1024 classes from B_r8's trained tied embedding (a bootstrap: the classes
reuse a trained model's embeddings).

  k-means:  heads.balanced_kmeans(U_B[:50257], 1024)          (rows L2-normalized inside, capacity 50)
  random:   same size histogram, token ids shuffled with np.random.default_rng(0)

python round8_classes.py [--ckpt wsd_B_r8_s0] [--out runs/round8]
  -> classes_kmeans1024.npy, classes_random1024.npy, classes_report.json / .txt
"""
import argparse
import json
import os
import time

import numpy as np

from arms import ARMS
from round8_heads import V_REAL, H, load_params

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="wsd_B_r8_s0")
ap.add_argument("--out", default="runs/round8")
ap.add_argument("--C", type=int, default=1024)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

p = load_params(ARMS["tB"], a.ckpt)
U = np.asarray(p["emb"])[:V_REAL]
t = time.time()
km = H.balanced_kmeans(U, a.C).astype(np.int32)
t_km = time.time() - t
sizes = np.bincount(km, minlength=a.C)
perm = np.random.default_rng(0).permutation(V_REAL)
rnd = np.empty(V_REAL, np.int32)
rnd[perm] = np.repeat(np.arange(a.C, dtype=np.int32), sizes)  # same histogram, shuffled ids
np.save(os.path.join(a.out, f"classes_kmeans{a.C}.npy"), km)
np.save(os.path.join(a.out, f"classes_random{a.C}.npy"), rnd)

import tiktoken  # noqa: E402
enc = tiktoken.get_encoding("gpt2")
rs = np.bincount(rnd, minlength=a.C)
rep = {"ckpt": a.ckpt, "C": a.C, "V": V_REAL, "capacity": -(-V_REAL // a.C), "kmeans_s": t_km,
       "kmeans_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                        "n_full": int((sizes == sizes.max()).sum())},
       "random_sizes": {"min": int(rs.min()), "median": float(np.median(rs)), "max": int(rs.max())},
       "same_histogram": bool(np.array_equal(np.sort(sizes), np.sort(rs)))}
pick = np.random.default_rng(1).choice(a.C, 8, replace=False)
lines = []
for name, cl in (("kmeans", km), ("random", rnd)):
    for c in pick:
        toks = np.flatnonzero(cl == c)
        lines.append(f"[{name} class {c}, {len(toks)} tokens] " + " | ".join(repr(enc.decode([int(w)])) for w in toks[:50]))
rep["examples_classes"] = [int(c) for c in pick]
json.dump(rep, open(os.path.join(a.out, "classes_report.json"), "w"), indent=1)
open(os.path.join(a.out, "classes_examples.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(json.dumps(rep, indent=1))
print("\n".join(lines))
