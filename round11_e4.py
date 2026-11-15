"""Round 11 E4: knowledge stored from synthetic bios (round11_bios.py). GPU, minutes. Rules pinned in round11_analyze.py.

Per person i and attribute a: sequence = [EOT] + GPT-2(prompt) + GPT-2(" " + value), prompt = World.probe (template variant 0,
full name as subject, cut before the value). NLL_bits = sum over the value tokens of -log2 p (the training head's per-token
path, round11_eval._nll_rows). bits(i, a) = max(0, log2|A| - NLL_bits). Exact match = every value token is the argmax of the
full next-token distribution (round11_eval._full_logprobs). All N people (timed; the 10-min rule is stated in the output).

python round11_e4.py [labels]          -> runs/round11/eval_bios.jsonl
labels: bioB_20000, bioH_20000, bioB_100000, bioH_100000, bioH_100000_eqF
"""
import json
import math
import os
import sys
import time

import numpy as np

import round11_eval as E
from round11_bios import ATTRS, SIZES, World

E.MODELS.update({
    "bioB_20000": ("bioB_20000", "wsd_bioB_20000_101M_s0"),
    "bioH_20000": ("bioH_20000", "wsd_bioH_20000_101M_s0"),
    "bioB_100000": ("bioB_100000", "wsd_bioB_100000_505M_s0"),
    "bioH_100000": ("bioH_100000", "wsd_bioH_100000_505M_s0"),
    "bioH_100000_eqF": ("bioH_100000", "wsd_bioH_100000_1200M_s0"),
})
EXPOSURES = {"bioB_20000": (1542, 20000), "bioH_20000": (1542, 20000), "bioB_100000": (7710, 100000),
             "bioH_100000": (7710, 100000), "bioH_100000_eqF": (18303, 100000)}
TPB = 55.6362


def probes(N):
    W = World(N)
    out = []
    for i in range(N):
        for a in ATTRS:
            out.append(W.probe(i, a))
    return out


def evaluate(lab, bs=1024, n_limit=None, write=True):
    m = E.Model(lab)
    N = int(E.MODELS[lab][0].split("_")[1])
    N = min(N, n_limit or N)
    t0 = time.time()
    pr = probes(N)
    ctx = m.encode([p for p, _ in pr])
    val = m.encode([v for _, v in pr])
    seqs = [[m.eot] + c + v for c, v in zip(ctx, val)]
    nbits = np.zeros(len(seqs))
    exact = np.zeros(len(seqs), bool)
    for s in range(0, len(seqs), bs):
        part = seqs[s:s + bs]
        X, L = E.pad_seqs(part, 32)
        X = np.concatenate([X, np.zeros((len(X), 1), np.int64)], 1)
        nll = m.nll(X)                                             # [b, L]: nll[j, t] predicts X[j, t + 1]
        pos, owner = [], []
        for j, sq in enumerate(part):
            nv = len(val[s + j])
            st = len(sq) - nv
            nbits[s + j] = nll[j, st - 1: st - 1 + nv].sum() / math.log(2)
            pos += [[j, st - 1 + k] for k in range(nv)]
            owner += [j] * nv
        pos = np.array(pos)
        tgt = X[pos[:, 0], pos[:, 1] + 1]
        am, _ = m.argmax_lp(X[:, :-1], pos, tgt, P=2048)
        ok = np.ones(len(part), bool)
        np.logical_and.at(ok, np.array(owner), am == tgt)
        exact[s:s + len(part)] = ok
    secs = time.time() - t0
    nb = nbits.reshape(N, 5)
    cap = np.array([math.log2(SIZES[a]) for a in ATTRS])
    bits = np.maximum(0.0, cap[None, :] - nb)
    import jax
    nparam = sum(x.size for x in jax.tree.leaves(m.params))
    nemb = m.params["emb"].size
    steps, _ = EXPOSURES[lab]
    rec = {"label": lab, "N": N, "ckpt": E.MODELS[lab][1], "eval_s": secs, "over_10_min": secs > 600,
           "exposures": (steps + round(0.1 * steps)) * 64 * 1025 / (EXPOSURES[lab][1] * TPB),
           "bits": float(bits.sum()), "bits_capacity": float(cap.sum() * N), "bits_per_param": float(bits.sum() / nparam),
           "bits_per_nonemb_param": float(bits.sum() / (nparam - nemb)), "params": int(nparam), "nonemb_params": int(nparam - nemb),
           "per_attr": {a: {"bits": float(bits[:, k].sum()), "exact": float(exact.reshape(N, 5)[:, k].mean()),
                            "mean_nll_bits": float(nb[:, k].mean()), "capacity_bits": float(cap[k] * N)} for k, a in enumerate(ATTRS)},
           "exact": float(exact.mean())}
    # share of attribute values whose first GPT-2 token has training rank >= 4,096 (over people x attributes and per attribute)
    first = np.array([v[0] for v in val]).reshape(N, 5)
    rare = E.RANK[first] >= 4096
    rec["rare_first_token_share"] = float(rare.mean())
    rec["rare_first_token_share_per_attr"] = {a: float(rare[:, k].mean()) for k, a in enumerate(ATTRS)}
    rec["exact_rare_vs_common"] = [float(exact.reshape(N, 5)[rare].mean()), float(exact.reshape(N, 5)[~rare].mean())]
    if write:
        E.jl_append(os.path.join(E.OUT, "eval_bios.jsonl"), rec)
    print(f"[bios] {lab:16s} N {N} bits {rec['bits']:.0f} / {rec['bits_capacity']:.0f}  exact {100 * rec['exact']:.2f}%  "
          f"bits/param {rec['bits_per_param']:.4f}  {secs:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    labs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(EXPOSURES)
    for lab in labs:
        if E.have(lab):
            evaluate(lab)
        else:
            print(f"[bios] {lab}: no checkpoint {E.MODELS[lab][1]}", flush=True)
