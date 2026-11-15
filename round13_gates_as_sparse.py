"""Round 13 experiment 3, 2026-11-12: gates for the sparse band path of the adaptive-softmax head (cfg.head_sparse,
round13_heads._band_sparse), CPU. The sparse path must give the dense path's numbers (the dense path passed
round13_gates_as.py A1-A6 against the independent reference):
  S1 sparse branch (caps above the band counts): per-token NLL = dense (max |diff| < 1e-5) and = reference (< 1e-4)
  S2 overflow branch (caps below the band counts -> lax.cond takes the dense path): NLL = dense (max |diff| < 1e-5)
  S3 gradients through the sparse branch = dense gradients (relative L2 < 1e-5, every head tensor and the embedding)
  S4 a realistic chunk: 4,096 targets drawn from the training unigram distribution, the training caps (band_caps),
     summed head_loss_sum sparse = dense (relative < 1e-6) and gradients (relative < 1e-5); band counts vs caps printed
Both vocabularies.  JAX_PLATFORMS=cpu python round13_gates_as_sparse.py  -> runs/round13/gates_as_sparse.json
2026-11-13 (experiment 4): JAX_PLATFORMS=cpu python round13_gates_as_sparse.py AS128k384c49=AS128k384c49d c49
  (sparse arm=its dense twin, output tag) -> runs/round13/gates_as_sparse_c49.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from model import init_params
import round13_heads as R13

jax.config.update("jax_default_matmul_precision", "highest")
PAIRS = ([tuple(x.split("=")) for x in sys.argv[1].split(",")] if len(sys.argv) > 1
         else [("AS384s", "AS384"), ("AS128k384s", "AS128k384")])
OUT = os.path.join(R13.HERE, "runs", "round13", f"gates_as_sparse{'_' + sys.argv[2] if len(sys.argv) > 2 else ''}.json")
res, ok = {}, True


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def perturbed(cfg, seed=0):
    p = init_params(cfg, jax.random.PRNGKey(seed))
    k = iter(jax.random.split(jax.random.PRNGKey(7), 8))
    hp = dict(p["head"])
    hp["bb"] = jax.random.normal(next(k), hp["bb"].shape) * 2.0
    hp["Vb"] = hp["Vb"] * 3.0
    for b in (1, 2):
        if f"P{b}" in hp:  # full-width arms have no projection
            hp[f"P{b}"] = hp[f"P{b}"] * 2.0
        hp[f"W{b}"] = hp[f"W{b}"] + 0.3 * jax.random.normal(next(k), hp[f"W{b}"].shape)
    return {"emb": p["emb"] * 2.0, "head": hp}


for arm, dense_arm in PAIRS:
    cs = ARMS[arm]
    cd = ARMS[dense_arm]
    assert cs.head_sparse and not cd.head_sparse
    V = R13.vreal(cs)
    t = R13.tabs_np(cs)
    pp = perturbed(cs)
    c0, c1 = R13.cuts(cs)
    ranks = np.concatenate([np.arange(0, 20), np.arange(c0, c0 + 20), np.arange(c1, c1 + 20), [V - 1, V - 2, c0 - 1, c1 - 1]])
    n = len(ranks)
    h = jax.random.normal(jax.random.PRNGKey(3), (n, cs.d))
    y = jnp.asarray(t["order"][ranks])
    nb = [int(np.sum(t["band"][np.asarray(y)] == b)) for b in (1, 2)]
    dense = R13.nll_tokens(h, y, pp, cd)
    ref = -jnp.take_along_axis(R13.ref_logprobs_full(pp, h, cs), y[:, None], 1)[:, 0]
    sp = R13.nll_tokens(h, y, pp, cs, caps=(32, 32))
    ov = R13.nll_tokens(h, y, pp, cs, caps=(8, 8))
    r = {"band_counts": nb, "S1_vs_dense": float(jnp.max(jnp.abs(sp - dense))), "S1_vs_ref": float(jnp.max(jnp.abs(sp - ref))),
         "S2_vs_dense": float(jnp.max(jnp.abs(ov - dense)))}
    g1 = jax.grad(lambda q: jnp.sum(R13.nll_tokens(h, y, q, cs, caps=(32, 32))))(pp)
    g2 = jax.grad(lambda q: jnp.sum(R13.nll_tokens(h, y, q, cd)))(pp)
    r["S3_rel"] = {"emb": rel(g1["emb"], g2["emb"]), **{f"head.{x}": rel(g1["head"][x], g2["head"][x]) for x in g2["head"]}}
    # S4: realistic chunk
    N = 4096 if V == 50257 else 1024
    p_tok = np.load(R13.counts_file(V))[:V].astype(np.float64)
    y4 = jnp.asarray(np.random.default_rng(0).choice(V, N, p=p_tok / p_tok.sum()))
    h4 = jax.random.normal(jax.random.PRNGKey(5), (N, cs.d))
    caps = R13.band_caps(cs, N)
    cnt = [int(np.sum(t["band"][np.asarray(y4)] == b)) for b in (1, 2)]
    l_s = R13.head_loss_sum(pp, h4, y4, cs)
    l_d = R13.head_loss_sum(pp, h4, y4, cd)
    G1 = jax.grad(lambda q: R13.head_loss_sum(q, h4, y4, cs))(pp)
    G2 = jax.grad(lambda q: R13.head_loss_sum(q, h4, y4, cd))(pp)
    r["S4"] = {"chunk": N, "caps": caps, "band_counts": cnt, "sparse_branch_taken": all(c <= k for c, k in zip(cnt, caps)),
               "loss_rel": float(abs(l_s - l_d) / abs(l_d)),
               "grad_rel": {"emb": rel(G1["emb"], G2["emb"]), **{f"head.{x}": rel(G1["head"][x], G2["head"][x]) for x in G2["head"]}}}
    passed = (all(c > 8 for c in nb) and all(c <= 32 for c in nb) and r["S1_vs_dense"] < 1e-5 and r["S1_vs_ref"] < 1e-4
              and r["S2_vs_dense"] < 1e-5 and max(r["S3_rel"].values()) < 1e-5 and r["S4"]["sparse_branch_taken"]
              and r["S4"]["loss_rel"] < 1e-6 and max(r["S4"]["grad_rel"].values()) < 1e-5)
    r["PASS"] = bool(passed)
    ok &= passed
    res[arm] = r
    print(arm, json.dumps(r), flush=True)
res["ALL_PASS"] = bool(ok)
json.dump(res, open(OUT, "w"), indent=1)
print("ALL_PASS", ok)
