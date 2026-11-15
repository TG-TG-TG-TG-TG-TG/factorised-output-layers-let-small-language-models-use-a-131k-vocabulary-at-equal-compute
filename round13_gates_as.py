"""Round 13 experiment 3 gates for the adaptive-softmax head (round13_heads.py), CPU (JAX_PLATFORMS=cpu), before any run:
  A1 the reference full-vocabulary distribution sums to 1 (|logsumexp_w log p(w|h)| < 1e-5) for every row
  A2 fast-path NLL (nll_tokens) = reference NLL at the targets (max |diff| < 1e-4, mean < 1e-5)
  A3 gradients of the chunked head_loss_sum = gradients of the reference NLL sum (relative L2 < 1e-4, every head tensor
     and the embedding)
  A4 the backbone tensors at init are bit-identical to the flat B arm's for the same seed (emb, blocks, copies, ln_out)
  A5 model.flops_per_token head = 3 x the hand count, printed with the dense count and the band shares
  A6 round11_eval._nll_rows (the evaluator path) mean = model.lm_loss on the same rows (|diff| < 1e-5)
  A7 the shortlist is the two-level head's frequent list (asserted inside round13_heads.tables_np)
The params are perturbed away from init (bb, Vb, P, W scaled / shifted) so A2 / A3 are not checked at a near-uniform point.
Both vocabularies: AS384 (GPT-2, 50,257) and AS128k384 (bpe131072).

JAX_PLATFORMS=cpu python round13_gates_as.py  -> runs/round13/gates_as.json
2026-11-13 (experiment 4): optional arm list and output tag, same gates:
JAX_PLATFORMS=cpu python round13_gates_as.py AS128k384c49 c49  -> runs/round13/gates_as_c49.json
2026-11-13 (experiment 5): full-width arm (no projection P_b; the hand count drops the d x d_b term):
JAX_PLATFORMS=cpu python round13_gates_as.py AS128k384c49w c49w  -> runs/round13/gates_as_c49w.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from model import flops_per_token, init_params, lm_loss
import round11_eval as E
import round13_heads as R13

jax.config.update("jax_default_matmul_precision", "highest")
ARM_LIST = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("AS384", "AS128k384")
OUT = os.path.join(R13.HERE, "runs", "round13", f"gates_as{'_' + sys.argv[2] if len(sys.argv) > 2 else ''}.json")
res, ok = {}, True


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


for arm in ARM_LIST:
    cfg = ARMS[arm]
    V = R13.vreal(cfg)
    key = jax.random.PRNGKey(0)
    p = init_params(cfg, key)
    pb = init_params(ARMS["B"] if V == 50257 else ARMS["F128k384"], key)
    same = all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves({k: p[k] for k in pb}), jax.tree.leaves(pb)))
    r = {"A4_backbone_identical": same}
    k = iter(jax.random.split(jax.random.PRNGKey(7), 16))
    hp = dict(p["head"])
    hp["bb"] = jax.random.normal(next(k), hp["bb"].shape) * 2.0
    hp["Vb"] = hp["Vb"] * 3.0
    for b in (1, 2):
        if f"P{b}" in hp:  # full-width arms have no projection
            hp[f"P{b}"] = hp[f"P{b}"] * 2.0
        hp[f"W{b}"] = hp[f"W{b}"] + 0.3 * jax.random.normal(next(k), hp[f"W{b}"].shape)
    pp = {"emb": p["emb"] * 2.0, "head": hp}
    t = R13.tabs_np(cfg)
    proj = 0 if R13.t_full(cfg) else 1
    # targets: a mix over the shortlist and both bands, every band edge included
    c0, c1 = R13.cuts(cfg)
    ranks = np.concatenate([np.arange(0, 20), np.arange(c0, c0 + 20), np.arange(c1, c1 + 20), [V - 1, V - 2, c0 - 1, c1 - 1]])
    n = len(ranks)  # 64
    h = jax.random.normal(next(k), (n, cfg.d))
    y = jnp.asarray(t["order"][ranks])
    lp = R13.ref_logprobs_full(pp, h, cfg)
    r["A1_max_abs_lse"] = float(jnp.max(jnp.abs(jax.nn.logsumexp(lp, -1))))
    ref = -jnp.take_along_axis(lp, y[:, None], 1)[:, 0]
    fast = R13.nll_tokens(h, y, pp, cfg)
    r["A2_max_abs"] = float(jnp.max(jnp.abs(fast - ref)))
    r["A2_mean_abs"] = float(jnp.mean(jnp.abs(fast - ref)))
    g1 = jax.grad(lambda q: R13.head_loss_sum(q, h, y, cfg, chunk=16))(pp)
    g2 = jax.grad(lambda q: -jnp.sum(jnp.take_along_axis(R13.ref_logprobs_full(q, h, cfg), y[:, None], 1)))(pp)
    r["A3_rel"] = {kk: rel(a, b) for kk, a, b in [("emb", g1["emb"], g2["emb"])] + [(f"head.{x}", g1["head"][x], g2["head"][x]) for x in hp]}
    fl = flops_per_token(cfg)
    hand = 2 * (8192 + 2) * cfg.d + sum(s * 2 * (proj * cfg.d * db + nb * db) for db, nb, s in zip(t["dims"], t["sizes"], t["share"]))
    r["A5"] = {"head_train": fl["head"], "hand_x3": 3 * hand, "total": fl["total"], "non_embedding": fl["non_embedding"],
               "head_fwd_dense": R13.head_flops_fwd(cfg, dense=True), "shares": t["share"], "sizes": t["sizes"], "dims": t["dims"],
               "head_params": R13.n_head_params(cfg)}
    rows = jax.random.randint(next(k), (2, 257), 0, V)
    pf = {**p, "head": hp, "emb": p["emb"]}
    l_model = float(lm_loss(pf, rows, cfg))
    l_eval = float(jnp.mean(E._nll_rows(pf, rows, cfg)))
    r["A6"] = {"lm_loss": l_model, "eval_nll_rows": l_eval, "diff": abs(l_model - l_eval)}
    passed = (same and r["A1_max_abs_lse"] < 1e-5 and r["A2_max_abs"] < 1e-4 and r["A2_mean_abs"] < 1e-5
              and max(r["A3_rel"].values()) < 1e-4 and abs(fl["head"] - 3 * hand) < 1 and r["A6"]["diff"] < 1e-5)
    r["PASS"] = bool(passed)
    ok &= passed
    res[arm] = r
    print(arm, json.dumps(r), flush=True)
res["ALL_PASS"] = bool(ok)
json.dump(res, open(OUT, "w"), indent=1)
print("ALL_PASS", ok)
