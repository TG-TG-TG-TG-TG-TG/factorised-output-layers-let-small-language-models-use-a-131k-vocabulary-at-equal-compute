"""Round 11 Part C gates for the re-scored two-level head (round11_heads.py). CPU, fp32 reference.

Batch: the first 2 rows of data.val_rows(1024) (2,048 targets; a local file, no HF stream).
Params: the trained H3k8c4 checkpoint (wsd_H3k8c4_s0) with a_c ~ N(0, 0.5) and the given M, so neither the
re-scoring nor the offsets are trivial; plus the M = 0 / init checks.
  G1  the full-vocabulary reference (round11_heads.ref_logprobs_full: dense one-hot class means, segment-op word LSEs,
      the same top-M rule on its own tied-mean logits) sums to 1 for every token (|logsumexp| < 1e-5), picks the
      same M clusters as the fast path, and the fast mean NLL matches the reference's to 1e-5.
  G2  d(mean NLL)/d(emb) and d/d(ac) of the fast path match the reference's to 1e-5 relative (||diff|| / ||ref||).
  G3  M = 0: tH3rs0 has exactly tH3k8c4's params, loss and gradients at init (bitwise).
  G4  causality: changing every input token after position t leaves the per-token NLL of targets 0..t unchanged
      (fast path, M > 0); and the jitted loss traces twice (two shapes) without a leaked tracer.
  (data hash: checked from the smoke logs by round11_smoke_check.py)

python round11_gates.py 16 [64]      -> runs/round11/gates_partC.txt / .json
"""
import json
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

from arms import ARMS
from data import val_rows
from model import forward, init_params, lm_loss
from round8_heads import load_params
import round11_heads as R11
import round9_heads as R9

OUT = os.path.join("runs", "round11")
lines, res, fails = [], {}, []


def say(m):
    print(m, flush=True)
    lines.append(m)


def check(name, ok, detail):
    res[name] = {"ok": bool(ok), "detail": detail}
    say(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


rows = jnp.asarray(val_rows(1024)[:2])
inp, tgt = rows[:, :-1], rows[:, 1:]
y = tgt.reshape(-1)
F32 = jnp.float32
for M in [int(m) for m in sys.argv[1:]] or [16]:
    cfg = replace(ARMS[f"tH3rs{M}"], compute_dtype="float32", attn_impl=None)
    base = load_params(replace(ARMS["tH3k8c4"], compute_dtype="float32", attn_impl=None), "wsd_H3k8c4_s0")
    p = jax.tree.map(lambda x: x, base)
    p["head"] = dict(p["head"], ac=0.5 * jax.random.normal(jax.random.PRNGKey(7), (cfg.head_C,), F32))
    h = forward(p, inp, cfg).reshape(-1, cfg.d)

    # G1
    ref, idx_r = R11.ref_logprobs_full(p, h, cfg)
    lse = jax.nn.logsumexp(ref, -1)
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, _, _ = tb
    Wt, bt = R9.top_matrix(p, cfg, tb)
    _, idx_f = R11.rescored_top(h, Wt, bt, R9.word_matrix(p, cfg), p["head"]["ac"], members, mask, cfg.head_adapt, M)
    same = float(jnp.mean(jnp.all(jnp.sort(idx_f, 1) == jnp.sort(idx_r, 1), 1)))
    nll_ref = float(-jnp.mean(jnp.take_along_axis(ref, y[:, None], 1)))
    nll_fast = float(lm_loss(p, rows, cfg))
    tot, hits, ntail = R11.head_loss_sum(p, h, y, cfg, with_hits=True)
    check(f"G1 M={M} reference sums to 1", float(jnp.max(jnp.abs(lse))) < 1e-5, f"max |logsumexp| {float(jnp.max(jnp.abs(lse))):.2e}")
    check(f"G1 M={M} same top-M clusters", same == 1.0, f"fraction of tokens with identical top-{M} sets {same:.6f}")
    check(f"G1 M={M} fast NLL == reference", abs(nll_fast - nll_ref) < 1e-5,
          f"fast {nll_fast:.7f} ref {nll_ref:.7f} |diff| {abs(nll_fast - nll_ref):.2e}; tail targets in top-{M}: {int(hits)}/{int(ntail)}")
    res[f"hit_fraction_gate_batch_M{M}"] = int(hits) / max(1, int(ntail))

    # G2
    def f_fast(pp):
        return lm_loss(pp, rows, cfg)

    def f_ref(pp):
        hh = forward(pp, inp, cfg).reshape(-1, cfg.d)
        lp, _ = R11.ref_logprobs_full(pp, hh, cfg, idx=idx_r)
        return -jnp.mean(jnp.take_along_axis(lp, y[:, None], 1))

    gf, gr = jax.grad(f_fast)(p), jax.grad(f_ref)(p)
    for nm, a_, b_ in (("emb", gf["emb"], gr["emb"]), ("ac", gf["head"]["ac"], gr["head"]["ac"]), ("R", gf["head"]["R"], gr["head"]["R"])):
        rel = float(jnp.linalg.norm(a_ - b_) / jnp.linalg.norm(b_))
        check(f"G2 M={M} grad {nm}", rel < 1e-5, f"relative error {rel:.2e} (|ref| {float(jnp.linalg.norm(b_)):.3e})")

    # G4
    t = 500
    rows2 = rows.at[:, t + 2:].set((rows[:, t + 2:] * 7 + 13) % 50257)  # inputs after position t change (targets <= t do not)

    def per_tok(rr):
        hh = forward(p, rr[:, :-1], cfg).reshape(-1, cfg.d)
        l, _ = R11.nll_tokens(hh, rr[:, 1:].reshape(-1), R9.word_matrix(p, cfg), Wt, bt, p["head"]["ac"], slot, tail, grp, members, mask, pos, cfg.head_adapt, M)
        return l.reshape(rr.shape[0], -1)

    a1, a2 = per_tok(rows), per_tok(rows2)
    d_before = float(jnp.max(jnp.abs(a1[:, :t + 1] - a2[:, :t + 1])))
    d_after = float(jnp.max(jnp.abs(a1[:, t + 2:] - a2[:, t + 2:])))
    check(f"G4 M={M} causal", d_before == 0.0 and d_after > 0, f"max |ΔNLL| targets <= t: {d_before:.1e}; after: {d_after:.2e}")
    jl = jax.jit(lambda pp, rr: lm_loss(pp, rr, cfg))
    try:
        jl(p, rows).block_until_ready()
        jl(p, rows[:, :513]).block_until_ready()
        jax.grad(lambda pp: lm_loss(pp, rows[:1], cfg))(p)
        check(f"G4 M={M} no tracer leak", True, "jit at two shapes and grad traced cleanly")
    except Exception as e:
        check(f"G4 M={M} no tracer leak", False, f"{type(e).__name__}: {e}")

# G3 (bitwise, the training compute dtype)
c0, cb = replace(ARMS["tH3rs0"], attn_impl=None), replace(ARMS["tH3k8c4"], attn_impl=None)
p0, pb = init_params(c0, jax.random.PRNGKey(0)), init_params(cb, jax.random.PRNGKey(0))
same_tree = jax.tree.structure(p0) == jax.tree.structure(pb) and all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(p0), jax.tree.leaves(pb)))
l0, g0 = jax.value_and_grad(lambda pp: lm_loss(pp, rows, c0))(p0)
lb, gb = jax.value_and_grad(lambda pp: lm_loss(pp, rows, cb))(pb)
same_g = all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(gb)))
check("G3 M=0 bit-identical to H3k8c4", same_tree and float(l0) == float(lb) and same_g,
      f"params equal {same_tree}, loss {float(l0):.7f} vs {float(lb):.7f}, grads equal {same_g}")

say("ALL GATES OK" if not fails else f"GATES FAILED: {fails}")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "gates_partC.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"checks": res, "fails": fails}, open(os.path.join(OUT, "gates_partC.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
