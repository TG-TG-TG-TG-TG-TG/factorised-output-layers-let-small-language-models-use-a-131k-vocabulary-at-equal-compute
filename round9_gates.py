"""Round 9 gates (CPU), on one real batch (the first micro-batch of data.Batches(64, seed 0): 16 x 1024 targets).

For every new arm, twice: (a) its seed-0 init, (b) B_r8's trained backbone + embedding with random head params
(R, bc, Vc ~ N(0, .)), where logits are sharp enough to expose a wrong index map.
 1 fast NLL (model.lm_loss, the training path) == -ref[y] (round9_heads.ref_logprobs_full, built independently)
   to 1e-5 on the mean and per token (2048 positions); max |sum exp(ref) - 1| < 1e-5
 2 dL/d(emb) and dL/d(head) of the mean NLL, fast vs reference, max |diff| / max |grad| < 1e-5 (2048 positions;
   for the tied-mean arms the reference's ubar is a dense one-hot matmul of emb, so dL/dU flows through it)
 3 adaptive split: frequent and tail sets disjoint, together all 50,257 real tokens; cluster sizes
 4 backbone tensors at init bit-identical to tB's (seed 0); the data-hash half is checked on the smoke runs
   (round9_smoke_check.py vs runs/round8/data_hash_ref.json)
 5 tracer leak: a jitted lax.scan of value_and_grad steps, then a separate jit eval, then the scan again
 + accounting: params, new params, head forward FLOPs/token (x B)

python round9_gates.py [arms...]   -> runs/round9/gates.txt, gates.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import Batches
from model import count_params, flops_per_token, forward, init_params, lm_loss
from round8_heads import head_flops_fwd as hf8, load_params
import round9_heads as R9

OUT = "runs/round9"
ARM_LIST = sys.argv[1:] or ["tH2tm", "tH2a", "tH2atm", "tH2u"]
lines, res, fails = [], {}, []


def say(s):
    print(s, flush=True)
    lines.append(s)


def gate(name, ok, detail):
    say(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    res[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


data = Batches(64, 1024, seed=0, log=lambda m: None)
rows = jnp.asarray(data.next()[:16])
NF = 2048
PB = {k: init_params(ARMS[k], jax.random.PRNGKey(0)) for k in ("tB", "B")}
trained = load_params(ARMS["tB"], "wsd_B_r8_s0")

for arm in ARM_LIST:
    cfg = ARMS[arm]
    base = "tB" if cfg.d == 128 else "B"
    p0 = init_params(cfg, jax.random.PRNGKey(0))
    pc, pb = count_params(p0), count_params(PB[base])
    hf, hb = R9.head_flops_fwd(cfg), hf8(ARMS[base])
    res[f"acct {arm}"] = {"total_params": pc["total"], "new_params": pc["total"] - pb["total"],
                          "head_fwd_flops_per_token": hf, "head_ratio_to_B": hf / hb,
                          "train_flops_per_token": flops_per_token(cfg)}
    say(f"{arm}: params {pc['total']:,} (new {pc['total'] - pb['total']:,}); head fwd FLOPs/token {hf:,.0f} = {hf / hb:.4f} x B")
    # gate 4
    bb = {k: v for k, v in p0.items() if k != "head"}
    la, lb = jax.tree.leaves(bb), jax.tree.leaves(PB[base])
    same = jax.tree.structure(bb) == jax.tree.structure(PB[base]) and all(
        x.dtype == y.dtype and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))
    gate(f"4 {arm} backbone init bit-identical to {base}", same, f"{len(la)} tensors")
    # gate 3
    tb = R9.tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt)
    slot, tail, grp, members, mask, pos, freq, count = tb
    if cfg.head_adapt:
        ft, tt = set(freq.tolist()), set(np.flatnonzero(tail).tolist())
        ok = not (ft & tt) and len(ft | tt) == R9.V_REAL and len(ft) == cfg.head_adapt
        h = np.bincount(count.astype(int))
        gate(f"3 {arm} frequent/tail disjoint and cover", ok,
             f"{len(ft)} frequent + {len(tt)} tail = {len(ft | tt)}; cluster sizes min {int(count.min())} "
             f"max {int(count.max())}, histogram {{{', '.join(f'{s}: {n}' for s, n in enumerate(h) if n)}}}")
    # slot map is a bijection onto the top level for singletons, and member tables invert pos
    inv_ok = all(members[grp[w], pos[w]] == w for w in np.flatnonzero(tail)[::97])
    if not inv_ok:
        gate(f"tables {arm}", False, "members[grp, pos] != w")
    if cfg.d != 128:
        continue  # d=384 arms: accounting and gate 4 only (the head code is the same)
    for tag, p in (("init", p0), ("trainedB", None)):
        if p is None:  # trained backbone + emb, random head params
            k = iter(jax.random.split(jax.random.PRNGKey(7), 8))
            hp = {n: (0.5 / np.sqrt(cfg.d)) * jax.random.normal(next(k), v.shape, jnp.float32) if n != "Uo"
                  else trained["emb"][:R9.V_REAL] + 0.01 * jax.random.normal(next(k), v.shape)
                  for n, v in p0["head"].items()}
            hp["bc"] = 0.5 * jax.random.normal(next(k), hp["bc"].shape)
            p = dict(trained, head=hp)
        h_all = forward(p, rows[:, :-1], cfg).reshape(-1, cfg.d)
        y = rows[:, 1:].reshape(-1)
        lm = float(lm_loss(p, rows, cfg))
        nll_fast = R9.hier9_nll(h_all[:NF], y[:NF], R9.word_matrix(p, cfg), *R9.top_matrix(p, cfg, R9.tables(cfg)),
                                *R9.tables(cfg)[:6])
        lp = R9.ref_logprobs_full(p, h_all[:NF], cfg)
        nll_ref = -jnp.take_along_axis(lp, y[:NF, None], 1)[:, 0]
        # full-batch mean via the reference, in 2048-token pieces
        ref_mean = float(np.mean([float(-jnp.take_along_axis(R9.ref_logprobs_full(p, h_all[i:i + NF], cfg),
                                                             y[i:i + NF, None], 1).mean()) for i in range(0, len(y), NF)]))
        d_tok = float(jnp.abs(nll_fast - nll_ref).max())
        d_sum = float(jnp.abs(jnp.exp(lp).sum(-1) - 1).max())
        gate(f"1 {arm} [{tag}] lm_loss == mean -ref[y] (16384 tok)", abs(lm - ref_mean) < 1e-5, f"{lm:.6f} vs {ref_mean:.6f}")
        gate(f"1 {arm} [{tag}] fast == -ref[y] per token ({NF})", d_tok < 1e-5, f"max |diff| {d_tok:.2e}")
        gate(f"1 {arm} [{tag}] sum exp(ref) == 1", d_sum < 1e-5, f"max |sum - 1| {d_sum:.2e}")
        # gate 2: gradients wrt emb and head params, on the first NF positions, h held fixed
        hN, yN = h_all[:NF], y[:NF]

        def f_fast(q):
            tbj = R9.tables(cfg)
            return jnp.mean(R9.hier9_nll(hN, yN, R9.word_matrix(q, cfg), *R9.top_matrix(q, cfg, tbj), *tbj[:6]))

        def f_ref(q):
            return -jnp.mean(jnp.take_along_axis(R9.ref_logprobs_full(q, hN, cfg), yN[:, None], 1)[:, 0])

        sub = {"emb": p["emb"], "head": p["head"]}
        gf, gr = jax.grad(f_fast)(sub), jax.grad(f_ref)(sub)
        for name, a_, b_ in [("emb", gf["emb"], gr["emb"])] + [(f"head.{n}", gf["head"][n], gr["head"][n]) for n in gf["head"]]:
            m = float(jnp.abs(b_).max())
            rel = float(jnp.abs(a_ - b_).max()) / (m + 1e-30)
            gate(f"2 {arm} [{tag}] dL/d{name} fast == ref", rel < 1e-5, f"max |diff| / max |grad| {rel:.2e} (max |grad| {m:.2e})")
    # gate 5: tracer leak (scan of training-like steps, then a separate jit eval, then the scan again)
    r2 = rows[:2]

    @jax.jit
    def steps(q):
        def body(q, _):
            l, g = jax.value_and_grad(lambda z: lm_loss(z, r2, cfg))(q)
            return jax.tree.map(lambda a, b: a - 1e-3 * b, q, g), l
        return jax.lax.scan(body, q, None, length=2)

    try:
        q, ls = steps(p0)
        ev = float(jax.jit(lambda z: lm_loss(z, r2, cfg))(q))
        q, ls2 = steps(q)
        gate(f"5 {arm} tracer leak (scan, jit eval, scan)", np.isfinite(ev) and bool(jnp.all(jnp.isfinite(ls2))),
             f"eval {ev:.5f}, losses {[round(float(x), 5) for x in ls]} -> {[round(float(x), 5) for x in ls2]}")
    except Exception as e:  # noqa: BLE001
        gate(f"5 {arm} tracer leak (scan, jit eval, scan)", False, f"{type(e).__name__}: {str(e)[:300]}")

say(f"\nALL GATES {'OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
res["fails"] = fails
tag = "" if len(sys.argv) == 1 else "_" + "_".join(ARM_LIST)
open(os.path.join(OUT, f"gates{tag}.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, f"gates{tag}.json"), "w"), indent=1)
