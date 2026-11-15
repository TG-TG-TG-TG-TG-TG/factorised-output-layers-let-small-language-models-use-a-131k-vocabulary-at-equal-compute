"""Round 10 gates for Parts C and D (CPU), on the first micro-batch rows of data.Batches(64, seed 0).

Every arm: accounting (params split into blocks / per-copy (norms, alphas, gates) / tied embedding / head /
n-gram tables; training FLOPs/token x tB), head gate 1 (fast NLL == independent full-vocab reference, sums to 1;
init backbone with random head params R, bc ~ N(0, .) so the class logits are not flat) and gate 2 (dL/demb,
dL/dhead fast == reference), and the tracer-leak test (gate 5).
Part C:
  C1 schedule: R-loop = 6 unique blocks x 4 passes in cycle order (0..5, 0'..5', ...), 24 distinct copies;
     R-deep 24 unique blocks; R-wide 6 blocks at d 256
  C2 alpha init = 1 / sqrt(2 * n_eff) on every copy (round-5 rule scaled to the block applications)
  C3 R-loop's shared block tensors at init are bit-identical to tB's (same key split), gates gam = 1, beta = 0
Part D (n-gram arms vs their head arm H3*):
  D1 init: lm_loss and every shared gradient bit-identical to the H3* arm (tables are zeros)
  D2 hash ids == a pure-Python reference (python ints, explicit document scan) on 2 real rows, which contain
     EOTs and the row start (both must give the reserved id for missing previous tokens)
  D3 lazy path: segment-summed row gradients (round10_lazy.sparse_grads) == the dense table gradient
  D4 lazy AdamW rows (round10_lazy.adamw_rows) == optax.adamw on a toy table for touched rows, 3 steps
     (rows touched every step), and untouched rows unchanged

python round10_gates_cd.py Rloop Rdeep Rwide D1   -> runs/round10/gates_cd_<arms>.txt / .json (exit 1 on a failure)
"""
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

import model as M
import round9_heads as R9
import round10_lazy as LZ
from arms import ARMS
from data import Batches

OUT = "runs/round10"
ARM_LIST = sys.argv[1:]
lines, res, fails = [], {}, []


def say(s):
    print(s, flush=True)
    lines.append(s)


def gate(name, ok, detail):
    say(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    res[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


n_ = lambda t: int(sum(x.size for x in jax.tree.leaves(t)))
rows = jnp.asarray(Batches(64, 1024, seed=0, log=lambda m: None).next()[:4])
NF = 2048
FB = M.flops_per_token(ARMS["tB"])["total"]
PB = M.init_params(ARMS["tB"], jax.random.PRNGKey(0))

for arm in ARM_LIST:
    cfg = ARMS[arm]
    p0 = M.init_params(cfg, jax.random.PRNGKey(0))
    fl = M.flops_per_token(cfg)
    acct = {"blocks": n_(p0["blocks"]), "per_copy": n_(p0["copies"]) + n_(p0["ln_out"]), "tied_embedding": int(p0["emb"].size),
            "head": n_(p0.get("head", {})), "ngram": sum(int(p0[k].size) for k in p0 if k.startswith("ng")),
            "total": n_(p0), "train_flops_per_token": fl, "train_flops_vs_tB": fl["total"] / FB,
            "n_eff": cfg.n_eff, "alpha0": 1.0 / math.sqrt(2 * cfg.n_eff), "d": cfg.d, "f": cfg.f, "H": cfg.H}
    res[f"acct {arm}"] = acct
    say(f"{arm}: params total {acct['total']:,} = blocks {acct['blocks']:,} + per-copy {acct['per_copy']:,} + emb "
        f"{acct['tied_embedding']:,} + head {acct['head']:,} + n-gram {acct['ngram']:,}; train FLOPs/token "
        f"{fl['total']:,.0f} = {acct['train_flops_vs_tB']:.4f} x tB; n_eff {cfg.n_eff}, alpha0 {acct['alpha0']:.5f}")
    # C1 / C2 / C3
    sch = M.schedule(cfg)
    if cfg.arch == "cycle":
        want = [(i, min(r, 1), r * cfg.n_unique + i) for r in range(cfg.n_loops) for i in range(cfg.n_unique)]
        gate(f"C1 {arm} schedule", sch == want and len(p0["copies"]) == cfg.n_eff == len({c for _, _, c in sch}),
             f"{len(sch)} applications, blocks {[b for b, _, _ in sch]}, {len(p0['copies'])} copies")
        same = all(np.array_equal(np.asarray(p0["blocks"][i][t]), np.asarray(PB["blocks"][i][t]))
                   for i in range(6) for t in M.SHARED) and cfg.n_unique == 6
        gg = all(np.all(np.asarray(c["gam"]) == 1) and np.all(np.asarray(c["beta"]) == 0) for c in p0["copies"])
        gate(f"C3 {arm} shared blocks == tB init, gates gam 1 beta 0", same and gg and cfg.gate, f"blocks {same}, gates {gg}")
    else:
        gate(f"C1 {arm} schedule", len(sch) == cfg.n_eff and [b for b, _, _ in sch] == list(range(cfg.n_unique)),
             f"{len(sch)} unique blocks, d {cfg.d}")
    if cfg.learn_alpha:
        a0 = 1.0 / math.sqrt(2 * cfg.n_eff)
        ok = all(np.allclose(np.asarray(c[k]), a0, rtol=0, atol=0) for c in p0["copies"] for k in ("a_attn", "a_ffn"))
        gate(f"C2 {arm} alpha init 1/sqrt(2*{cfg.n_eff})", ok, f"{a0:.6f} on all {len(p0['copies'])} copies")
    # head gates 1 and 2 (init backbone, random head params); a flat-head arm (RB24) has no two-level head to check
    if "head" in p0:
        k = iter(jax.random.split(jax.random.PRNGKey(7), 4))
        hp = {"R": (0.5 / np.sqrt(cfg.d)) * jax.random.normal(next(k), p0["head"]["R"].shape),
              "bc": 0.5 * jax.random.normal(next(k), p0["head"]["bc"].shape)}
        p = dict(p0, head=hp)
        if cfg.ngram:  # non-zero tables so the n-gram path is live in the head check
            p = dict(p, **{kk: 0.02 * jax.random.normal(jax.random.PRNGKey(11 + i), p0[kk].shape) for i, kk in enumerate(sorted(p0)) if kk.startswith("ng")})
        h_all = M.forward(p, rows[:2, :-1], cfg).reshape(-1, cfg.d)
        y = rows[:2, 1:].reshape(-1)
        lm = float(M.lm_loss(p, rows[:2], cfg))
        tbj = R9.tables(cfg)
        ref_mean = float(np.mean([float(-jnp.take_along_axis(R9.ref_logprobs_full(p, h_all[i:i + NF], cfg), y[i:i + NF, None], 1).mean())
                                  for i in range(0, len(y), NF)]))
        nll_fast = R9.hier9_nll(h_all[:NF], y[:NF], R9.word_matrix(p, cfg), *R9.top_matrix(p, cfg, tbj), *tbj[:6])
        lp = R9.ref_logprobs_full(p, h_all[:NF], cfg)
        d_tok = float(jnp.abs(nll_fast + jnp.take_along_axis(lp, y[:NF, None], 1)[:, 0]).max())
        gate(f"1 {arm} lm_loss == mean -ref[y] (2048 tok)", abs(lm - ref_mean) < 1e-5, f"{lm:.6f} vs {ref_mean:.6f}")
        gate(f"1 {arm} fast == -ref[y] per token", d_tok < 1e-5, f"max |diff| {d_tok:.2e}")
        gate(f"1 {arm} sum exp(ref) == 1", float(jnp.abs(jnp.exp(lp).sum(-1) - 1).max()) < 1e-5, "")
        hN, yN = h_all[:NF], y[:NF]
        f_fast = lambda q: jnp.mean(R9.hier9_nll(hN, yN, R9.word_matrix(q, cfg), *R9.top_matrix(q, cfg, R9.tables(cfg)), *R9.tables(cfg)[:6]))
        f_ref = lambda q: -jnp.mean(jnp.take_along_axis(R9.ref_logprobs_full(q, hN, cfg), yN[:, None], 1)[:, 0])
        sub = {"emb": p["emb"], "head": p["head"]}
        gf, gr = jax.grad(f_fast)(sub), jax.grad(f_ref)(sub)
        for nm, a_, b_ in [("emb", gf["emb"], gr["emb"])] + [(f"head.{q}", gf["head"][q], gr["head"][q]) for q in gf["head"]]:
            m = float(jnp.abs(b_).max())
            rel = float(jnp.abs(a_ - b_).max()) / (m + 1e-30)
            gate(f"2 {arm} dL/d{nm} fast == ref", rel < 1e-5, f"rel {rel:.2e}")
    # Part D
    if cfg.ngram:
        hname = next(kk for kk, vv in ARMS.items() if vv == M.Config(**{**cfg.__dict__, "ngram": 0}))
        cb = ARMS[hname]
        pb = M.init_params(cb, jax.random.PRNGKey(0))
        la, lb = float(M.lm_loss(p0, rows[:2], cfg)), float(M.lm_loss(pb, rows[:2], cb))
        ga = jax.grad(lambda q: M.lm_loss(q, rows[:2], cfg))(p0)
        gb = jax.grad(lambda q: M.lm_loss(q, rows[:2], cb))(pb)
        gsame = all(np.array_equal(np.asarray(x), np.asarray(z)) for x, z in zip(jax.tree.leaves({kk: ga[kk] for kk in gb}), jax.tree.leaves(gb)))
        bb = all(np.array_equal(np.asarray(x), np.asarray(z)) for x, z in zip(jax.tree.leaves({kk: p0[kk] for kk in pb}), jax.tree.leaves(pb)))
        gate(f"D1 {arm} init == {hname} (params, loss, shared grads)", la == lb and gsame and bb, f"loss {la:.7f} vs {lb:.7f}, grads {gsame}, params {bb}")
        # D2: hash ids vs pure python
        tok = np.asarray(rows[:2, :-1])
        for n in range(2, cfg.ngram + 1):
            fast = np.asarray(M.ngram_ids(jnp.asarray(tok), n, cfg))
            ms = M.NG_MULT[n]
            ref = np.zeros_like(fast)
            n_miss = 0
            for r in range(tok.shape[0]):
                doc_start = 0
                for t in range(tok.shape[1]):
                    ctx = []
                    for lag in range(n - 1, -1, -1):
                        j = t - lag
                        ctx.append(int(tok[r, j]) if j >= doc_start else M.NG_MISSING)
                    n_miss += ctx[0] == M.NG_MISSING
                    ref[r, t] = (sum(m_ * x for m_, x in zip(ms, ctx)) % M.NG_P) % cfg.ngram_M
                    if tok[r, t] == M.EOT_ID:
                        doc_start = t + 1
            n_eot = int((tok == M.EOT_ID).sum())
            gate(f"D2 {arm} ng{n} ids == python reference", np.array_equal(fast, ref),
                 f"{fast.size} positions, {n_eot} EOTs, {n_miss} with a missing previous token, {len(np.unique(fast))} distinct ids")
        # D3: sparse row grads == dense table grad
        pz = dict(p0, **{kk: 0.02 * jax.random.normal(jax.random.PRNGKey(3), p0[kk].shape) for kk in p0 if kk.startswith("ng")})
        b = rows[:2]
        gd = jax.grad(lambda q: M.lm_loss(q, b, cfg))(pz)
        NG = [kk for kk in pz if kk.startswith("ng")]
        ids = {kk: M.ngram_ids(b[:, :-1], int(kk[2:]), cfg) for kk in NG}
        dense = {kk: v for kk, v in pz.items() if kk not in NG}
        rr = {kk + "_rows": pz[kk][ids[kk]] for kk in NG}
        g_rows = jax.grad(lambda rr_: M.lm_loss({**dense, **rr_}, b, cfg))(rr)
        for kk in NG:
            N = ids[kk].size
            u, g = LZ.sparse_grads(ids[kk].reshape(N), g_rows[kk + "_rows"].reshape(N, -1), cfg.ngram_M, 1)
            un = np.asarray(u)
            valid = un < cfg.ngram_M
            G = np.asarray(gd[kk])
            err = np.abs(np.asarray(g)[valid] - G[un[valid]]).max() / (np.abs(G).max() + 1e-30)
            others = np.setdiff1d(np.arange(cfg.ngram_M), un[valid])
            zero_else = float(np.abs(G[others]).max()) == 0.0
            gate(f"D3 {arm} {kk} sparse row grads == dense grad", err < 1e-5 and zero_else,
                 f"{valid.sum()} rows, rel err {err:.2e}, untouched rows' dense grad all zero {zero_else}")

# D4: lazy AdamW vs optax on a toy table (independent of arms)
if any(ARMS[a].ngram for a in ARM_LIST):
    Mt, d = 64, 8
    P = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (Mt, d))
    touched = jnp.array([3, 7, 7, 20, 41, 63], jnp.int32)
    opt = optax.adamw(1.0, b1=0.9, b2=0.95, weight_decay=0.1)
    st = opt.init(P)
    Pl, m, v = P, jnp.zeros_like(P), jnp.zeros_like(P)
    Pd = P
    lr = 2e-3
    for s in range(1, 4):
        gro = jax.random.normal(jax.random.PRNGKey(10 + s), (len(touched), d))
        Gd = jnp.zeros_like(P).at[touched].add(gro)
        upd, st = opt.update(Gd, st, Pd)
        Pd = optax.apply_updates(Pd, jax.tree.map(lambda x: lr * x, upd))
        u, g = LZ.sparse_grads(touched, gro, Mt, 1)
        Pl, m, v = LZ.adamw_rows(Pl, m, v, u, g, jnp.int32(s), lr, 0.1)
    tu = np.unique(np.asarray(touched))
    un = np.setdiff1d(np.arange(Mt), tu)
    e_t = float(np.abs(np.asarray(Pl)[tu] - np.asarray(Pd)[tu]).max())
    e_u = float(np.abs(np.asarray(Pl)[un] - np.asarray(P)[un]).max())
    gate("D4 lazy AdamW == optax.adamw on touched rows (3 steps), untouched rows unchanged", e_t < 1e-7 and e_u == 0.0,
         f"max |diff| touched {e_t:.2e}, untouched change {e_u:.1e} (dense AdamW decays untouched rows: "
         f"{float(np.abs(np.asarray(Pd)[un] - np.asarray(P)[un]).max()):.2e})")

# gate 5: tracer leak, every arm
for arm in ARM_LIST:
    cfg = ARMS[arm]
    p0 = M.init_params(cfg, jax.random.PRNGKey(0))
    r2 = rows[:2]

    @jax.jit
    def steps(q):
        def body(q, _):
            l, g = jax.value_and_grad(lambda z: M.lm_loss(z, r2, cfg))(q)
            return jax.tree.map(lambda a, b: a - 1e-3 * b, q, g), l
        return jax.lax.scan(body, q, None, length=2)

    try:
        q, ls = steps(p0)
        ev = float(jax.jit(lambda z: M.lm_loss(z, r2, cfg))(q))
        q, ls2 = steps(q)
        gate(f"5 {arm} tracer leak (scan, jit eval, scan)", np.isfinite(ev) and bool(jnp.all(jnp.isfinite(ls2))),
             f"eval {ev:.5f}, losses {[round(float(x), 5) for x in ls]} -> {[round(float(x), 5) for x in ls2]}")
    except Exception as e:  # noqa: BLE001
        gate(f"5 {arm} tracer leak (scan, jit eval, scan)", False, f"{type(e).__name__}: {str(e)[:300]}")

say(f"\nALL GATES {'OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
res["fails"] = fails
tag = "_".join(ARM_LIST)
open(os.path.join(OUT, f"gates_cd_{tag}.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, f"gates_cd_{tag}.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
