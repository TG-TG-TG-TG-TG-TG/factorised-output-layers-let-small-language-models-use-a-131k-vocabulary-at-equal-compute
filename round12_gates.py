"""Round 12 Part B gates (rules pinned in round12_analyze.py). CPU, fp32 reference (jax_default_matmul_precision highest).

Batch: data.val_rows(1024, source="bpe131072") (the bpe131072 re-tokenised val docs, a local cache).
python round12_gates.py flat     G3: F128k-384's chunked CE (model.lm_loss, 1,024-token chunks, 2 rows = 2 chunks) equals
                                 an unchunked reference (full logits, logsumexp) within 1e-5, loss and gradients (rel.).
python round12_gates.py hier     (needs the class files) H128k-384 at init with R ~ N(0, 0.5/sqrt d), bc ~ N(0, 1):
                                 G1 the full-vocabulary reference (round9_heads.ref_logprobs_full) sums to 1 (|logsumexp|
                                 < 1e-5) and its mean NLL = the fast path's (lm_loss) within 1e-5; G2 gradients of the
                                 two (emb, R, bc) agree to 1e-5 relative; exact greedy (round9_decode.exact_greedy) =
                                 argmax of the reference on 1,000 val contexts (h and 8h, so both the certified path and
                                 the fallback run); exact sampling: two-stage p(slot) p(w | cluster) (members-table gather,
                                 a different code path) = exp(reference) within 1e-5 on the same 1,000 contexts, and an
                                 empirical TV of 200k two-stage samples within 1.5x that of 200k direct reference samples.
python round12_gates.py trained <ckpt>   greedy on the trained checkpoint (4,096 val positions): exact-match rate,
                                 fallback rate, mean clusters expanded (3 when certified; all C when the fallback fires).
G4 (data hash identical for the F128k / H128k smokes) is checked from the smoke logs (round12_smoke_check.py / by hand).
-> runs/round12/gates_<mode>.txt / .json
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
import round9_heads as R9
from round9_decode import exact_greedy

OUT = os.path.join("runs", "round12")
F32 = jnp.float32
lines, res, fails = [], {}, []
mode = sys.argv[1]


def say(m):
    print(m, flush=True)
    lines.append(m)


def check(name, ok, detail):
    res[name] = {"ok": bool(ok), "detail": detail}
    say(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def chunked(g, x, n=8):
    """vmap turns exact_greedy's fallback cond into a select, so every vmapped context materialises the full
    [C, L, d] cluster gather (8 GB at V = 131k); run the same jitted vmap 8 contexts at a time."""
    outs = [g(x[i:i + n]) for i in range(0, len(x), n)]
    return tuple(jnp.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))


val = val_rows(1024, source="bpe131072")
assert val.max() < 131072

if mode == "flat":
    cfg = replace(ARMS["F128k384"], compute_dtype="float32", attn_impl=None)
    p = init_params(cfg, jax.random.PRNGKey(0))
    rows = jnp.asarray(val[:2].astype(np.int32))

    def f_ref(pp):
        h = forward(pp, rows[:, :-1], cfg).reshape(-1, cfg.d)
        z = h @ pp["emb"].astype(F32).T
        return jnp.mean(jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, rows[:, 1:].reshape(-1)[:, None], 1)[:, 0])

    lf, gf = jax.value_and_grad(lambda pp: lm_loss(pp, rows, cfg))(p)
    lr_, gr = jax.value_and_grad(f_ref)(p)
    check("G3 F128k chunked CE == unchunked (loss)", abs(float(lf) - float(lr_)) < 1e-5,
          f"chunked {float(lf):.7f} unchunked {float(lr_):.7f} |diff| {abs(float(lf) - float(lr_)):.2e} (2,048 targets, 2 chunks of 1,024)")
    e = max(rel(a, b) for a, b in zip(jax.tree.leaves(gf), jax.tree.leaves(gr)) if float(jnp.linalg.norm(b)) > 0)
    check("G3 F128k chunked CE == unchunked (grads)", e < 1e-5, f"max relative gradient error over leaves {e:.2e}")

elif mode == "hier":
    cfg = replace(ARMS["H128k384"], compute_dtype="float32", attn_impl=None)
    K, C, V = cfg.head_adapt, cfg.head_C, R9.vreal(cfg)
    p = init_params(cfg, jax.random.PRNGKey(0))
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    p["head"] = dict(p["head"], R=0.5 / np.sqrt(cfg.d) * jax.random.normal(k1, (C, cfg.d), F32), bc=jax.random.normal(k2, (C,), F32))
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, freq, count = tb
    res["tables"] = {"V": V, "K": K, "C": C, "L": int(members.shape[1]), "min_count": int(count.min()), "max_count": int(count.max())}
    say(f"tables: V {V} K {K} C {C} L {members.shape[1]} cluster sizes {int(count.min())}..{int(count.max())}")
    rows = jnp.asarray(val[:1].astype(np.int32))
    h = forward(p, rows[:, :-1], cfg).reshape(-1, cfg.d)
    y = rows[:, 1:].reshape(-1)
    ref = R9.ref_logprobs_full(p, h, cfg)
    lse = jax.nn.logsumexp(ref, -1)
    check("G1 reference sums to 1", float(jnp.max(jnp.abs(lse))) < 1e-5, f"max |logsumexp| {float(jnp.max(jnp.abs(lse))):.2e}")
    nll_ref = float(-jnp.mean(jnp.take_along_axis(ref, y[:, None], 1)))
    nll_fast = float(lm_loss(p, rows, cfg))
    check("G1 fast NLL == reference", abs(nll_fast - nll_ref) < 1e-5, f"fast {nll_fast:.7f} ref {nll_ref:.7f} |diff| {abs(nll_fast - nll_ref):.2e}")
    del ref

    r2 = rows[:, :513]

    def f_ref(pp):
        hh = forward(pp, r2[:, :-1], cfg).reshape(-1, cfg.d)
        return -jnp.mean(jnp.take_along_axis(R9.ref_logprobs_full(pp, hh, cfg), r2[:, 1:].reshape(-1)[:, None], 1))

    gf, gr = jax.grad(lambda pp: lm_loss(pp, r2, cfg))(p), jax.grad(f_ref)(p)
    for nm, a_, b_ in (("emb", gf["emb"], gr["emb"]), ("R", gf["head"]["R"], gr["head"]["R"]), ("bc", gf["head"]["bc"], gr["head"]["bc"])):
        e = rel(a_, b_)
        check(f"G2 grad {nm}", e < 1e-5, f"relative error {e:.2e} (|ref| {float(jnp.linalg.norm(b_)):.3e})")
    del gf, gr

    # greedy + sampling on 1,000 contexts
    hs = h[:1000]
    Wt, bt = R9.top_matrix(p, cfg, tb)
    U = R9.word_matrix(p, cfg)
    tbd = (slot, tail, grp, members, mask, freq)
    g = jax.jit(jax.vmap(lambda x: exact_greedy(x, U, Wt[K:], bt[K:], K, tbd)))
    for sc in (1.0, 8.0):
        x = hs * sc
        w, fired = chunked(g, x)
        ex = jnp.concatenate([jnp.argmax(R9.ref_logprobs_full(p, x[i:i + 250], cfg), -1) for i in range(0, 1000, 250)])
        m = float(jnp.mean(w == ex))
        check(f"greedy exact (h x {sc:g})", m == 1.0, f"exact-match {m:.4f} on 1,000 contexts; fallback rate {float(fired.mean()):.3f}")
        res[f"greedy_fallback_h{sc:g}"] = float(fired.mean())

    def two_stage_p(x):
        lt = jax.nn.log_softmax(x @ Wt.T + bt, -1)                                   # [n, K + C]
        zc = jnp.where(mask[None], jnp.einsum("nd,cld->ncl", x, U[members]), -jnp.inf)
        lw = jax.nn.log_softmax(zc, -1)                                              # [n, C, L]
        return jnp.exp(lt[:, slot] + jnp.where(tail[None], lw[:, grp, pos], 0.0))

    worst = 0.0
    for sc in (1.0, 8.0):
        for i in range(0, 1000, 125):
            x = hs[i:i + 125] * sc
            worst = max(worst, float(jnp.max(jnp.abs(two_stage_p(x) - jnp.exp(R9.ref_logprobs_full(p, x, cfg))))))
    check("sampling: two-stage probabilities == reference", worst < 1e-5, f"max |p_two-stage - p_ref| {worst:.2e} over 1,000 contexts (h and 8h) x {V} ids")

    x0 = hs[3] * 8.0
    pref = jnp.exp(R9.ref_logprobs_full(p, x0[None], cfg)[0])
    N = 200_000
    ka, kb, kc = jax.random.split(jax.random.PRNGKey(11), 3)
    lt0 = jax.nn.log_softmax(Wt @ x0 + bt)
    # inverse-CDF sampling: categorical(shape=(N,)) materialises an [N, vocab] Gumbel array (~105 GB at 131k)
    icdf = lambda key, pr: jnp.minimum(jnp.searchsorted(jnp.cumsum(pr), jax.random.uniform(key, (N,)) * jnp.sum(pr)), pr.shape[0] - 1)
    s = icdf(ka, jnp.exp(lt0))
    c = jnp.clip(s - K, 0, C - 1)
    zall = jnp.where(mask, jnp.einsum("d,cld->cl", x0, U[members]), -jnp.inf)  # one context: [C, L] once, then index
    zc = zall[c]
    j = jax.random.categorical(kb, zc, axis=-1)
    w2 = jnp.where(s < K, freq[jnp.minimum(s, K - 1)], members[c, j])
    wd = icdf(kc, pref)
    tv = lambda ww: 0.5 * float(jnp.sum(jnp.abs(jnp.bincount(ww, length=V) / N - pref)))
    t2, td = tv(w2), tv(wd)
    check("sampling: empirical TV", t2 <= 1.5 * td,
          f"TV(two-stage, 200k) {t2:.4f} vs TV(direct, 200k) {td:.4f} (ratio {t2 / td:.3f}); max p {float(pref.max()):.3f}")

elif mode == "trained":
    from round8_heads import load_params
    ck = sys.argv[2]
    cfg = ARMS["H128k384"]
    p = load_params(cfg, ck)
    c32 = replace(cfg, compute_dtype="float32", attn_impl=None)
    p = jax.tree.map(lambda a: a.astype(F32) if a.dtype == jnp.bfloat16 else a, p)
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, freq, count = tb
    K = cfg.head_adapt
    rows = jnp.asarray(val[:: len(val) // 4][:4].astype(np.int32))
    hs = jnp.concatenate([forward(p, rows[i:i + 1, :-1], c32)[0] for i in range(4)]).astype(F32)
    Wt, bt = R9.top_matrix(p, c32, tb)
    U = R9.word_matrix(p, c32)
    w, fired = chunked(jax.jit(jax.vmap(lambda x: exact_greedy(x, U, Wt[K:], bt[K:], K, (slot, tail, grp, members, mask, freq)))), hs)
    ex = jnp.concatenate([jnp.argmax(R9.ref_logprobs_full(p, hs[i:i + 256], c32), -1) for i in range(0, len(hs), 256)])
    fr = float(fired.mean())
    m = float(jnp.mean(w == ex))
    res.update(ckpt=ck, positions=int(len(hs)), fallback_rate=fr, exact_match=m, mean_clusters_expanded=3 * (1 - fr) + cfg.head_C * fr)
    check("trained greedy exact", m == 1.0,
          f"exact-match {m:.4f} on {len(hs)} positions; fallback rate {fr:.4f}; mean clusters expanded {res['mean_clusters_expanded']:.1f}")

say("ALL GATES OK" if not fails else f"GATES FAILED: {fails}")
os.makedirs(OUT, exist_ok=True)
tag = mode if mode != "trained" else "trained_" + sys.argv[2]
open(os.path.join(OUT, f"gates_{tag}.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"checks": res, "fails": fails}, open(os.path.join(OUT, f"gates_{tag}.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
