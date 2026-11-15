"""Round 9 (2026-11-07): two-level head variants, cfg.head == "hier9". Round-8 code paths are untouched
(round8_heads.py handles head == "hier" / "cb").

One general two-level head covers every round-9 arm:
  top level  one softmax over T = K + C logits: [U[freq] . h ; W_c . h + bc]
             K = cfg.head_adapt frequent tokens as singletons (0 for the plain H2 shape),
             C = cfg.head_C clusters of the other tokens;
             W_c = Vc^T (a free C x d matrix, as in round 8) or, if cfg.head_tm, ubar_c + R_c with
             ubar_c = the mean of the current U rows of cluster c (gradients flow into U), R zero-init.
  within     tail target y in cluster c: log-softmax over the cluster's word logits U[w] . h.
             U = the tied embedding emb[:50257], or an untied Uo (V x d) if cfg.head_untied (word level only).
  log p(y) = top[slot(y)] (+ within part for tail y).
No output bias anywhere (as B), so the frequent singletons' "b[freq]" is the constant 0.

Tables, built once in numpy (never cache jnp arrays: round-8 tracer leak):
  slot [V]    top-level index (frequent: 0..K-1 by count rank; tail: K + cluster)
  tail [V]    bool, has a within part
  grp  [V]    cluster of a tail token; frequent tokens point at cluster 0 / position 0, a valid slot,
              and their within part is zeroed with jnp.where (finite values, so no NaN gradients)
  members [C, L], mask [C, L], pos [V]   as heads.class_tables, over the tail tokens only
  freq [K]    token ids of the singletons; count [C] real cluster sizes (for ubar)
Class files: K = 0 -> runs/round8/classes_<name>1024.npy; K > 0 -> runs/round9/adapt_freq<K>.npy (token ids,
count order) and runs/round9/adapt_tail<K>_<C>.npy (cluster per token, -1 for frequent ones).

Training / eval loss: hier9_nll in fp32, remat'd 4096-token chunks (round 8's gather + remat-chunk path).
"""
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F32 = jnp.float32
V_REAL = 50257
HEAD_SALT = 0x8EAD  # same fold_in salt as round 8: the backbone key stream is untouched
DIR8 = os.environ.get("ROUND8_CLASS_DIR", os.path.join(HERE, "runs", "round8"))
DIR9 = os.environ.get("ROUND9_CLASS_DIR", os.path.join(HERE, "runs", "round9"))
DIR10 = os.path.join(HERE, "runs", "round10")
DIR12 = os.path.join(HERE, "runs", "round12")


def vreal(cfg):
    """Real (non-pad) ids: GPT-2's 50,257 inside the 50,304-row embedding; a BPE vocabulary's own size otherwise
    (round 12: bpe131072). Every pre-round-12 arm has vocab 50,304 -> 50,257, unchanged."""
    return V_REAL if cfg.vocab == 50304 else cfg.vocab


@functools.lru_cache(None)
def tables_np(classes, C, K, V=V_REAL):
    """numpy tables (slot, tail, grp, members, mask, pos, freq, count). V != 50,257: round-12 class files
    runs/round12/v<V>_adapt_freq<K>.npy and v<V>_adapt_tail<K>_<C>.npy."""
    if V != V_REAL:
        freq = np.load(os.path.join(DIR12, f"v{V}_adapt_freq{K}.npy")).astype(np.int64)
        cls = np.load(os.path.join(DIR12, f"v{V}_adapt_tail{K}_{C}.npy")).astype(np.int64)
        assert len(freq) == K and np.all(cls[freq] == -1) and (cls == -1).sum() == K
    elif classes == "adapt13":  # round 13: the same split rebuilt from a cheap, charged d=128 proxy (round13_classes.py)
        freq = np.load(os.path.join(HERE, "runs", "round13", f"adapt_freq{K}.npy")).astype(np.int64)
        cls = np.load(os.path.join(HERE, "runs", "round13", f"adapt_tail{K}_{C}.npy")).astype(np.int64)
        assert len(freq) == K and np.all(cls[freq] == -1) and (cls == -1).sum() == K
    elif K == 0:
        cls = np.load(os.path.join(DIR8, f"classes_{classes}{C}.npy")).astype(np.int64)
        freq = np.zeros(0, np.int64)
    else:  # round 10: splits not built in round 9 (K = 8192 or C' = 4096) live in runs/round10
        dk = DIR9 if os.path.exists(os.path.join(DIR9, f"adapt_tail{K}_{C}.npy")) else DIR10
        freq = np.load(os.path.join(dk, f"adapt_freq{K}.npy")).astype(np.int64)
        cls = np.load(os.path.join(dk, f"adapt_tail{K}_{C}.npy")).astype(np.int64)
        assert len(freq) == K and np.all(cls[freq] == -1) and (cls == -1).sum() == K
    assert cls.shape == (V,) and cls.max() < C
    tail = cls >= 0
    count = np.bincount(cls[tail], minlength=C)
    assert count.min() > 0, "empty cluster"
    L = int(count.max())
    members = np.zeros((C, L), np.int32)
    mask = np.zeros((C, L), bool)
    pos = np.zeros(V, np.int32)
    fill = np.zeros(C, np.int64)
    for w in np.flatnonzero(tail):
        c = cls[w]
        members[c, fill[c]], mask[c, fill[c]], pos[w] = w, True, fill[c]
        fill[c] += 1
    slot = np.empty(V, np.int32)
    slot[freq] = np.arange(K)
    slot[tail] = K + cls[tail]
    grp = np.where(tail, cls, 0).astype(np.int32)
    return (slot, tail, grp, members, mask, pos, freq.astype(np.int32), count.astype(np.float32))


def tables(cfg):
    return tuple(jnp.asarray(a) for a in tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt, vreal(cfg)))


def init_head(cfg, key):
    k = iter(jax.random.split(jax.random.fold_in(key, HEAD_SALT), 8))
    C, d = cfg.head_C, cfg.d
    out = {"bc": jnp.zeros((C,), F32)}
    vc = (1.0 / math.sqrt(d)) * jax.random.normal(next(k), (d, C), F32)  # always drawn: same Vc as round 8
    if cfg.head_tm:
        out["R"] = jnp.zeros((C, d), F32)
    else:
        out["Vc"] = vc
    if cfg.head_untied:  # like the embedding (neutral init: N(0, 1/d))
        out["Uo"] = (d ** -0.5) * jax.random.normal(next(k), (vreal(cfg), d), F32)
    return out


def word_matrix(params, cfg):
    return params["head"]["Uo"] if cfg.head_untied else params["emb"][:vreal(cfg)].astype(F32)


def class_matrix(params, cfg, grp, tail, count):
    """[C, d] class-logit weights: Vc^T, or ubar + R (ubar from the current tied U, gradients flow)."""
    hp = params["head"]
    if not cfg.head_tm:
        return hp["Vc"].T
    U = params["emb"][:vreal(cfg)].astype(F32)  # the tied embedding (the tied-mean arms are never untied)
    s = jax.ops.segment_sum(jnp.where(tail[:, None], U, 0.0), grp, num_segments=cfg.head_C)
    return s / count[:, None] + hp["R"]


def top_matrix(params, cfg, tb):
    slot, tail, grp, members, mask, pos, freq, count = tb
    Wc = class_matrix(params, cfg, grp, tail, count)
    if cfg.head_adapt:
        Wf = params["emb"][freq].astype(F32) if not cfg.head_untied else params["head"]["Uo"][freq]
        return jnp.concatenate([Wf, Wc], 0), jnp.concatenate([jnp.zeros((cfg.head_adapt,), F32), params["head"]["bc"]])
    return Wc, params["head"]["bc"]


def hier9_nll(h, y, Uw, Wt, bt, slot, tail, grp, members, mask, pos):
    """Per-token NLL: top-level softmax over T logits + (tail targets only) the within-cluster softmax."""
    lp_top = jnp.take_along_axis(jax.nn.log_softmax(h @ Wt.T + bt), slot[y][:, None], 1)[:, 0]
    c = grp[y]
    mem = members[c]
    zc = jnp.where(mask[c], jnp.einsum("bd,bld->bl", h, Uw[mem]), -jnp.inf)
    lp_w = jnp.take_along_axis(jax.nn.log_softmax(zc, -1), pos[y][:, None], 1)[:, 0]
    return -(lp_top + jnp.where(tail[y], lp_w, 0.0))


def head_loss_sum(params, h, y, cfg, chunk=4096):
    """Summed NLL over the flattened tokens. h [n, d] fp32 (after the final norm), y [n]."""
    tb = tables(cfg)
    slot, tail, grp, members, mask, pos, _, _ = tb
    Wt, bt = top_matrix(params, cfg, tb)   # once per call, outside the chunk scan (gradients flow)
    Uw = word_matrix(params, cfg)
    n = h.shape[0]
    chunk = min(chunk, n)

    @jax.checkpoint
    def piece(Uw, Wt, bt, hc, yc):
        return jnp.sum(hier9_nll(hc.astype(F32), yc, Uw, Wt, bt, slot, tail, grp, members, mask, pos))

    def body(acc, xs):
        return acc + piece(Uw, Wt, bt, *xs), None

    total, _ = jax.lax.scan(body, jnp.zeros((), F32), (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
    return total


# ---------------- independent full-vocabulary reference (gates, analyses) ----------------
def ref_logprobs_full(params, h, cfg):
    """log p(w|h) [n, V_REAL] built a different way from the fast path: full word logits z = h U^T, class
    means through a dense one-hot matmul, frequent logits as columns of z, within-cluster LSE by segment ops."""
    slot, tail, grp, members, mask, pos, freq, count = (np.asarray(a) for a in tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt, vreal(cfg)))
    C, K = cfg.head_C, cfg.head_adapt
    hp = params["head"]
    Uw = word_matrix(params, cfg)
    z = h @ Uw.T                                                        # [n, V]
    if cfg.head_tm:
        S = jax.nn.one_hot(jnp.where(jnp.asarray(tail), jnp.asarray(grp), -1), C, dtype=F32)  # [V, C], frequent rows 0
        ubar = (S.T @ params["emb"][:vreal(cfg)].astype(F32)) / S.sum(0)[:, None]
        zc = h @ (ubar + hp["R"]).T + hp["bc"]
    else:
        zc = h @ hp["Vc"] + hp["bc"]
    top = jnp.concatenate([z[:, freq], zc], 1) if K else zc           # [n, K + C]
    lst = jax.nn.log_softmax(top, -1)
    seg = jnp.asarray(np.where(tail, grp, C))                          # frequent tokens -> dummy segment C
    zT = z.T
    mx = jax.lax.stop_gradient(jax.ops.segment_max(zT, seg, num_segments=C + 1))
    lse = jnp.log(jax.ops.segment_sum(jnp.exp(zT - mx[seg]), seg, num_segments=C + 1)) + mx  # [C+1, n]
    within = jnp.where(jnp.asarray(tail)[None, :], (zT - lse[seg]).T, 0.0)
    return lst[:, jnp.asarray(slot)] + within


def head_flops_fwd(cfg):
    """Head forward FLOPs per token (multiply-add = 2): the T = K + C top-level logits plus one cluster of
    L words. tied-mean arms add the ubar segment sum, V*d adds once per 16,384-token micro-batch."""
    slot, tail, grp, members, mask, pos, freq, count = tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt, vreal(cfg))
    L = members.shape[1]
    f = 2 * (cfg.head_adapt + cfg.head_C + L) * cfg.d
    if cfg.head_tm:
        f += vreal(cfg) * cfg.d / 16384
    return f
