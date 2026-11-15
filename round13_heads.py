"""Round 13, experiment 3 (2026-11-12): adaptive softmax baseline (Grave et al. 2017), cfg.head == "adasm".
Round 8-12 code paths are untouched; model.py / round11_eval.py dispatch here only for head "adasm".

The tokens are ranked by training-set target count (stable sort, ties by id; the same counts files as the two-level
head's frequent list: runs/round9/train_counts.npy for GPT-2, runs/round12/v131072_train_counts.npy for bpe131072).
  head (shortlist)  ranks [0, K) with K = cfg.head_adapt = 8,192: singletons, logits U[w] . h with U = the tied
                    embedding (identical to the two-level head's frequent part; asserted equal to its frequent list),
                    plus one logit per tail band  Vb[:, b] . h + bb[b]
  tail band b       ranks [c_b, c_{b+1}) with cutoffs ADASM_CUTS = (8,192, 24,576, V): its own low-rank output layer,
                    logits (h P_b) . W_b[j], P_b = d x d_b, W_b = n_b x d_b, d_b = d / 4^b (d/4 = 96, d/16 = 24 at d=384);
                    untied from the embedding, as in Grave et al. Per-arm cutoffs: cfg.head_cuts (() = ADASM_CUTS);
                    2026-11-13 adds AS128k384c49 = cutoffs (8,192, 49,152) (round 13 experiment 4).
                    2026-11-13 (experiment 5) adds cfg.head_full: every band at full width d with NO projection,
                    logits h . W_b[j], W_b = n_b x d, still untied (arm AS128k384c49w). Only the width changes.
  log p(y) = log_softmax(head)[slot(y)] + (tail y) log_softmax(band(y))[pos(y)]
No word bias anywhere (as B and the two-level head).

Init (one rule: every initial logit has std ~1, as the flat head's h . u with u ~ N(0, 1/d) and |h| ~ sqrt(d)):
Vb ~ N(0, 1/d) (the two-level head's Vc rule), bb = 0, P_b ~ N(0, 1/d), W_b ~ N(0, 1/d_b). Drawn from
fold_in(key, ADASM_SALT), so the backbone tensors are bit-identical to B's for a seed. Weight decay: model.decay_mask
(every >= 2-D tensor: Vb, P_b, W_b; not bb), as for the two-level head's Vc / R.

FLOPs (multiply-add = 2), forward per token, charged as Grave et al.'s training algorithm runs: the head for every token,
band b only for the targets that fall in it:  2 (K + nb) d + sum_b p_b 2 (d d_b + n_b d_b),  p_b = the band's share of
training targets (from the counts file). Training = 3 x forward, as rounds 8-12 count every head. The dense cost (every
band for every token, what this implementation actually computes) is head_flops_fwd(cfg, dense=True), reported only.
"""
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F32 = jnp.float32
ADASM_SALT = 0xADA5
ADASM_CUTS = (8192, 24576)  # shortlist end, first-band end; the last band runs to V (pinned 2026-11-12, not swept)
ADASM_DIV = 4


def vreal(cfg):
    return 50257 if cfg.vocab == 50304 else cfg.vocab


def counts_file(V):
    return (os.path.join(HERE, "runs", "round9", "train_counts.npy") if V == 50257
            else os.path.join(HERE, "runs", "round12", f"v{V}_train_counts.npy"))


def freq_file(V, K):
    return (os.path.join(HERE, "runs", "round10", f"adapt_freq{K}.npy") if V == 50257
            else os.path.join(HERE, "runs", "round12", f"v{V}_adapt_freq{K}.npy"))


def cuts(cfg):
    """the arm's cutoffs (cfg.head_cuts; () = ADASM_CUTS, the experiment-3 setting)"""
    return tuple(cfg.head_cuts) or ADASM_CUTS


def tabs_np(cfg):
    """tables_np for this arm (cutoffs and width from the config)"""
    return tables_np(vreal(cfg), cfg.head_adapt, cfg.d, cuts(cfg), bool(getattr(cfg, "head_full", False)))


@functools.lru_cache(None)
def tables_np(V, K, d, cuts=ADASM_CUTS, full=False):
    """rank-ordered ids, band edges, per-token slot / band / pos, band dims and band target shares (cache key includes
    the cutoffs, so two configurations in one process never share tables)."""
    assert K == cuts[0] and all(a < b for a, b in zip(cuts, cuts[1:] + (V,)))
    counts = np.load(counts_file(V))[:V].astype(np.float64)
    order = np.argsort(-counts, kind="stable")
    assert np.array_equal(order[:K], np.load(freq_file(V, K))), "shortlist differs from the two-level head's frequent list"
    edges = (0,) + tuple(cuts) + (V,)
    nb = len(edges) - 2
    rank = np.empty(V, np.int64)
    rank[order] = np.arange(V)
    band = np.searchsorted(np.asarray(edges[1:]), rank, side="right")   # 0 = shortlist, b >= 1 = tail band b
    slot = np.where(band == 0, rank, K + band - 1).astype(np.int32)
    pos = (rank - np.asarray(edges)[band]).astype(np.int32)
    dims = tuple(d if full else d // ADASM_DIV ** b for b in range(1, nb + 1))
    p = counts[order] / counts.sum()
    share = tuple(float(p[edges[b]:edges[b + 1]].sum()) for b in range(1, nb + 1))
    sizes = tuple(edges[b + 1] - edges[b] for b in range(1, nb + 1))
    return dict(order=order.astype(np.int32), edges=edges, nb=nb, slot=slot, band=band.astype(np.int32), pos=pos,
                dims=dims, share=share, sizes=sizes)


def init_head(cfg, key):
    t = tabs_np(cfg)
    k = iter(jax.random.split(jax.random.fold_in(key, ADASM_SALT), 1 + 2 * t["nb"]))
    d, nb = cfg.d, t["nb"]
    out = {"Vb": d ** -0.5 * jax.random.normal(next(k), (d, nb), F32), "bb": jnp.zeros((nb,), F32)}
    for b, (db, n) in enumerate(zip(t["dims"], t["sizes"]), 1):
        kp, kw = next(k), next(k)  # drawn in the same order either way
        if not t_full(cfg):
            out[f"P{b}"] = d ** -0.5 * jax.random.normal(kp, (d, db), F32)
        out[f"W{b}"] = db ** -0.5 * jax.random.normal(kw, (n, db), F32)
    return out


def t_full(cfg):
    return bool(getattr(cfg, "head_full", False))


def _tabs(cfg):
    t = tabs_np(cfg)
    return t, jnp.asarray(t["order"][:cfg.head_adapt]), jnp.asarray(t["slot"]), jnp.asarray(t["band"]), jnp.asarray(t["pos"])


def _band_dense(h, m, py, P, W):
    """band log-prob of every token's target (0 where m is False), computed for all n tokens"""
    z = (h if P is None else h @ P) @ W.T                                      # [n, n_b]
    lw = jnp.take_along_axis(jax.nn.log_softmax(z, -1), jnp.where(m, py, 0)[:, None], 1)[:, 0]
    return jnp.where(m, lw, 0.0)


def _band_sparse(h, m, py, P, W, cap):
    """the same, computed only for the (at most cap) tokens with m True: gather, score, scatter back"""
    n = h.shape[0]
    idx = jnp.nonzero(m, size=cap, fill_value=0)[0]                           # positions of the band's tokens
    valid = jnp.arange(cap) < jnp.sum(m)
    z = (h[idx] if P is None else h[idx] @ P) @ W.T                            # [cap, n_b]
    lw = jnp.take_along_axis(jax.nn.log_softmax(z, -1), py[idx][:, None], 1)[:, 0]
    return jnp.zeros((n,), F32).at[idx].add(jnp.where(valid, lw, 0.0))


def band_caps(cfg, n):
    """sparse capacity per band for an n-token chunk: 3x the band's expected count + 64, rounded up to 128 (at most n);
    a chunk with more band tokens than that takes the dense path (lax.cond), so the result is exact for any chunk"""
    t = tabs_np(cfg)
    return tuple(min(n, -(-int(3 * s * n + 64) // 128) * 128) for s in t["share"])


def nll_tokens(h, y, params, cfg, caps=None):
    """Per-token NLL [n] (fast path: table lookups). h [n, d] fp32. cfg.head_sparse: each band only for its tokens."""
    t, freq, slot, band, pos = _tabs(cfg)
    hp = params["head"]
    Wf = params["emb"][freq].astype(F32)
    top = jnp.concatenate([h @ Wf.T, h @ hp["Vb"] + hp["bb"]], 1)            # [n, K + nb]
    lp = jnp.take_along_axis(jax.nn.log_softmax(top, -1), slot[y][:, None], 1)[:, 0]
    by, py = band[y], pos[y]
    caps = caps or (band_caps(cfg, h.shape[0]) if cfg.head_sparse else None)
    for b in range(1, t["nb"] + 1):
        m, P, W = by == b, hp.get(f"P{b}"), hp[f"W{b}"]
        if caps is None or caps[b - 1] >= h.shape[0]:
            lp = lp + _band_dense(h, m, py, P, W)
        else:
            cap = caps[b - 1]
            lp = lp + jax.lax.cond(jnp.sum(m) <= cap, lambda a: _band_sparse(*a, cap), lambda a: _band_dense(*a),
                                   (h, m, py, P, W))
    return -lp


def _chunk(cfg):
    return 1024 if vreal(cfg) > 65536 else 4096


def head_loss_sum(params, h, y, cfg, chunk=None):
    """Summed NLL over the flattened tokens, remat'd chunks (as round9_heads.head_loss_sum)."""
    n = h.shape[0]
    chunk = min(chunk or _chunk(cfg), n)

    @jax.checkpoint
    def piece(p, hc, yc):
        return jnp.sum(nll_tokens(hc.astype(F32), yc, p, cfg))

    hpar = {"emb": params["emb"], "head": params["head"]}

    def body(acc, xs):
        return acc + piece(hpar, *xs), None

    total, _ = jax.lax.scan(body, jnp.zeros((), F32), (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
    return total


def ref_logprobs_full(params, h, cfg):
    """Independent full-vocabulary log p(w|h) [n, V] built another way: the distribution is assembled in rank order
    (shortlist columns, then each band's column block = its band log-prob + its within-band log-softmax via an
    explicit logsumexp), then permuted back to token ids with the inverse permutation computed here by argsort."""
    t = tabs_np(cfg)
    hp = params["head"]
    order = t["order"]
    K = cfg.head_adapt
    zf = jnp.einsum("nd,kd->nk", h, params["emb"][jnp.asarray(order[:K])].astype(F32))
    zb = jnp.einsum("nd,db->nb", h, hp["Vb"]) + hp["bb"]
    Z = jax.nn.logsumexp(jnp.concatenate([zf, zb], 1), -1, keepdims=True)
    cols = [zf - Z]
    for b in range(1, t["nb"] + 1):
        z = (jnp.einsum("nd,wd->nw", h, hp[f"W{b}"]) if t_full(cfg)
             else jnp.einsum("nd,de,we->nw", h, hp[f"P{b}"], hp[f"W{b}"]))
        cols.append((zb[:, b - 1:b] - Z) + (z - jax.nn.logsumexp(z, -1, keepdims=True)))
    by_rank = jnp.concatenate(cols, 1)                                         # [n, V] in rank order
    inv = np.argsort(order)                                                    # token id -> rank
    return by_rank[:, jnp.asarray(inv)]


def head_flops_fwd(cfg, dense=False):
    """Forward FLOPs per token: the head for every token; band b weighted by its target share (dense: weight 1)."""
    t = tabs_np(cfg)
    d = cfg.d
    f = 2 * (cfg.head_adapt + t["nb"]) * d
    for db, n, s in zip(t["dims"], t["sizes"], t["share"]):
        f += (1.0 if dense else s) * 2 * ((0 if t_full(cfg) else d * db) + n * db)  # full: no projection
    return f


def n_head_params(cfg):
    t = tabs_np(cfg)
    return cfg.d * t["nb"] + t["nb"] + sum(((0 if t_full(cfg) else cfg.d) + n) * db for db, n in zip(t["dims"], t["sizes"]))
