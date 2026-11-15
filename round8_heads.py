"""Round 8 (2026-11-06): LM-head variants wired into model.lm_loss. The math lives in
src/heads/heads.py (copied unchanged from the chat; md5 4930c6f9...); this file only wraps it.

Conventions
- Vocab: the model's emb has 50304 rows (GPT-2 50257 padded to a multiple of 64). The class
  tables cover the 50257 real tokens, and the H2 / CB heads use U = emb[:50257]. The flat B head
  keeps its round-5 softmax over all 50304 rows (47 pad rows that never occur as targets).
- No output bias in B, so b = zeros(50257) (a constant, not a parameter).
- New head params come from jax.random.fold_in(key, HEAD_SALT), so the backbone's key stream,
  and hence its init, is bit-identical to B's for a seed (gate 4).
- H2: training and eval loss = heads.hier_nll (gather form) in fp32, chunked (exact model NLL).
  hier_nll_grouped is not used for training: ragged_dot lowers densely here (see head_loss_sum).
- CB: the chunked/remat cross-entropy of lm_loss, logits = heads.class_bias_logits. The flat
  product h U^T runs in the compute dtype (bf16 in, fp32 out) exactly as in B; the class term in
  fp32. Gate CB checks it against heads.class_bias_logits.
"""
import functools
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src", "heads"))
import heads as H  # noqa: E402

F32 = jnp.float32
V_REAL = 50257
HEAD_SALT = 0x8EAD
CLASS_DIR = os.environ.get("ROUND8_CLASS_DIR", os.path.join(HERE, "runs", "round8"))  # env: dry runs only
CLASS_FILES = {"kmeans": "classes_kmeans1024.npy", "random": "classes_random1024.npy"}


@functools.lru_cache(None)
def _tables_np(name, C):
    cls = np.load(os.path.join(CLASS_DIR, CLASS_FILES[name])).astype(np.int32)
    assert cls.shape == (V_REAL,) and cls.min() >= 0 and cls.max() < C, (cls.shape, cls.min(), cls.max())
    with jax.ensure_compile_time_eval():   # heads.class_tables returns jnp arrays; keep them concrete
        members, mask, pos = (np.asarray(a) for a in H.class_tables(cls, C))
    return cls, members, mask, pos


def tables(name, C):
    """(cls, members, mask, pos) as jnp arrays. Only the numpy tables are cached: caching jnp arrays
    made inside a jit/scan trace leaked a tracer into the next jit (wsd_H2_s0 try 1, 2026-11-06)."""
    return tuple(jnp.asarray(a) for a in _tables_np(name, C))


def init_head(cfg, key):
    k = iter(jax.random.split(jax.random.fold_in(key, HEAD_SALT), 8))
    C = cfg.head_C
    if cfg.head_wide:  # H2w: h' = GELU(W_up h), untied U' (V x w), Vc (w x C)
        w = cfg.head_wide
        return {"Wup": jax.random.normal(next(k), (cfg.d, w), F32) / math.sqrt(cfg.d),
                "Uw": jax.random.normal(next(k), (V_REAL, w), F32) / math.sqrt(w),
                "Vc": jax.random.normal(next(k), (w, C), F32) / math.sqrt(w), "bc": jnp.zeros((C,), F32)}
    std = 1.0 / math.sqrt(cfg.d)  # N(0, 1/d)
    if cfg.head == "hier":
        return {"Vc": std * jax.random.normal(next(k), (cfg.d, C), F32), "bc": jnp.zeros((C,), F32)}
    if cfg.head == "cb":
        return {"Vg": std * jax.random.normal(next(k), (cfg.d, C), F32), "ag": jnp.zeros((C,), F32)}
    raise ValueError(cfg.head)


def head_inputs(params, h, cfg):
    """(h_for_head, U, b, head params) in fp32."""
    hp = params["head"]
    if cfg.head_wide:
        return jax.nn.gelu(h @ hp["Wup"]), hp["Uw"], jnp.zeros((V_REAL,), F32), hp
    return h, params["emb"][:V_REAL].astype(F32), jnp.zeros((V_REAL,), F32), hp


def head_loss_sum(params, h, y, cfg, chunk=4096):
    """Summed NLL over the flattened tokens. h [n, d] fp32 (after the final norm), y [n]."""
    cls, members, mask, pos = tables(cfg.head_classes, cfg.head_C)
    if cfg.head == "hier":
        # heads.hier_nll (gather form), in remat'd 4096-token chunks. Not hier_nll_grouped: on this
        # JAX 0.9 / sm_120 build jax.lax.ragged_dot lowers densely to an [n, C, d] product (n=16384:
        # f32[16384,1024,128], 8.6 GB) -> autotune/OOM crash and ~V*d work. Gates 1-2 show the two
        # agree to 1.9e-6 (NLL) and 1.2e-7 (dL/dU). (2026-11-06)
        n = h.shape[0]
        chunk = min(chunk, n)
        hp = params["head"]

        @jax.checkpoint
        def hpiece(params_head, emb, hc, yc):
            hh, U, b, hq = head_inputs({"head": params_head, "emb": emb}, hc, cfg)
            return jnp.sum(H.hier_nll(hh, yc, U, b, hq["Vc"], hq["bc"], cls, members, mask, pos))

        def hbody(acc, xs):
            return acc + hpiece(hp, params["emb"], *xs), None

        total, _ = jax.lax.scan(hbody, jnp.zeros((), F32), (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
        return total
    assert cfg.head == "cb" and not cfg.head_wide
    hp = params["head"]
    E = params["emb"][:V_REAL].astype(cfg.cdt)
    n = h.shape[0]
    chunk = min(chunk, n)

    @jax.checkpoint
    def piece(E, Vg, ag, hc, yc):
        logits = cb_logits(hc, E, Vg, ag, cls, cfg)
        lse = jax.nn.logsumexp(logits, -1)
        return jnp.sum(lse - jnp.take_along_axis(logits, yc[:, None], -1)[:, 0])

    def body(acc, xs):
        return acc + piece(E, hp["Vg"], hp["ag"], *xs), None

    total, _ = jax.lax.scan(body, jnp.zeros((), F32), (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
    return total


def cb_logits(hc, E, Vg, ag, cls, cfg):
    """heads.class_bias_logits(h, U, 0, Vg, ag, cls) with the h U^T product in B's precision
    (compute dtype in, fp32 out). With cfg.compute_dtype = float32 it is the same expression."""
    flat = jnp.einsum("nd,vd->nv", hc.astype(cfg.cdt), E.astype(cfg.cdt), preferred_element_type=F32)
    return flat + jax.nn.log_sigmoid(hc.astype(F32) @ Vg + ag)[:, cls]


def load_params(cfg, fname, seed=0):
    """Rebuild a pytree saved by train.save_params (~/lm-data/ckpt/<fname>.npz)."""
    from data import DATA_DIR
    from model import init_params
    z = np.load(os.path.join(DATA_DIR, "ckpt", fname + ".npz"))
    tmpl = init_params(cfg, jax.random.PRNGKey(seed))
    flat, tree = jax.tree_util.tree_flatten_with_path(tmpl)
    keys = [jax.tree_util.keystr(k) for k, _ in flat]
    assert set(keys) == set(z.files), (sorted(set(keys) ^ set(z.files)))[:10]
    return jax.tree_util.tree_unflatten(tree, [jnp.asarray(z[k]) for k in keys])


def head_flops_fwd(cfg):
    """Head forward FLOPs per token (multiply-add = 2)."""
    d, C = cfg.d, cfg.head_C
    cap = -(-V_REAL // C)
    if not cfg.head:
        return 2 * cfg.vocab * d
    if cfg.head == "hier":
        if cfg.head_wide:
            w = cfg.head_wide
            return 2 * d * w + 2 * (C + cap) * w
        return 2 * (C + cap) * d
    assert cfg.head == "cb", cfg.head  # round 13: an unknown head must not be counted as CB
    return 2 * V_REAL * d + 2 * C * d
