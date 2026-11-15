"""Decoder-only transformer used for the output-layer study, with its output heads.

Params are a plain pytree instead of Flax modules. Blocks are described by Config:
  arch="unique"  every block applied once (all models in the paper)
  arch="pairs"   1, 1', 2, 2', ...  (each block applied twice in a row)
  arch="cycle"   1..P, then 1'..P', n_loops times (looped body; round 10 part C)
A primed (repeated) use shares the block's weights and owns its RMSNorm gains (and alpha) if per_copy.
"""
import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32


@dataclass(frozen=True)
class Config:
    vocab: int = 50304
    d: int = 384
    f: int = 1536
    H: int = 6
    n_unique: int = 6
    arch: str = "unique"       # "unique" | "pairs" | "cycle"
    per_copy: bool = True      # a repeated use owns its RMSNorm gains (and alpha)
    init: str = "neutral"      # "neutral" | "gpt" (vanilla GPT-2 recipe)
    attn_impl: str | None = None  # passed to jax.nn.dot_product_attention
    compute_dtype: str = "bfloat16"
    remat: bool = False
    # per-neuron FFN gates: gelu((x W_in) * gam + beta), gam = 1, beta = 0 at init
    gate: bool = False
    # round 8 (2026-11-06): output-head variants (round8_heads.py); "" = the flat tied head
    head: str = ""             # "" | "hier" (two-level) | "cb" (class-bias)
    head_C: int = 1024
    head_classes: str = "kmeans"  # "kmeans" | "random" (runs/round8/classes_*1024.npy)
    head_wide: int = 0         # H2w: width of h' = GELU(W_up h) with an untied U'
    # round 9 (2026-11-07): head == "hier9" (round9_heads.py)
    head_adapt: int = 0        # K frequent tokens as top-level singletons (0 = plain two-level)
    head_tm: bool = False      # class logits (ubar_c + R_c) . h, ubar_c = mean of the class's tied U rows
    head_untied: bool = False  # word level uses an untied Uo (V x d); input embedding unchanged
    # round 10 (2026-11-07)
    n_loops: int = 2           # arch "cycle": the P unique blocks are applied n_loops times in cycle order
    ngram: int = 0             # hashed n-gram input tables: 0 none, 2 bigram (ng2), 3 bigram + trigram (ng2, ng3)
    ngram_M: int = 2 ** 20     # rows per n-gram table
    # round 11 (2026-11-09): head "hier9" with the top-M tail clusters re-scored exactly (round11_heads.py); 0 = off
    head_rescore: int = 0
    # round 13 (2026-11-12): head "adasm" computes each tail band only for the tokens whose target is in it (round13_heads)
    head_sparse: bool = False
    # round 13 (2026-11-13): head "adasm" tail cutoffs (shortlist end, band ends); () = round13_heads.ADASM_CUTS
    head_cuts: tuple = ()
    # round 13 (2026-11-13, experiment 5): head "adasm" tail bands at full width d with no projection (logits h . W_b[j],
    # W_b untied, n_b x d); False = the d / 4^b low-rank bands of experiments 3-4
    head_full: bool = False
    # round 11: training corpus / tokenizer for data.Batches and data.val_rows ("" = FineWeb-Edu GPT-2; "bpe8192",
    # "bpe4096" = the same documents re-tokenised; "bios20000" etc. = round11_bios synthetic biographies)
    data: str = ""

    @property
    def n_eff(self):
        if self.arch == "cycle":
            return self.n_loops * self.n_unique
        assert self.n_loops == 2, "n_loops is only for arch='cycle'"
        return self.n_unique if self.arch == "unique" else 2 * self.n_unique

    @property
    def n_copies(self):
        return self.n_eff if (self.per_copy or self.arch == "unique") else self.n_unique

    @property
    def learn_alpha(self):
        return self.init == "neutral"

    @property
    def cdt(self):
        return jnp.dtype(self.compute_dtype)


def schedule(cfg):
    """(unique block, use, per-copy index) for every effective block, in order."""
    P = cfg.n_unique
    if cfg.arch == "unique":
        return [(i, 0, i) for i in range(P)]
    if cfg.arch == "pairs":
        out = []
        for i in range(P):
            out += [(i, 0, 2 * i if cfg.per_copy else i), (i, 1, 2 * i + 1 if cfg.per_copy else i)]
        return out
    if cfg.arch == "cycle":  # round 10: n_loops passes (use = 1 for every pass after the first)
        return [(i, min(r, 1), r * P + i if cfg.per_copy else i) for r in range(cfg.n_loops) for i in range(P)]
    raise ValueError(cfg.arch)


SHARED = ("Wq", "Wk", "Wv", "Wo", "Wup", "Wdown")


def init_params(cfg, key):
    d, f, H = cfg.d, cfg.f, cfg.H
    h = d // H
    keys = iter(jax.random.split(key, 32 * cfg.n_unique + 8))

    def nrm(shape, std):
        return std * jax.random.normal(next(keys), shape, F32)

    if cfg.init == "neutral":
        # one std for both projections of a pair; depth scaling lives in alpha
        s_attn_in = s_attn_out = 1.0 / math.sqrt(d)
        s_ffn_in = s_ffn_out = (d * f) ** -0.25
        s_emb = d ** -0.5
    else:
        s_attn_in = s_ffn_in = s_emb = 0.02
        s_attn_out = s_ffn_out = 0.02 / math.sqrt(2 * cfg.n_eff)

    def attn():
        return {"Wq": nrm((d, H, h), s_attn_in), "Wk": nrm((d, H, h), s_attn_in),
                "Wv": nrm((d, H, h), s_attn_in), "Wo": nrm((H, h, d), s_attn_out)}

    blocks = []
    for _ in range(cfg.n_unique):
        b = attn()
        b["Wup"] = nrm((d, f), s_ffn_in)
        b["Wdown"] = nrm((f, d), s_ffn_out)
        blocks.append(b)

    copies = []
    for _ in range(cfg.n_copies):
        c = {"ln_attn": jnp.ones((d,), F32), "ln_ffn": jnp.ones((d,), F32)}
        if cfg.learn_alpha:
            a0 = 1.0 / math.sqrt(2 * cfg.n_eff)
            c["a_attn"] = jnp.full((d,), a0, F32)
            c["a_ffn"] = jnp.full((d,), a0, F32)
        if cfg.gate:
            c["gam"], c["beta"] = jnp.ones((f,), F32), jnp.zeros((f,), F32)
        copies.append(c)

    out = {"emb": nrm((cfg.vocab, d), s_emb), "blocks": blocks, "copies": copies,
           "ln_out": jnp.ones((d,), F32)}
    if cfg.head:  # own key stream (fold_in), so the tensors above are bit-identical to the flat arm
        if cfg.head == "adasm":  # round 13: adaptive softmax baseline (round13_heads.py)
            from round13_heads import init_head as init_head13
            out["head"] = init_head13(cfg, key)
        elif cfg.head == "hier9":
            from round9_heads import init_head as init_head9
            out["head"] = init_head9(cfg, key)
            if cfg.head_rescore:  # round 11: a_c offsets, zeros (no RNG draw)
                from round11_heads import init_extra
                out["head"].update(init_extra(cfg))
        else:
            from round8_heads import init_head
            out["head"] = init_head(cfg, key)
    if cfg.ngram:  # round 10: zero-init tables (no RNG draws), so the model at init equals the arm without them
        for n in range(2, cfg.ngram + 1):
            out[f"ng{n}"] = jnp.zeros((cfg.ngram_M, d), F32)
    return out


# ---------------------------------------------------------------- round 10: hashed n-gram input embeddings
NG_P = 2147483647                      # 2^31 - 1 (prime)
NG_MULT = {2: (1000003, 998244353), 3: (1000000007, 999999937, 19260817)}  # fixed primes < NG_P
NG_MISSING = 50257                     # reserved id (never a data token) for a previous token outside the document/row
EOT_ID = 50256


@functools.lru_cache(None)
def _ng_tables(vocab):
    """(m * v) mod NG_P for every multiplier m and token id v < vocab, as numpy uint32 (exact, int64 math)."""
    v = np.arange(vocab, dtype=np.int64)
    return {n: tuple(((m * v) % NG_P).astype(np.uint32) for m in ms) for n, ms in NG_MULT.items()}


def ngram_ids(tokens, n, cfg):
    """Row index in ng<n> of every position: ((sum_j m_j * x_{t-n+1+j}) mod NG_P) mod M, computed exactly in
    uint32 (each term < NG_P < 2^31, pairwise sums reduced mod NG_P). A previous token that lies before the row
    start or in an earlier document (before the last EOT preceding t; EOT closes a document) -> NG_MISSING."""
    tb = _ng_tables(cfg.vocab)[n]
    T = tokens.shape[-1]
    eot = (tokens == EOT_ID).astype(jnp.int32)
    doc = jnp.cumsum(eot, -1) - eot                    # document index of each position
    acc = None
    for j in range(n):                                 # j = 0 is the oldest token
        lag = n - 1 - j
        if lag:
            prev = jnp.pad(tokens, ((0, 0), (lag, 0)), constant_values=NG_MISSING)[:, :T]
            pdoc = jnp.pad(doc, ((0, 0), (lag, 0)), constant_values=-1)[:, :T]
            prev = jnp.where(pdoc == doc, prev, NG_MISSING)
        else:
            prev = tokens
        term = jnp.asarray(tb[j])[prev]
        acc = term if acc is None else (acc + term) % jnp.uint32(NG_P)
    return (acc % jnp.uint32(cfg.ngram_M)).astype(jnp.int32)


def embed(params, tokens, cfg):
    x = params["emb"][tokens]
    for n in range(2, cfg.ngram + 1):  # train.py --ng_lazy passes the gathered rows as ng<n>_rows (row gradients)
        r = params.get(f"ng{n}_rows")
        x = x + (r if r is not None else params[f"ng{n}"][ngram_ids(tokens, n, cfg)])
    return x


def decay_mask(params):
    """Weight decay on matrices only: no RMSNorm gains or alpha."""
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)


# ---------------------------------------------------------------- reference blocks

def rmsnorm(x, g, eps=1e-6):
    x32 = x.astype(F32)
    return x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps) * g


def rope(x, pos, base=10000.0):
    # x: (B, T, H, h); rotate-half RoPE
    h = x.shape[-1]
    inv = 1.0 / (base ** (jnp.arange(0, h, 2) / h))
    ang = pos[:, None] * inv[None, :]
    cos, sin = jnp.cos(ang)[None, :, None, :], jnp.sin(ang)[None, :, None, :]
    x1, x2 = x[..., : h // 2], x[..., h // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).astype(x.dtype)


def attention(x, Wq, Wk, Wv, Wo, cfg):
    # Wq, Wk, Wv: (d, H, h)   Wo: (H, h, d)
    pos = jnp.arange(x.shape[1], dtype=F32)
    q = jnp.einsum("btd,dnh->btnh", x, Wq)
    k = jnp.einsum("btd,dnh->btnh", x, Wk)
    v = jnp.einsum("btd,dnh->btnh", x, Wv)
    q, k = rope(q, pos), rope(k, pos)
    o = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=cfg.attn_impl)
    return jnp.einsum("btnh,nhd->btd", o, Wo)


def ffn(x, Wup, Wdown, cfg):
    # Wup: (d, f)  Wdown: (f, d)
    return jnp.einsum("btf,fd->btd", jax.nn.gelu(jnp.einsum("btd,df->btf", x, Wup)), Wdown)


def gated_ffn(x, Win, Wout, c, cfg):
    """gelu((x W_in) * gam + beta) W_out with the per-neuron gates of copy c. The gates act in fp32
    and GELU in the compute dtype, so gam = 1, beta = 0 reproduces ffn() exactly."""
    u = jnp.einsum("btd,df->btf", x, Win)
    g = jax.nn.gelu((u.astype(F32) * c["gam"] + c["beta"]).astype(cfg.cdt))
    return jnp.einsum("btf,fd->btd", g, Wout)


def block(x, b, c, *, use, cfg, sg, blk=None):
    cdt = cfg.cdt
    W = jax.tree.map(lambda w: (jax.lax.stop_gradient(w) if sg else w).astype(cdt), b)
    a_attn = c.get("a_attn", 1.0)
    a_ffn = c.get("a_ffn", 1.0)
    xn = rmsnorm(x, c["ln_attn"]).astype(cdt)
    x = x + a_attn * attention(xn, W["Wq"], W["Wk"], W["Wv"], W["Wo"], cfg).astype(F32)
    xn = rmsnorm(x, c["ln_ffn"]).astype(cdt)
    if cfg.gate:
        y = gated_ffn(xn, W["Wup"], W["Wdown"], c, cfg)
    else:
        y = ffn(xn, W["Wup"], W["Wdown"], cfg)
    x = x + a_ffn * y.astype(F32)
    return x


def forward(params, tokens, cfg, sg_use=None):
    """Final normed hidden states (B, T, d) in fp32. sg_use: stop the shared-weight
    gradient through that use (0 = first, 1 = repeated) for a gradient diagnostic."""
    x = embed(params, tokens, cfg)
    for blk, use, cp in schedule(cfg):
        fn = functools.partial(block, use=use, cfg=cfg, sg=sg_use == use, blk=blk)
        if cfg.remat:
            fn = jax.checkpoint(fn)
        x = fn(x, params["blocks"][blk], params["copies"][cp])
    return rmsnorm(x, params["ln_out"])


def lm_loss(params, batch, cfg, sg_use=None, chunk=4096):
    """Mean next-token loss of rows (B, T+1). Tied head; logits are built in token
    chunks under remat so the full (tokens x 50k) fp32 matrix never exists."""
    inp, tgt = batch[:, :-1], batch[:, 1:]
    h = forward(params, inp, cfg, sg_use).reshape(-1, cfg.d)
    y = tgt.reshape(-1)
    n = h.shape[0]
    if cfg.head:
        if cfg.head == "adasm":  # round 13
            from round13_heads import head_loss_sum as head_loss_sum13
            return head_loss_sum13(params, h, y, cfg) / n
        if cfg.head == "hier9" and cfg.head_rescore:  # round 11
            from round11_heads import head_loss_sum as head_loss_sum11
            return head_loss_sum11(params, h, y, cfg) / n
        if cfg.head == "hier9":
            from round9_heads import head_loss_sum as head_loss_sum9
            return head_loss_sum9(params, h, y, cfg) / n
        from round8_heads import head_loss_sum
        return head_loss_sum(params, h, y, cfg) / n
    if cfg.vocab > 65536:  # round 12 (F128k): 1,024-token chunks so one fp32 logit chunk stays at 0.5 GB
        chunk = min(chunk, 1024)
    chunk = min(chunk, n)
    E = params["emb"].astype(cfg.cdt)

    @jax.checkpoint
    def piece(E, hc, yc):
        logits = jnp.einsum("nd,vd->nv", hc.astype(cfg.cdt), E, preferred_element_type=F32)
        lse = jax.nn.logsumexp(logits, -1)
        return jnp.sum(lse - jnp.take_along_axis(logits, yc[:, None], -1)[:, 0])

    def body(acc, xs):
        return acc + piece(E, *xs), None

    total, _ = jax.lax.scan(body, jnp.zeros((), F32), (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
    return total / n


def token_logprobs(params, inp, tgt, cfg):
    """Per-token log p(tgt | inp) of shape (B, T); used by zero-shot eval."""
    h = forward(params, inp, cfg)
    logits = jnp.einsum("btd,vd->btv", h.astype(cfg.cdt), params["emb"].astype(cfg.cdt),
                        preferred_element_type=F32)
    return jnp.take_along_axis(jax.nn.log_softmax(logits, -1), tgt[..., None], -1)[..., 0]


# ---------------------------------------------------------------- accounting

def count_params(params):
    n = lambda t: sum(x.size for x in jax.tree.leaves(t))
    total = n(params)
    return {"total": total, "non_embedding": total - params["emb"].size, "embedding": params["emb"].size}


def flops_per_token(cfg, T=1024):
    """Training FLOPs per token (6N rule + attention), split into non-embedding and head.
    Every effective block counts, since a reused weight is still a matmul each use."""
    d, f = cfg.d, cfg.f
    attn_proj, ffn_mm = 4 * d * d, 2 * d * f
    blocks = 6 * cfg.n_eff * (attn_proj + ffn_mm)
    attn_scores = 12 * cfg.n_eff * T * d // 2  # causal: half of QK^T and AV
    head = 6 * d * cfg.vocab
    if cfg.head:  # round 8: training = 3x the head forward FLOPs
        if cfg.head == "adasm":  # round 13
            from round13_heads import head_flops_fwd as head_flops_fwd13
            head = 3 * head_flops_fwd13(cfg)
        elif cfg.head == "hier9" and cfg.head_rescore:  # round 11
            from round11_heads import head_flops_fwd as head_flops_fwd11
            head = 3 * head_flops_fwd11(cfg)
        elif cfg.head == "hier9":
            from round9_heads import head_flops_fwd as head_flops_fwd9
            head = 3 * head_flops_fwd9(cfg)
        else:
            from round8_heads import head_flops_fwd
            head = 3 * head_flops_fwd(cfg)
    return {"non_embedding": blocks + attn_scores, "head": head, "total": blocks + attn_scores + head}
