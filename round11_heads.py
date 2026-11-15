"""Round 11 Part C (2026-11-09): exact re-scoring of the top-M tail clusters on top of round 9's two-level head
(cfg.head == "hier9", cfg.head_rescore = M > 0). With M = 0 nothing here runs and the model is round 9's head.

Top level, per token (h = final normed hidden state, fp32):
  zt = [U[freq] . h ; (ubar_c + R_c) . h + bc]                       (round 9, K singletons + C tail clusters)
  idx = top_M over the C tail-cluster logits (stop-gradient index choice, ties as jax.lax.top_k breaks them)
  for c in idx:  zt[K + c] <- s_c = LSE_{w in c}(u_w . h) + a_c       (no word bias exists; a_c learned, init 0)
  log p(y) = log_softmax(zt)[slot(y)] + (tail y) log_softmax_{w in cluster(y)}(u_w . h)[y]    (within part unchanged)
A valid normalised distribution for any choice of the M clusters; gradients reach U through the LSE (and through the
tied means of the clusters that were not replaced). Train and eval use the same rule; the target's cluster is never
forced into the top M.

Extra forward cost: 2 * M * L * d FLOPs/token (L = largest cluster) for the M x L word logits; top-k is not counted.
Training FLOPs = 3 x forward, as rounds 8-10 count the head.
"""
import jax
import jax.numpy as jnp
import numpy as np

import round9_heads as R9

F32 = jnp.float32
V_REAL = R9.V_REAL


def init_extra(cfg):
    """Head params added for M > 0: a_c (C,), zeros (no RNG draw, so every other tensor equals the M = 0 arm)."""
    return {"ac": jnp.zeros((cfg.head_C,), F32)}


def _chunk(cfg, n, L):
    """Largest power-of-two chunk <= 4096 dividing n with chunk * M * L * d <= 2^27 floats (512 MB fp32 gather)."""
    c = 4096
    while c > 256 and c * cfg.head_rescore * L * cfg.d > 2 ** 27:
        c //= 2
    while n % c:
        c //= 2
    return c


def rescored_top(h, Wt, bt, Uw, ac, members, mask, K, M):
    """Top-level logits [n, K + C] with the M re-scored tail clusters, and their indices [n, M]."""
    zt = h @ Wt.T + bt
    zc = zt[:, K:]
    idx = jax.lax.top_k(jax.lax.stop_gradient(zc), M)[1]                      # [n, M]
    mem = members[idx]                                                        # [n, M, L]
    z = jnp.where(mask[idx], jnp.einsum("nd,nmld->nml", h, Uw[mem]), -jnp.inf)
    s = jax.nn.logsumexp(z, -1) + ac[idx]                                    # [n, M]
    zc = zc.at[jnp.arange(h.shape[0])[:, None], idx].set(s)
    return jnp.concatenate([zt[:, :K], zc], 1), idx


def nll_tokens(h, y, Uw, Wt, bt, ac, slot, tail, grp, members, mask, pos, K, M):
    """Per-token NLL [n] and whether a tail target's cluster was re-scored [n] (False for frequent targets)."""
    zt, idx = rescored_top(h, Wt, bt, Uw, ac, members, mask, K, M)
    lp_top = jnp.take_along_axis(jax.nn.log_softmax(zt), slot[y][:, None], 1)[:, 0]
    c = grp[y]
    zw = jnp.where(mask[c], jnp.einsum("bd,bld->bl", h, Uw[members[c]]), -jnp.inf)
    lp_w = jnp.take_along_axis(jax.nn.log_softmax(zw, -1), pos[y][:, None], 1)[:, 0]
    hit = tail[y] & jnp.any(idx == c[:, None], 1)
    return -(lp_top + jnp.where(tail[y], lp_w, 0.0)), hit


def head_loss_sum(params, h, y, cfg, chunk=None, with_hits=False):
    """Summed NLL over flattened tokens (as round9_heads.head_loss_sum), remat'd chunks. with_hits: also the number
    of tail targets whose cluster was in the top M and the number of tail targets."""
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, _, _ = tb
    Wt, bt = R9.top_matrix(params, cfg, tb)
    Uw = R9.word_matrix(params, cfg)
    ac = params["head"]["ac"]
    K, M = cfg.head_adapt, cfg.head_rescore
    n = h.shape[0]
    chunk = chunk or _chunk(cfg, n, members.shape[1])

    @jax.checkpoint
    def piece(Uw, Wt, bt, ac, hc, yc):
        l, hit = nll_tokens(hc.astype(F32), yc, Uw, Wt, bt, ac, slot, tail, grp, members, mask, pos, K, M)
        return jnp.sum(l), jnp.sum(hit), jnp.sum(tail[yc])

    def body(acc, xs):
        l, a, b = piece(Uw, Wt, bt, ac, *xs)
        return (acc[0] + l, acc[1] + a, acc[2] + b), None

    z = jnp.zeros((), F32)
    (total, hits, ntail), _ = jax.lax.scan(body, (z, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32)),
                                           (h.reshape(-1, chunk, cfg.d), y.reshape(-1, chunk)))
    return (total, hits, ntail) if with_hits else total


def ref_logprobs_full(params, h, cfg, idx=None):
    """Independent full-vocabulary log p(w|h) [n, V_REAL] for the re-scored head: round 9's reference construction
    (dense one-hot class means, word LSEs by segment ops), then the same top-M rule (its own top-M of its own tied-mean
    cluster logits unless idx is given) with s_c = segment LSE + a_c. Also returns the indices used."""
    slot, tail, grp, members, mask, pos, freq, count = (np.asarray(a) for a in R9.tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt))
    C, K, M = cfg.head_C, cfg.head_adapt, cfg.head_rescore
    hp = params["head"]
    Uw = R9.word_matrix(params, cfg)
    z = h @ Uw.T
    S = jax.nn.one_hot(jnp.where(jnp.asarray(tail), jnp.asarray(grp), -1), C, dtype=F32)
    ubar = (S.T @ params["emb"][:V_REAL].astype(F32)) / S.sum(0)[:, None]
    zc = h @ (ubar + hp["R"]).T + hp["bc"]
    seg = jnp.asarray(np.where(tail, grp, C))
    zT = z.T
    mx = jax.lax.stop_gradient(jax.ops.segment_max(zT, seg, num_segments=C + 1))
    lse = (jnp.log(jax.ops.segment_sum(jnp.exp(zT - mx[seg]), seg, num_segments=C + 1)) + mx)[:C].T   # [n, C]
    if idx is None:
        idx = jax.lax.top_k(jax.lax.stop_gradient(zc), M)[1]
    rows = jnp.arange(h.shape[0])[:, None]
    zc = zc.at[rows, idx].set(jnp.take_along_axis(lse, idx, 1) + hp["ac"][idx])
    top = jnp.concatenate([z[:, freq], zc], 1)
    lst = jax.nn.log_softmax(top, -1)
    within = jnp.where(jnp.asarray(tail)[None, :], z - jnp.concatenate([lse, jnp.zeros((h.shape[0], 1))], 1)[:, jnp.asarray(seg)], 0.0)
    return lst[:, jnp.asarray(slot)] + within, idx


def head_flops_fwd(cfg):
    """round 9's head forward FLOPs plus 2 * M * L * d for the re-scored clusters' word logits."""
    L = R9.tables_np(cfg.head_classes, cfg.head_C, cfg.head_adapt)[3].shape[1]
    return R9.head_flops_fwd(cfg) + 2 * cfg.head_rescore * L * cfg.d
