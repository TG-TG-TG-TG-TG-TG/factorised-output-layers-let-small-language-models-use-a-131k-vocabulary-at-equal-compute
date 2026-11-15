"""Round 10, Part D: row-sparse ("lazy") AdamW for the hashed n-gram tables (train.py --ng_lazy).

For every row a batch touches, the update is exactly optax.adamw's (b1 0.9, b2 0.95, eps 1e-8, decoupled weight
decay wd on the row, bias correction with the global step count); rows the batch does not touch keep their
value and their moments (dense AdamW would keep moving them with their old momentum and decay them).
Gradient clipping stays global: the caller folds these rows' squared norms into the global norm.
"""
import jax
import jax.numpy as jnp

F32 = jnp.float32


def sparse_grads(ids, grows, M, accum):
    """ids [N] int32 row ids, grows [N, d] gradients of the gathered rows (summed over micro-batches, so / accum).
    Returns (u [N] unique ids, filled with M past the last unique id, g [N, d] summed per unique id)."""
    N = ids.shape[0]
    u, inv = jnp.unique(ids, size=N, fill_value=M, return_inverse=True)
    return u, jax.ops.segment_sum(grows, inv.reshape(N), num_segments=N) / accum


def adamw_rows(P, m, v, u, g, cnt, lr, wd, b1=0.9, b2=0.95, eps=1e-8):
    """One AdamW step on rows u of table P (u entries == len(P) are padding and dropped). cnt = this step's
    1-based count. Returns (P, m, v)."""
    c1, c2 = 1 - b1 ** cnt.astype(F32), 1 - b2 ** cnt.astype(F32)
    mr = b1 * m.at[u].get(mode="fill", fill_value=0) + (1 - b1) * g
    vr = b2 * v.at[u].get(mode="fill", fill_value=0) + (1 - b2) * g * g
    pr = P.at[u].get(mode="fill", fill_value=0)
    step = -lr * ((mr / c1) / (jnp.sqrt(vr / c2) + eps) + wd * pr)
    return P.at[u].add(step, mode="drop"), m.at[u].set(mr, mode="drop"), v.at[u].set(vr, mode="drop")
