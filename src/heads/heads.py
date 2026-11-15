"""Output-head variants for small LMs (pure JAX, drop-in).

Shapes: h [B, d] final hidden state, U [V, d] output embeddings (tied or not), b [V] bias,
cls [V] int32 class id of every token (from balanced_kmeans on a trained U).

  flat_logits          standard head: rank(log-probs) <= d + 1
  class_bias_logits    + g_c(h) = log sigmoid(h Vg + a)[c(w)]: rank <= d + C + 1, cost +C*d (+V adds)
  hier_*               two-level head p(c|h) p(w|c,h); rank <= d + C; train/sample cost (C + |c|)*d
  grad_retention       share of the logit-space gradient a head can pass back (diagnostic)
"""
import numpy as np
import jax
import jax.numpy as jnp


def flat_logits(h, U, b):
    return h @ U.T + b


def class_bias_logits(h, U, b, Vg, ag, cls):
    """Vg [d, C], ag [C]. Must stay nonlinear in h: a linear g adds no rank."""
    return h @ U.T + b + jax.nn.log_sigmoid(h @ Vg + ag)[:, cls]


# ---------------- two-level (hierarchical) head ----------------
def class_tables(cls, C):
    """members [C, L] padded token ids, mask [C, L], pos [V] position of each token inside its class."""
    cls = np.asarray(cls)
    L = int(np.bincount(cls, minlength=C).max())
    members = np.zeros((C, L), np.int32)
    mask = np.zeros((C, L), bool)
    pos = np.zeros(len(cls), np.int32)
    fill = np.zeros(C, np.int64)
    for w, c in enumerate(cls):
        members[c, fill[c]] = w
        mask[c, fill[c]] = True
        pos[w] = fill[c]
        fill[c] += 1
    return jnp.asarray(members), jnp.asarray(mask), jnp.asarray(pos)


def hier_logprobs_full(h, U, b, Vc, bc, cls, C):
    """Full-vocabulary log p(w|h) [B, V] (for eval / checking). Vc [d, C], bc [C]."""
    z = (h @ U.T + b).T                                                    # [V, B]
    mx = jax.lax.stop_gradient(jax.ops.segment_max(z, cls, num_segments=C))
    lse = jnp.log(jax.ops.segment_sum(jnp.exp(z - mx[cls]), cls, num_segments=C)) + mx
    return jax.nn.log_softmax(h @ Vc + bc)[:, cls] + (z - lse[cls]).T


def hier_nll(h, y, U, b, Vc, bc, cls, members, mask, pos):
    """Per-token NLL touching only C class logits + the target's class: cost (C + L)*d, not V*d."""
    c = cls[y]                                                             # [B]
    lp_c = jnp.take_along_axis(jax.nn.log_softmax(h @ Vc + bc), c[:, None], 1)[:, 0]
    mem = members[c]                                                       # [B, L]
    zc = jnp.einsum("bd,bld->bl", h, U[mem]) + b[mem]
    zc = jnp.where(mask[c], zc, -jnp.inf)
    lp_w = jnp.take_along_axis(jax.nn.log_softmax(zc, -1), pos[y][:, None], 1)[:, 0]
    return -(lp_c + lp_w)


def hier_sample(key, h, U, b, Vc, bc, members, mask):
    """Exact ancestral sample from the two-level model (temperature 1). Cost (C + L)*d per token."""
    k1, k2 = jax.random.split(key)
    c = jax.random.categorical(k1, h @ Vc + bc)
    mem = members[c]
    zc = jnp.where(mask[c], jnp.einsum("bd,bld->bl", h, U[mem]) + b[mem], -jnp.inf)
    j = jax.random.categorical(k2, zc)
    return jnp.take_along_axis(mem, j[:, None], 1)[:, 0]


def hier_greedy(h1, U, b, Vc, bc, members, mask):
    """Exact argmax for ONE context h1 [d], expanding classes by p(c|h) and stopping when
    p(best word) >= p(next class)  (valid because p(w|h) <= p(c(w)|h)). Host loop; returns (token, classes_expanded)."""
    lpc = np.asarray(jax.nn.log_softmax(h1 @ Vc + bc))
    order = np.argsort(-lpc)
    best_lp, best_w = -np.inf, -1
    for n, c in enumerate(order):
        if lpc[c] <= best_lp:
            return best_w, n
        mem = np.asarray(members[c])[np.asarray(mask[c])]
        z = np.asarray(U[mem] @ h1 + b[mem])
        lw = z - np.logaddexp.reduce(z)
        i = int(np.argmax(lw))
        if lpc[c] + lw[i] > best_lp:
            best_lp, best_w = lpc[c] + lw[i], int(mem[i])
    return best_w, len(order)


# ---------------- classes ----------------
def balanced_kmeans(U, C, iters=15, seed=0):
    """Equal-size clusters (capacity ceil(V/C)) of row-normalized U; equal sizes keep hier_* padding small."""
    X = np.asarray(U, np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    V = len(X)
    cap = -(-V // C)
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(V, C, replace=False)]
    for _ in range(iters):
        sim = X @ cent.T                                                   # [V, C]
        order = np.argsort(-sim.max(1))                                    # most confident tokens first
        pref = np.argsort(-sim, axis=1)
        load = np.zeros(C, np.int64)
        cls = np.empty(V, np.int32)
        for w in order:
            for c in pref[w]:
                if load[c] < cap:
                    cls[w] = c
                    load[c] += 1
                    break
        for c in range(C):
            m = X[cls == c].mean(0)
            cent[c] = m / (np.linalg.norm(m) + 1e-8)
    return cls


# ---------------- diagnostic ----------------
def grad_retention(U, S, p, y_onehot):
    """Fraction of ||p - y||^2 lying in the logit directions the backbone can move:
    col([U, 1]) for a flat head, col([U, S]) for class-bias / two-level heads (pass S=None for flat)."""
    basis = U if S is None else jnp.concatenate([U, S], 1)
    basis = jnp.concatenate([basis, jnp.ones((U.shape[0], 1))], 1)
    Q, _ = jnp.linalg.qr(basis)
    r = (p - y_onehot).T                                                   # [V, B]
    return float(jnp.sum((Q.T @ r) ** 2) / jnp.sum(r ** 2))


def hier_nll_grouped(h, y, U, b, Vc, bc, cls, members, mask, pos):
    """Same result as hier_nll, but as one grouped matmul (jax.lax.ragged_dot): tokens are sorted by target
    class and each class's member embeddings are read once per step instead of once per token."""
    C = members.shape[0]
    c = cls[y]
    order = jnp.argsort(c, stable=True)
    hs, ys, cs = h[order], y[order], c[order]
    sizes = jnp.bincount(c, length=C).astype(jnp.int32)
    rhs = jnp.swapaxes(U[members], 1, 2)                                   # [C, d, L]
    zc = jax.lax.ragged_dot(hs, rhs, sizes) + b[members][cs]               # [B, L]
    zc = jnp.where(mask[cs], zc, -jnp.inf)
    lp_w = jnp.take_along_axis(jax.nn.log_softmax(zc, -1), pos[ys][:, None], 1)[:, 0]
    lp_c = jnp.take_along_axis(jax.nn.log_softmax(hs @ Vc + bc), cs[:, None], 1)[:, 0]
    return jnp.zeros_like(lp_w).at[order].set(-(lp_c + lp_w))
