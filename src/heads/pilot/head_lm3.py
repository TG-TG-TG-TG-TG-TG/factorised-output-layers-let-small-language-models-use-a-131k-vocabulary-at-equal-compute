# Same neural 4-gram LM / data as head_lm2.py; adds Argon's heads (MCF, bilinear Kronecker, 2-tick sub-token)
# and a two-level head on the same balanced classes, for a like-for-like comparison.
import sys, os, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import class_tables, hier_nll, balanced_kmeans
W = os.environ.get("WORK", "/home/claude/work/")
V, CTX, E, HID, d, C = 8192, 4, 64, 512, 32, 256          # C=256 balanced classes of L=32 (V1=256, V2=32)
STEPS, BATCH, LR, WARM, DECAY_FRAC, EVAL_EVERY = int(os.environ.get("STEPS", 2000)), 1024, 2e-3, 100, 0.2, 500
z = np.load(W + "ids_en8k.npz"); tr, te = z["tr"], z["te"]
ev_pos = np.random.default_rng(123).integers(CTX, len(te), size=131072)
def batch_at(ids, pos): return np.stack([ids[pos - CTX + j] for j in range(CTX)], 1), ids[pos]
ev_ctx, ev_y = batch_at(te, ev_pos)
rms = lambda x: x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
def init(key, arm):
    ks = iter(jax.random.split(key, 40)); n = lambda s, sc: sc * jax.random.normal(next(ks), s, jnp.float32)
    p = {"E": n((V, E), 1.0), "W1": n((CTX * E, HID), (CTX * E) ** -0.5), "b1": jnp.zeros(HID),
         "W2": n((HID, HID), HID ** -0.5), "b2": jnp.zeros(HID), "W3": n((HID, d), HID ** -0.5), "g": jnp.ones(d)}
    if arm in ("flat32", "hierfast_bal", "hier_mod"): p.update(U=n((V, d), d ** -0.5), b=jnp.zeros(V))
    if arm in ("hierfast_bal", "hier_mod"): p.update(Vc=n((d, C), d ** -0.5), bc=jnp.zeros(C))
    if arm == "hier_mod": p.update(Wg=n((d, d), d ** -0.5), Wu=n((d, d), d ** -0.5), Ec=n((C, d), 1.0), Bc=n((C, d), 1.0))
    if arm == "mcf":   # Argon Idea 1, d_f = d
        p.update(Wc=n((d, C), d ** -0.5), bc=jnp.zeros(C), Wg=n((d, d), d ** -0.5), Wu=n((d, d), d ** -0.5),
                 Ec=n((C, d), 1.0), Bc=n((C, d), 1.0), Wf=n((d, 32), d ** -0.5), bf=jnp.zeros(32))
    if arm == "kron":  # Argon Idea 2, r = 8, rows = classes (V1=256), cols = slot (V2=32)
        r = 8; p.update(K1=n((d, d), d ** -0.5), K2=n((d, d), d ** -0.5), KU=n((d, C * r), d ** -0.5),
                        KB=n((d, 32 * r), d ** -0.5), gU=jnp.ones(r), gB=jnp.ones(r))
    if arm == "tick2": # Argon Idea 3: v = 256*hi + lo, shared 256-way W_sub, hi-byte embedding + one FFN pass
        p.update(Ws=n((256, d), d ** -0.5), bs=jnp.zeros(256), Eh=n((V // 256, d), 1.0),
                 F1=n((d, 4 * d), d ** -0.5), F2=n((4 * d, d), (4 * d) ** -0.5))
    return p
def trunk(p, ctx):
    x = p["E"][ctx].reshape(ctx.shape[0], -1)
    x = jax.nn.gelu(x @ p["W1"] + p["b1"]); x = jax.nn.gelu(x @ p["W2"] + p["b2"])
    return rms(x @ p["W3"]) * p["g"]
def make_loss(arm, cls, members, mask, pos):
    ce = lambda logits, t: -jnp.take_along_axis(jax.nn.log_softmax(logits), t[:, None], 1)[:, 0]
    if arm == "flat32":
        return lambda p, ctx, y: ce(trunk(p, ctx) @ p["U"].T + p["b"], y).mean()
    if arm == "hierfast_bal":
        return lambda p, ctx, y: hier_nll(trunk(p, ctx), y, p["U"], p["b"], p["Vc"], p["bc"], cls, members, mask, pos).mean()
    if arm == "hier_mod":   # two-level head, per-token embeddings kept, MCF-style cluster modulation of h
        def f(p, ctx, y):
            h = trunk(p, ctx); i = cls[y]
            g = jax.nn.silu(h @ p["Wg"] + p["Ec"][i]) * (h @ p["Wu"] + p["Bc"][i])
            mem = members[i]
            zc = jnp.where(mask[i], jnp.einsum("bd,bld->bl", g, p["U"][mem]) + p["b"][mem], -jnp.inf)
            lp_w = jnp.take_along_axis(jax.nn.log_softmax(zc, -1), pos[y][:, None], 1)[:, 0]
            return (ce(h @ p["Vc"] + p["bc"], i) - lp_w).mean()
        return f
    if arm == "mcf":
        def f(p, ctx, y):
            h = trunk(p, ctx); i, j = cls[y], pos[y]
            g = jax.nn.silu(h @ p["Wg"] + p["Ec"][i]) * (h @ p["Wu"] + p["Bc"][i])
            return (ce(h @ p["Wc"] + p["bc"], i) + ce(g @ p["Wf"] + p["bf"], j)).mean()
        return f
    if arm == "kron":
        def f(p, ctx, y):
            h = trunk(p, ctx); r = p["gU"].shape[0]
            Uh = rms((jax.nn.silu(h @ p["K1"]) @ p["KU"]).reshape(-1, C, r)) * p["gU"]
            Bh = rms((jax.nn.silu(h @ p["K2"]) @ p["KB"]).reshape(-1, 32, r)) * p["gB"]
            Z = jnp.einsum("bir,bjr->bij", Uh, Bh).reshape(h.shape[0], -1)       # grid index = class*32 + slot
            return ce(Z, cls[y] * 32 + pos[y]).mean()
        return f
    if arm == "tick2":
        def f(p, ctx, y):
            h = trunk(p, ctx); hi, lo = y // 256, y % 256; nh = V // 256
            l_hi = ce((h @ p["Ws"].T + p["bs"])[:, :nh], hi)
            h2 = h + p["Eh"][hi]; h3 = h2 + jax.nn.gelu(rms(h2) @ p["F1"]) @ p["F2"]
            return (l_hi + ce(rms(h3) @ p["Ws"].T + p["bs"], lo)).mean()
        return f
def run(arm, cls, seed=0):
    p = init(jax.random.PRNGKey(seed), arm)
    if cls is None: cls = np.zeros(V, np.int32)
    members, mask, pos = class_tables(cls, C); cls = jnp.asarray(cls)
    loss_fn = make_loss(arm, cls, members, mask, pos)
    def lr_at(s):
        dec0 = STEPS * (1 - DECAY_FRAC)
        return LR * jnp.minimum(1.0, (s + 1) / WARM) * jnp.where(s < dec0, 1.0, jnp.maximum(0.0, (STEPS - s) / (STEPS - dec0)))
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    @jax.jit
    def step(p, m, v, s, ctx, y):
        l, g = jax.value_and_grad(loss_fn)(p, ctx, y); lr = lr_at(s); b1, b2 = 0.9, 0.95
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, g); v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        c1, c2 = 1 - b1 ** (s + 1), 1 - b2 ** (s + 1)
        return jax.tree.map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + 1e-8), p, m, v), m, v, l
    ev = jax.jit(loss_fn)
    evaluate = lambda p: float(np.mean([ev(p, ev_ctx[i:i+8192], ev_y[i:i+8192]) for i in range(0, len(ev_y), 8192)]))
    rng = np.random.default_rng(1000 + seed); curve = {}; t0 = time.time()
    for s in range(STEPS):
        ctx, y = batch_at(tr, rng.integers(CTX, len(tr), size=BATCH))
        p, m, v, l = step(p, m, v, jnp.asarray(s, jnp.float32), ctx, y)
        if (s + 1) % EVAL_EVERY == 0 or s + 1 == STEPS: curve[s + 1] = evaluate(p)
    return p, curve, time.time() - t0
if __name__ == "__main__":
    RES = W + "head_results4.json"; res = json.load(open(RES)) if os.path.exists(RES) else {}
    for arm in sys.argv[1:]:
        cls = None if arm == "flat32" else np.load(W + "classes_bal256.npy")
        p, curve, dt = run(arm, cls)
        if arm == "flat32":
            cb = balanced_kmeans(np.asarray(p["U"]), C); np.save(W + "classes_bal256.npy", cb.astype(np.int32))
            print("balanced class sizes:", np.bincount(cb, minlength=C).min(), np.bincount(cb, minlength=C).max())
        n_head = sum(int(np.prod(x.shape)) for k, x in p.items() if k not in ("E", "W1", "b1", "W2", "b2", "W3", "g"))
        res[arm] = {"curve": curve, "final": curve[STEPS], "sec": dt, "head_params": n_head}
        print(arm, {k: round(v, 4) for k, v in curve.items()}, f"head params {n_head:,}", f"{dt:.0f}s", flush=True)
        json.dump(res, open(RES, "w"), indent=1)
