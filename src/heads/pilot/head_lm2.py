# Neural 4-gram LM on real English (FineWeb-Edu sample, 8k byte-level BPE). Same backbone; only the output head differs.
# Usage: STEPS=2000 python -I head_lm2.py ARM [ARM ...]
# arms: flat32 flat64 flat128 cb_gate cb_gate_rand cb_mlp hier hierfast hierfast_rand mos2 plif
# Needs: WORK/bpe_en_8192.json (any 8k BPE), DATA/en.txt (raw text, docs split by blank lines), heads.py next to this file.
# Run flat32 first: it writes the k-means classes (C=256) used by the cb_* and hier* arms.
import sys, os, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from heads import class_tables, hier_nll
W = os.environ.get("WORK", "/home/claude/work/"); D = os.environ.get("DATA", "/home/claude/dl_data/")
V, CTX, E, HID, C = 8192, 4, 64, 512, 256
STEPS, BATCH, LR, WARM, DECAY_FRAC, EVAL_EVERY = int(os.environ.get("STEPS", 3000)), 1024, 2e-3, 100, 0.2, int(os.environ.get("EVAL_EVERY", 500))
# ---------- data (cached) ----------
if not os.path.exists(W + "ids_en8k.npz"):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(W + "bpe_en_8192.json")
    docs = [d for d in open(D + "en.txt", encoding="utf-8").read().split("\n\n") if d.strip()]
    k = int(len(docs) * 0.85)
    enc = lambda ds: np.concatenate([np.array(e.ids, np.int32) for e in tok.encode_batch(ds)])
    tr, te = enc(docs[:k]), enc(docs[k:])
    np.savez(W + "ids_en8k.npz", tr=tr, te=te)
z = np.load(W + "ids_en8k.npz"); tr, te = z["tr"], z["te"]
rng_eval = np.random.default_rng(123)
ev_pos = rng_eval.integers(CTX, len(te), size=131072)
def batch_at(ids, pos):
    ctx = np.stack([ids[pos - CTX + j] for j in range(CTX)], 1); return ctx, ids[pos]
ev_ctx, ev_y = batch_at(te, ev_pos)
# ---------- model ----------
def init(key, arm, d, cls):
    ks = iter(jax.random.split(key, 32)); n = lambda s, sc: sc * jax.random.normal(next(ks), s, jnp.float32)
    p = {"E": n((V, E), 1.0), "W1": n((CTX * E, HID), (CTX * E) ** -0.5), "b1": jnp.zeros(HID),
         "W2": n((HID, HID), HID ** -0.5), "b2": jnp.zeros(HID), "W3": n((HID, d), HID ** -0.5), "g": jnp.ones(d),
         "U": n((V, d), d ** -0.5), "b": jnp.zeros(V)}
    if arm.startswith("cb_gate"): p.update(Vg=n((d, C), d ** -0.5), ag=jnp.zeros(C))
    if arm == "cb_mlp": p.update(M1=n((d, C), d ** -0.5), m1=jnp.zeros(C), M2=n((C, C), C ** -0.5 * 0.1), m2=jnp.zeros(C))
    if arm.startswith("hier"): p.update(Vc=n((d, C), d ** -0.5), bc=jnp.zeros(C))
    if arm == "mos2": p.update(P=n((2, d, d), d ** -0.5), Wpi=n((d, 2), d ** -0.5))
    if arm == "plif": p.update(alpha=jnp.full(8, -3.0))
    return p
def trunk(p, ctx):
    x = p["E"][ctx].reshape(ctx.shape[0], -1)
    x = jax.nn.gelu(x @ p["W1"] + p["b1"]); x = jax.nn.gelu(x @ p["W2"] + p["b2"]); h = x @ p["W3"]
    return h * jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + 1e-6) * p["g"]
def logprobs(p, ctx, arm, cls):
    h = trunk(p, ctx); z = h @ p["U"].T + p["b"]
    if arm in ("flat32", "flat64", "flat128"): return jax.nn.log_softmax(z)
    if arm.startswith("cb_gate"): return jax.nn.log_softmax(z + jax.nn.log_sigmoid(h @ p["Vg"] + p["ag"])[:, cls])
    if arm == "cb_mlp": return jax.nn.log_softmax(z + (jax.nn.gelu(h @ p["M1"] + p["m1"]) @ p["M2"] + p["m2"])[:, cls])
    if arm == "hier":
        zt = z.T                                                     # V x B
        mx = jax.lax.stop_gradient(jax.ops.segment_max(zt, cls, num_segments=C))   # C x B
        lse = jnp.log(jax.ops.segment_sum(jnp.exp(zt - mx[cls]), cls, num_segments=C)) + mx  # C x B
        within = zt - lse[cls]
        return (jax.nn.log_softmax(h @ p["Vc"] + p["bc"])[:, cls] + within.T)
    if arm == "mos2":
        pi = jax.nn.log_softmax(h @ p["Wpi"])                        # B x 2
        comps = [jax.nn.log_softmax(jnp.tanh(h @ p["P"][k]) @ p["U"].T + p["b"]) + pi[:, k:k+1] for k in range(2)]
        return jnp.logaddexp(comps[0], comps[1])
    if arm == "plif":
        t = jnp.linspace(-4.0, 6.0, 8); a = jax.nn.softplus(p["alpha"])   # piecewise-linear increasing f (PLIF-style)
        f = z
        for j in range(8): f = f + a[j] * jax.nn.relu(z - t[j])
        return jax.nn.log_softmax(f)
    raise ValueError(arm)
def run(arm, cls, seed=0):
    d = {"flat64": 64, "flat128": 128, "hierfast64": 64, "hierfast128": 128}.get(arm, 32)
    p = init(jax.random.PRNGKey(seed), arm, d, cls)
    cls_j = jnp.asarray(cls if cls is not None else np.zeros(V, np.int32))
    if arm.startswith("hierfast"):
        members, mask, pos = class_tables(np.asarray(cls_j), C)
        def loss_fn(p, ctx, y):
            return hier_nll(trunk(p, ctx), y, p["U"], p["b"], p["Vc"], p["bc"], cls_j, members, mask, pos).mean()
    else:
        def loss_fn(p, ctx, y):
            lp = logprobs(p, ctx, arm, cls_j); return -jnp.take_along_axis(lp, y[:, None], 1).mean()
    def lr_at(s):
        dec0 = STEPS * (1 - DECAY_FRAC)
        return LR * jnp.minimum(1.0, (s + 1) / WARM) * jnp.where(s < dec0, 1.0, jnp.maximum(0.0, (STEPS - s) / (STEPS - dec0)))
    m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
    @jax.jit
    def step(p, m, v, s, ctx, y):
        l, g = jax.value_and_grad(loss_fn)(p, ctx, y); lr = lr_at(s); b1, b2 = 0.9, 0.95
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, g); v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        c1, c2 = 1 - b1 ** (s + 1), 1 - b2 ** (s + 1)
        p = jax.tree.map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + 1e-8), p, m, v)
        return p, m, v, l
    ev = jax.jit(loss_fn)
    def evaluate(p):
        return float(np.mean([ev(p, ev_ctx[i:i+8192], ev_y[i:i+8192]) for i in range(0, len(ev_y), 8192)]))
    rng = np.random.default_rng(1000 + seed); curve = {}; t0 = time.time()
    for s in range(STEPS):
        ctx, y = batch_at(tr, rng.integers(CTX, len(tr), size=BATCH))
        p, m, v, l = step(p, m, v, jnp.asarray(s, jnp.float32), ctx, y)
        if (s + 1) % EVAL_EVERY == 0 or s + 1 == STEPS: curve[s + 1] = evaluate(p)
    dt = time.time() - t0
    return p, curve, dt
if __name__ == "__main__":
    res = json.load(open(W + os.environ.get("RES", "head_results2.json"))) if os.path.exists(W + os.environ.get("RES", "head_results2.json")) else {}
    for arm in sys.argv[1:]:
        cls = None
        if arm.startswith("cb_") or arm.startswith("hier"):
            if arm.endswith("_rand"):
                cls = np.random.default_rng(7).integers(0, C, V).astype(np.int32)
            else:
                cls = np.load(W + "classes_kmeans256.npy")
        p, curve, dt = run(arm, cls)
        if arm == "flat32" and not os.path.exists(W + "classes_kmeans256.npy"):
            from sklearn.cluster import KMeans
            Un = np.asarray(p["U"]); Un = Un / np.linalg.norm(Un, axis=1, keepdims=True)
            km = KMeans(n_clusters=C, n_init=1, max_iter=100, random_state=0).fit(Un)
            np.save(W + "classes_kmeans256.npy", km.labels_.astype(np.int32))
            sizes = np.bincount(km.labels_, minlength=C); print("kmeans class sizes min/med/max", sizes.min(), np.median(sizes), sizes.max())
        res[f"{arm}@{STEPS}"] = {"curve": curve, "final": curve[STEPS], "sec": dt}
        print(arm, {k: round(v, 4) for k, v in curve.items()}, f"{dt:.0f}s", flush=True)
        json.dump(res, open(W + os.environ.get("RES", "head_results2.json"), "w"), indent=1)
