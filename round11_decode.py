"""Round 11 (copy of round10_decode.py: d=384 arms, small-vocab flat heads, the re-scored two-level head; ms per byte =
ms/token / val bytes per token of the arm's tokenizer). Round 10: batch-1 decode ms/token for B, H3*, R-loop, R-deep, R-wide and the n-gram arms (round-8 tiny protocol).

Protocol (round8_decode.py): random bf16 weights passed as jit arguments; KV-cache decode of NEW = 256 tokens from
position 512; the 256-step scan is timed whole, ms/token = time / 256; median of REPS = 5 interleaved repeats
(warm-up/compile excluded) per process; 3 processes on an idle GPU. Every application of a block has its own KV
cache and its own per-copy RMSNorm gains and alphas (the trained arms all learn alpha); R-loop's 24 applications
read 6 weight sets and add the per-neuron gates gelu(u * gam + beta). Heads:
  flat_sample / flat_greedy   categorical / argmax of h E^T over the 50,304 tied rows (B)
  hier9_sample                exact ancestral sample of the two-level head: one categorical over the T = K + C
                              top-level logits (the top matrix [U[freq]; ubar + R] is fixed at inference, so it is a
                              precomputed [T, d] weight), then, if a cluster was drawn, one categorical over its <= L
                              member words (computed for every step and selected, no branch)
  ngram (D arms)              + G2[h2(prev, tok)] (+ G3[h3(prev2, prev, tok)]) added to the input embedding,
                              hashes as model.ngram_ids, tables 2^20 x d bf16
Configs: B, H3* (6 x d128, H 2, f 512); Rloop (6 blocks x 4 passes); Rdeep (24 x d128); Rwide (6 x d256, H 4, f 1024);
D1 / D2 = H3* + n-gram gathers.

python round10_decode.py run <proc> <K> <C>    appends runs/round10/decode.jsonl
python round10_decode.py md                    -> runs/round10/decode.md
"""
import json
import math
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import round8_decode as D8
from model import NG_MISSING, NG_MULT, NG_P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round11")
F32, BF = jnp.float32, jnp.bfloat16
PROMPT, NEW, REPS = D8.PROMPT, D8.NEW, D8.REPS
V, VH = 50304, 50257
# name: (unique blocks, passes, d, H, f, gates, ngram, head)
# name: (unique blocks, passes, d, H, f, gates, ngram, head, vocab rows, M re-scored clusters)
ARMS = {"B": (6, 1, 128, 2, 512, False, 0, "flat_sample", 50304, 0),
        "H3s": (6, 1, 128, 2, 512, False, 0, "hier9_sample", 50304, 0),
        "H3rs16": (6, 1, 128, 2, 512, False, 0, "hier9_sample", 50304, 16),
        "H3rs64": (6, 1, 128, 2, 512, False, 0, "hier9_sample", 50304, 64),
        "B384": (6, 1, 384, 6, 1536, False, 0, "flat_sample", 50304, 0),
        "H3s384": (6, 1, 384, 6, 1536, False, 0, "hier9_sample", 50304, 0),
        "H3rs16_384": (6, 1, 384, 6, 1536, False, 0, "hier9_sample", 50304, 16),
        "F8k384": (6, 1, 384, 6, 1536, False, 0, "flat_sample", 8192, 0),
        "F4k384": (6, 1, 384, 6, 1536, False, 0, "flat_sample", 4096, 0),
        "H3s512": (6, 1, 512, 8, 2048, False, 0, "hier9_sample", 50304, 0)}
BYTES_PER_TOKEN = {"": 4.633, "F8k384": 3.933, "F4k384": 3.473}


def build(arm, K, C, seed=0):
    P, R, d, H, f, gates, ng, head, Vv, M = ARMS[arm]
    hd = d // H
    ks = iter(jax.random.split(jax.random.PRNGKey(seed), 16 * P + 16))
    n = lambda s, std=0.02: (std * jax.random.normal(next(ks), s, F32)).astype(BF)
    blocks = [{"Wq": n((d, H, hd)), "Wk": n((d, H, hd)), "Wv": n((d, H, hd)), "Wo": n((H, hd, d)),
               "Wup": n((d, f)), "Wdown": n((f, d))} for _ in range(P)]
    A = P * R
    copies = [{"ln1": jnp.ones((d,), F32), "ln2": jnp.ones((d,), F32), "a1": jnp.full((d,), 1 / math.sqrt(2 * A), F32),
               "a2": jnp.full((d,), 1 / math.sqrt(2 * A), F32),
               **({"gam": jnp.ones((f,), F32), "beta": jnp.zeros((f,), F32)} if gates else {})} for _ in range(A)]
    L = -(-(VH - K) // C)
    members = (K + (np.arange(C)[:, None] + C * np.arange(L)[None, :])).astype(np.int32)  # round-robin tail clusters
    mask = members < VH
    members = np.where(mask, members, K)
    p = {"blocks": blocks, "copies": copies, "emb": n((Vv, d)), "ac": jnp.zeros((C,), F32), "ln_out": jnp.ones((d,), F32),
         "Wtop": n((K + C, d), d ** -0.5), "btop": jnp.zeros((K + C,), F32), "freq": jnp.arange(K, dtype=jnp.int32),
         "members": jnp.asarray(members), "mask": jnp.asarray(mask)}
    for k in range(2, ng + 1):
        p[f"ng{k}"] = n((2 ** 20, d))
        p[f"ngt{k}"] = tuple(jnp.asarray(((m * np.arange(V, dtype=np.int64)) % NG_P).astype(np.uint32)) for m in NG_MULT[k])
    return dict(P=P, R=R, d=d, H=H, hd=hd, A=A, ng=ng, head=head, K=K, C=C, M=M), p


def make_step(c):
    P, R, H, hd, A, K = c["P"], c["R"], c["H"], c["hd"], c["A"], c["K"]
    Tmax = PROMPT + NEW
    inv = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2) / hd))

    def rope1(x, pos):
        ang = pos * inv
        co, si = jnp.cos(ang), jnp.sin(ang)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return jnp.concatenate([x1 * co - x2 * si, x1 * si + x2 * co], -1)

    def step(p, carry, _):
        tok, prev, prev2, pos, Kc, Vc, key = carry
        x = p["emb"][tok].astype(F32)
        if c["ng"] >= 2:
            t2 = p["ngt2"]
            h2 = (t2[0][prev] + t2[1][tok]) % jnp.uint32(NG_P)
            x = x + p["ng2"][(h2 % jnp.uint32(2 ** 20)).astype(jnp.int32)].astype(F32)
        if c["ng"] >= 3:
            t3 = p["ngt3"]
            h3 = ((t3[0][prev2] + t3[1][prev]) % jnp.uint32(NG_P) + t3[2][tok]) % jnp.uint32(NG_P)
            x = x + p["ng3"][(h3 % jnp.uint32(2 ** 20)).astype(jnp.int32)].astype(F32)
        Ks, Vs = [], []
        for ai in range(A):
            l, cp = p["blocks"][ai % P], p["copies"][ai]
            xn = D8.rms(x, cp["ln1"]).astype(BF)
            q = rope1(jnp.einsum("d,dnh->nh", xn, l["Wq"]).astype(F32), pos)
            k = rope1(jnp.einsum("d,dnh->nh", xn, l["Wk"]).astype(F32), pos)
            v = jnp.einsum("d,dnh->nh", xn, l["Wv"])
            Kl = jax.lax.dynamic_update_slice(Kc[ai], k[None].astype(BF), (pos, 0, 0))
            Vl = jax.lax.dynamic_update_slice(Vc[ai], v[None].astype(BF), (pos, 0, 0))
            s = jnp.einsum("nh,tnh->nt", q.astype(BF), Kl, preferred_element_type=F32) / math.sqrt(hd)
            s = jnp.where(jnp.arange(Tmax)[None] <= pos, s, -1e30)
            o = jnp.einsum("nt,tnh->nh", jax.nn.softmax(s, -1).astype(BF), Vl)
            x = x + cp["a1"] * jnp.einsum("nh,nhd->d", o, l["Wo"]).astype(F32)
            xn = D8.rms(x, cp["ln2"]).astype(BF)
            u = xn @ l["Wup"]
            if "gam" in cp:
                u = (u.astype(F32) * cp["gam"] + cp["beta"]).astype(BF)
            x = x + cp["a2"] * (jax.nn.gelu(u) @ l["Wdown"]).astype(F32)
            Ks.append(Kl)
            Vs.append(Vl)
        hN = D8.rms(x, p["ln_out"]).astype(BF)
        key, k1, k2 = jax.random.split(key, 3)
        if c["head"].startswith("flat"):
            logits = jnp.einsum("d,vd->v", hN, p["emb"], preferred_element_type=F32)
            nxt = jnp.argmax(logits) if c["head"] == "flat_greedy" else jax.random.categorical(k1, logits)
        else:
            zt = (p["Wtop"] @ hN).astype(F32) + p["btop"]
            if c["M"]:  # round 11: the top-M tail clusters get exact LSE_{w in c}(u_w . h) + a_c
                idx = jax.lax.top_k(zt[K:], c["M"])[1]
                zr = jnp.where(p["mask"][idx], (p["emb"][p["members"][idx]] @ hN).astype(F32), -jnp.inf)
                zt = zt.at[K + idx].set(jax.nn.logsumexp(zr, -1) + p["ac"][idx])
            s_top = jax.random.categorical(k1, zt)
            cl = jnp.maximum(s_top - K, 0)
            mem = p["members"][cl]
            zw = jnp.where(p["mask"][cl], (p["emb"][mem] @ hN).astype(F32), -jnp.inf)
            w = mem[jax.random.categorical(k2, zw)]
            nxt = jnp.where(s_top < K, p["freq"][jnp.minimum(s_top, K - 1)], w) if K else w
        return (nxt.astype(jnp.int32), tok, prev, pos + 1, jnp.stack(Ks), jnp.stack(Vs), key), None

    carry0 = (jnp.int32(1), jnp.int32(NG_MISSING), jnp.int32(NG_MISSING), jnp.int32(PROMPT),
              jnp.zeros((A, Tmax, H, hd), BF), jnp.zeros((A, Tmax, H, hd), BF), jax.random.PRNGKey(1))
    return step, carry0


def run(proc, K, C):
    os.makedirs(OUT, exist_ok=True)
    pre = D8.gpu_state()
    runs = {}
    for arm in ARMS:
        c, p = build(arm, K, C)
        step, carry0 = make_step(c)
        comp = jax.jit(lambda p, carry: jax.lax.scan(lambda cc, x: step(p, cc, x), carry, None, length=NEW)[0]
                       ).lower(p, carry0).compile()
        jax.block_until_ready(comp(p, carry0))  # warm-up, not timed
        runs[arm] = (comp, (p, carry0))
    times = {k: [] for k in runs}
    for _ in range(REPS):
        for k, (comp, args) in runs.items():
            t = time.perf_counter()
            jax.block_until_ready(comp(*args))
            times[k].append(1000 * (time.perf_counter() - t) / NEW)
    post = D8.gpu_state()
    with open(os.path.join(OUT, "decode.jsonl"), "a") as f:
        for arm, ts in times.items():
            rec = {"proc": proc, "K": K, "C": C, "arm": arm, "spec": ARMS[arm], "ms_per_token": statistics.median(ts), "all": ts,
                   "prompt": PROMPT, "new": NEW, "reps": REPS, "gpu_before": pre, "gpu_after": post, "device": str(jax.devices()[0])}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k != "all"}), flush=True)


def write_md():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "decode.jsonl"))]
    procs = sorted({r["proc"] for r in rows})
    get = lambda a: [r["ms_per_token"] for r in rows if r["arm"] == a]
    L = ["# Round 11 decode: batch 1, random bf16 weights, round-8 tiny protocol", "",
         f"K = {rows[0]['K']} singletons + C = {rows[0]['C']} tail clusters. ms/token = median of {REPS} repeats per process; "
         f"mean (min-max) over {len(procs)} processes. ms/byte = ms/token / val bytes per token (GPT-2 4.633, BPE-8k 3.933, BPE-4k 3.473).", "",
         "| arm | d | head | vocab | ms/token | min-max | ms/byte | vs flat GPT-2 of the same width |", "|---|---|---|---|---|---|---|---|"]
    for a, (P, R, d, H, f, g, ng, head, Vv, M) in ARMS.items():
        t = get(a)
        if t:
            base = np.mean(get("B" if d == 128 else "B384")) if get("B" if d == 128 else "B384") else float("nan")
            bpt = BYTES_PER_TOKEN.get(a, BYTES_PER_TOKEN[""])
            L.append(f"| {a} | {d} | {head}{f' + top-{M} re-scoring' if M else ''} | {Vv:,} | {np.mean(t):.4f} | {min(t):.4f}-{max(t):.4f} | "
                     f"{np.mean(t) / bpt:.4f} | {np.mean(t) / base:.2f}x |")
    open(os.path.join(OUT, "decode.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        write_md()
