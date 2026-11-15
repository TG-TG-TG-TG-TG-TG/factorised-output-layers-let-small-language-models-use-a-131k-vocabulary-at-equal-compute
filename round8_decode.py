"""Round 8, step 7: batch-1 decode, flat head vs two-level head (round-6 decode protocol).

Random bf16 weights passed as jit arguments; KV-cache decode of NEW = 256 tokens from position 512;
the 256-step scan is timed whole, ms/token = time / 256; median of REPS = 5 interleaved repeats
(warm-up/compile excluded) per process; 3 processes on an idle GPU.

Configs
  tiny: the round-5/8 proxy, 6 x d128, MHA 2 heads (hd 64), GELU FFN 512, tied V = 50304 flat / 50257
        two-level with C = 1024 (capacity 50)
  big:  the 600M systems config of the brief, 28 x d1280, GQA 20 q / 4 kv heads (hd 64), SwiGLU 3456,
        tied V = 65536, two-level C = 1024 (capacity 64)
Heads
  none         backbone only (next token = argmax of h[:8]); "head ms" = variant - none
  flat_greedy  argmax(h E^T)           flat_sample  categorical(h E^T)
  hier_sample  heads.hier_sample (exact ancestral sample)
  hier_greedy  jitted exact-greedy: expand the top-2 classes of p(c|h); if the best word's log-prob is
               below the 3rd class's log-prob, expand the 3rd too (lax.cond). Counts how often that
               fallback fires, and how often even 3 classes don't certify the argmax (bound vs 4th class)
Random weights make p(c|h) nearly flat, so the fallback counts here say nothing about trained models;
`fallback` measures them on the trained H2 checkpoint.

python round8_decode.py run <proc>     appends runs/round8/decode.jsonl
python round8_decode.py fallback       trained H2: jitted greedy vs exact argmax on val -> runs/round8/decode_fallback.json
python round8_decode.py md             -> runs/round8/decode.md
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from round8_heads import H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round8")
F32, BF = jnp.float32, jnp.bfloat16
PROMPT, NEW, REPS = 512, 256, 5
CFG = {"tiny": dict(L=6, d=128, Hq=2, Hkv=2, hd=64, f=512, ffn="gelu", V=50304, Vh=50257, C=1024),
       "big": dict(L=28, d=1280, Hq=20, Hkv=4, hd=64, f=3456, ffn="swiglu", V=65536, Vh=65536, C=1024)}
HEADS = ["none", "flat_greedy", "flat_sample", "hier_sample", "hier_greedy"]


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "n/a"


def build(kind, seed=0):
    c = CFG[kind]
    d, hd, f = c["d"], c["hd"], c["f"]
    ks = iter(jax.random.split(jax.random.PRNGKey(seed), 16 * c["L"] + 8))
    n = lambda s, std=0.02: (std * jax.random.normal(next(ks), s, F32)).astype(BF)
    layers = []
    for _ in range(c["L"]):
        l = {"Wq": n((d, c["Hq"], hd)), "Wk": n((d, c["Hkv"], hd)), "Wv": n((d, c["Hkv"], hd)),
             "Wo": n((c["Hq"], hd, d)), "ln1": jnp.ones((d,), F32), "ln2": jnp.ones((d,), F32)}
        if c["ffn"] == "gelu":
            l.update(Wup=n((d, f)), Wdown=n((f, d)))
        else:
            l.update(Wg=n((d, f)), Wu=n((d, f)), Wd=n((f, d)))
        layers.append(l)
    C, Vh = c["C"], c["Vh"]
    cls = np.arange(Vh) % C  # balanced, sizes ceil/floor(Vh / C); timing does not depend on which tokens
    members, mask, _ = H.class_tables(cls, C)
    p = {"layers": layers, "emb": n((c["V"], d)), "ln_out": jnp.ones((d,), F32),
         "Vc": n((d, C), d ** -0.5), "bc": jnp.zeros((C,), F32), "members": members, "mask": mask,
         "b": jnp.zeros((Vh,), F32)}
    return c, p


def rms(x, g):
    x = x.astype(F32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def hier_greedy_jit(h, U, b, Vc, bc, members, mask):
    """Exact-or-flagged greedy for one context: top-2 classes, a 3rd only if the bound isn't met."""
    lpc = jax.nn.log_softmax((h @ Vc).astype(F32) + bc)
    top, idx = jax.lax.top_k(lpc, 4)

    def expand(j):
        c = idx[j]
        mem = members[c]
        z = jnp.where(mask[c], (U[mem] @ h).astype(F32) + b[mem], -jnp.inf)
        lw = jax.nn.log_softmax(z)
        i = jnp.argmax(lw)
        return top[j] + lw[i], mem[i]

    s0, w0 = expand(0)
    s1, w1 = expand(1)
    s, w = jnp.where(s1 > s0, s1, s0), jnp.where(s1 > s0, w1, w0)
    need3 = s < top[2]
    s2, w2 = jax.lax.cond(need3, lambda: expand(2), lambda: (jnp.float32(-jnp.inf), jnp.int32(-1)))
    s, w = jnp.where(s2 > s, s2, s), jnp.where(s2 > s, w2, w)
    unresolved = need3 & (s < top[3])
    return w.astype(jnp.int32), need3, unresolved


def make_step(c, head):
    L, Hq, Hkv, hd = c["L"], c["Hq"], c["Hkv"], c["hd"]
    G = Hq // Hkv
    Tmax = PROMPT + NEW
    inv = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2) / hd))

    def rope1(x, pos):
        ang = pos * inv
        co, si = jnp.cos(ang), jnp.sin(ang)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return jnp.concatenate([x1 * co - x2 * si, x1 * si + x2 * co], -1)

    def step(p, carry, _):
        tok, pos, K, Vv, key, n3, nun = carry
        x = p["emb"][tok].astype(F32)
        Ks, Vs = [], []
        for li in range(L):
            l = p["layers"][li]
            xn = rms(x, l["ln1"]).astype(BF)
            q = rope1(jnp.einsum("d,dnh->nh", xn, l["Wq"]).astype(F32), pos)
            k = rope1(jnp.einsum("d,dnh->nh", xn, l["Wk"]).astype(F32), pos)
            v = jnp.einsum("d,dnh->nh", xn, l["Wv"])
            Kl = jax.lax.dynamic_update_slice(K[li], k[None].astype(BF), (pos, 0, 0))
            Vl = jax.lax.dynamic_update_slice(Vv[li], v[None].astype(BF), (pos, 0, 0))
            s = jnp.einsum("kgh,tkh->kgt", q.reshape(Hkv, G, hd).astype(BF), Kl, preferred_element_type=F32) / math.sqrt(hd)
            s = jnp.where(jnp.arange(Tmax)[None, None] <= pos, s, -1e30)
            o = jnp.einsum("kgt,tkh->kgh", jax.nn.softmax(s, -1).astype(BF), Vl).reshape(Hq, hd)
            x = x + jnp.einsum("nh,nhd->d", o, l["Wo"]).astype(F32)
            xn = rms(x, l["ln2"]).astype(BF)
            if "Wup" in l:
                y = jax.nn.gelu(xn @ l["Wup"]) @ l["Wdown"]
            else:
                y = (jax.nn.silu(xn @ l["Wg"]) * (xn @ l["Wu"])) @ l["Wd"]
            x = x + y.astype(F32)
            Ks.append(Kl)
            Vs.append(Vl)
        hN = rms(x, p["ln_out"])
        key, sk = jax.random.split(key)
        Uh = p["emb"][: c["Vh"]]
        if head == "none":
            nxt = jnp.argmax(hN[:8]).astype(jnp.int32)
        elif head in ("flat_greedy", "flat_sample"):
            logits = jnp.einsum("d,vd->v", hN.astype(BF), p["emb"], preferred_element_type=F32)
            nxt = jnp.argmax(logits) if head == "flat_greedy" else jax.random.categorical(sk, logits)
        elif head == "hier_sample":
            nxt = H.hier_sample(sk, hN.astype(BF)[None], Uh, p["b"], p["Vc"], p["bc"], p["members"], p["mask"])[0]
        else:
            nxt, f3, un = hier_greedy_jit(hN.astype(BF), Uh, p["b"], p["Vc"], p["bc"], p["members"], p["mask"])
            n3, nun = n3 + f3.astype(jnp.int32), nun + un.astype(jnp.int32)
        return (nxt.astype(jnp.int32), pos + 1, jnp.stack(Ks), jnp.stack(Vs), key, n3, nun), None

    carry0 = (jnp.int32(0), jnp.int32(PROMPT), jnp.zeros((L, Tmax, Hkv, hd), BF), jnp.zeros((L, Tmax, Hkv, hd), BF),
              jax.random.PRNGKey(1), jnp.int32(0), jnp.int32(0))
    return step, carry0


def run(proc):
    os.makedirs(OUT, exist_ok=True)
    pre = gpu_state()
    runs = {}
    for kind in ("tiny", "big"):
        c, p = build(kind)
        for head in HEADS:
            step, carry0 = make_step(c, head)
            comp = jax.jit(lambda p, carry: jax.lax.scan(lambda cc, x: step(p, cc, x), carry, None, length=NEW)[0]
                           ).lower(p, carry0).compile()
            out = jax.block_until_ready(comp(p, carry0))  # warm-up, not timed
            runs[(kind, head)] = (comp, (p, carry0), {"fallback3": int(out[5]), "unresolved": int(out[6])})
    times = {k: [] for k in runs}
    for _ in range(REPS):
        for k, (comp, args, _) in runs.items():
            t = time.perf_counter()
            jax.block_until_ready(comp(*args))
            times[k].append(1000 * (time.perf_counter() - t) / NEW)
    post = gpu_state()
    with open(os.path.join(OUT, "decode.jsonl"), "a") as f:
        for (kind, head), ts in times.items():
            rec = {"proc": proc, "config": kind, "head": head, "ms_per_token": statistics.median(ts), "all": ts,
                   **runs[(kind, head)][2], "prompt": PROMPT, "new": NEW, "reps": REPS, "gpu_before": pre,
                   "gpu_after": post, "device": str(jax.devices()[0]), "cfg": CFG[kind]}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k not in ("all", "cfg")}), flush=True)


def fallback(ckpt="wsd_H2_s0", n=4096):
    """Trained H2: the jitted top-2(+1) greedy against the exact argmax of hier_logprobs_full."""
    from arms import ARMS
    from data import val_rows
    from model import forward
    from round8_heads import head_inputs, load_params, tables
    cfg = ARMS["tH2"]
    p = load_params(cfg, ckpt)
    cls, members, mask, _ = tables(cfg.head_classes, cfg.head_C)
    val = val_rows(1024)
    rows = jnp.asarray(val[:: len(val) // (n // 1024)][: n // 1024])
    hs = jnp.concatenate([forward(p, rows[i:i + 1, :-1], cfg)[0] for i in range(len(rows))])
    _, U, b, hp = head_inputs(p, hs, cfg)
    w, f3, un = jax.jit(jax.vmap(lambda h: hier_greedy_jit(h, U, b, hp["Vc"], hp["bc"], members, mask)))(hs)
    exact = []
    for i in range(0, len(hs), 512):
        exact.append(jnp.argmax(H.hier_logprobs_full(hs[i:i + 512], U, b, hp["Vc"], hp["bc"], cls, cfg.head_C), -1))
    exact = jnp.concatenate(exact)
    ex = [H.hier_greedy(hs[i], U, b, hp["Vc"], hp["bc"], members, mask)[1] for i in range(256)]
    res = {"ckpt": ckpt, "positions": int(len(hs)), "fallback3_rate": float(f3.mean()), "unresolved_rate": float(un.mean()),
           "jit_greedy_equals_exact_argmax": float((w == exact).mean()),
           "jit_greedy_equals_exact_when_resolved": float(((w == exact) | un)[~un].mean()) if (~un).any() else None,
           "host_greedy_classes_expanded_mean_256": float(np.mean(ex)), "host_greedy_classes_expanded_max_256": int(max(ex))}
    json.dump(res, open(os.path.join(OUT, "decode_fallback.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def write_md():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "decode.jsonl"))]
    procs = sorted({r["proc"] for r in rows})
    L = ["# Round 8 decode benchmark (batch 1, random bf16 weights)", "",
         f"KV-cache decode of {NEW} tokens from position {PROMPT}; ms/token = median of {REPS} interleaved repeats per "
         f"process; mean over {len(procs)} processes. head ms = variant - backbone-only ('none'). "
         "`python round8_decode.py run <proc>` (proc 0, 1, 2), then `python round8_decode.py md`.", ""]
    for kind in ("tiny", "big"):
        c = CFG[kind]
        get = lambda hd: [next(r for r in rows if r["config"] == kind and r["head"] == hd and r["proc"] == q) for q in procs]
        base = np.mean([r["ms_per_token"] for r in get("none")])
        L += [f"## {kind}: {c['L']} x d{c['d']}, {c['Hq']}/{c['Hkv']} heads, {c['ffn']} {c['f']}, V {c['V']}, C {c['C']}", "",
              "| head | " + " | ".join(f"proc {q}" for q in procs) + " | mean ms/tok | head ms/tok | vs flat_greedy (whole) | fallback-3 / unresolved (of 256) |",
              "|---|" + "---|" * len(procs) + "---|---|---|---|"]
        fg = np.mean([r["ms_per_token"] for r in get("flat_greedy")])
        for hd in HEADS:
            rs = get(hd)
            m = np.mean([r["ms_per_token"] for r in rs])
            fb = f"{rs[0]['fallback3']} / {rs[0]['unresolved']}" if hd == "hier_greedy" else ""
            L.append(f"| {hd} | " + " | ".join(f"{r['ms_per_token']:.4f}" for r in rs) +
                     f" | {m:.4f} | {m - base:+.4f} | {100 * (m / fg - 1):+.1f}% | {fb} |")
        L.append("")
    L.append("GPU (util, mem, SM clock) before/after: " + "; ".join(
        f"proc {q}: {next(r['gpu_before'] for r in rows if r['proc'] == q)} / {next(r['gpu_after'] for r in rows if r['proc'] == q)}" for q in procs))
    fp = os.path.join(OUT, "decode_fallback.json")
    if os.path.exists(fp):
        L += ["", "Trained H2 (val contexts): " + json.dumps(json.load(open(fp)))]
    open(os.path.join(OUT, "decode.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2])), "fallback": fallback, "md": write_md}[sys.argv[1]]()
