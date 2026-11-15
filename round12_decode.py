"""Round 12 decode timing: round11_decode.py's protocol (random bf16 weights, KV-cache decode of 256 tokens from position
512, the 256-step scan timed whole, median of 5 interleaved repeats per process, 3 processes) for the d=384 arms B384,
H3*-384, F8k-384, F128k-384 and H128k-384, at batch 1 and batch 32 (the step vmapped over 32 independent sequences, one
shared position). ms/step = scan time / 256; ms/token = ms/step / batch; ms/byte = ms/token / val bytes per token of the
arm's tokenizer (GPT-2 4.633, BPE-8k 3.933, BPE-131k 4.944 = runs/round12/bpe131072.json). Two-level heads: K = 8,192
singletons + C = 4,096 round-robin tail clusters over the arm's real vocabulary (50,257 or 131,072).

python round12_decode.py run <proc> <batch>   appends runs/round12/decode.jsonl
python round12_decode.py md                   -> runs/round12/decode.md
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
import round11_decode as D11

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round12")
F32, BF = jnp.float32, jnp.bfloat16
NEW, REPS = D11.NEW, D11.REPS
K, C = 8192, 4096
# name: (d, H, f, head, vocab rows, real vocab, bytes/token)
ARMS = {"B384": (384, 6, 1536, "flat_sample", 50304, 50257, 4.633),
        "H3s384": (384, 6, 1536, "hier9_sample", 50304, 50257, 4.633),
        "F8k384": (384, 6, 1536, "flat_sample", 8192, 8192, 3.933),
        "F128k384": (384, 6, 1536, "flat_sample", 131072, 131072, 4.944),
        "H128k384": (384, 6, 1536, "hier9_sample", 131072, 131072, 4.944)}


def build(arm, seed=0):
    d, H, f, head, Vv, VH, _ = ARMS[arm]
    P = 6
    hd = d // H
    ks = iter(jax.random.split(jax.random.PRNGKey(seed), 16 * P + 16))
    n = lambda s, std=0.02: (std * jax.random.normal(next(ks), s, F32)).astype(BF)
    blocks = [{"Wq": n((d, H, hd)), "Wk": n((d, H, hd)), "Wv": n((d, H, hd)), "Wo": n((H, hd, d)),
               "Wup": n((d, f)), "Wdown": n((f, d))} for _ in range(P)]
    copies = [{"ln1": jnp.ones((d,), F32), "ln2": jnp.ones((d,), F32), "a1": jnp.full((d,), 1 / math.sqrt(2 * P), F32),
               "a2": jnp.full((d,), 1 / math.sqrt(2 * P), F32)} for _ in range(P)]
    L = -(-(VH - K) // C)
    members = (K + (np.arange(C)[:, None] + C * np.arange(L)[None, :])).astype(np.int32)
    mask = members < VH
    members = np.where(mask, members, K)
    p = {"blocks": blocks, "copies": copies, "emb": n((Vv, d)), "ac": jnp.zeros((C,), F32), "ln_out": jnp.ones((d,), F32),
         "Wtop": n((K + C, d), d ** -0.5), "btop": jnp.zeros((K + C,), F32), "freq": jnp.arange(K, dtype=jnp.int32),
         "members": jnp.asarray(members), "mask": jnp.asarray(mask)}
    return dict(P=P, R=1, d=d, H=H, hd=hd, A=P, ng=0, head=head, K=K, C=C, M=0), p


def run(proc, B):
    os.makedirs(OUT, exist_ok=True)
    pre = D8.gpu_state()
    runs = {}
    for arm in ARMS:
        c, p = build(arm)
        step, carry0 = D11.make_step(c)
        if B > 1:
            ax = (0, 0, 0, None, 0, 0, 0)
            carry0 = tuple(x if a is None else jnp.broadcast_to(x, (B,) + x.shape) for x, a in zip(carry0, ax))
            carry0 = carry0[:6] + (jax.random.split(jax.random.PRNGKey(1), B),)
            vstep = jax.vmap(lambda pp, cc: step(pp, cc, None)[0], in_axes=(None, ax), out_axes=ax)
            body = lambda pp, cc, _: (vstep(pp, cc), None)
        else:
            body = step
        comp = jax.jit(lambda p, carry, body=body: jax.lax.scan(lambda cc, x: body(p, cc, x), carry, None, length=NEW)[0]
                       ).lower(p, carry0).compile()
        jax.block_until_ready(comp(p, carry0))
        runs[arm] = (comp, (p, carry0))
    times = {k: [] for k in runs}
    for _ in range(REPS):
        for k, (comp, args) in runs.items():
            t = time.perf_counter()
            jax.block_until_ready(comp(*args))
            times[k].append(1000 * (time.perf_counter() - t) / NEW)
    post = D8.gpu_state()
    with open(os.path.join(OUT, "decode.jsonl"), "a") as fh:
        for arm, ts in times.items():
            ms = statistics.median(ts)
            rec = {"proc": proc, "batch": B, "arm": arm, "spec": ARMS[arm], "ms_per_step": ms, "ms_per_token": ms / B,
                   "ms_per_byte": ms / B / ARMS[arm][6], "all": ts, "new": NEW, "reps": REPS, "gpu_before": pre, "gpu_after": post,
                   "device": str(jax.devices()[0])}
            fh.write(json.dumps(rec) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k != "all"}), flush=True)


def write_md():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "decode.jsonl"))]
    L = ["# Round 12 decode: random bf16 weights, round-8 protocol, batch 1 and 32", "",
         "ms/step = median of 5 repeats per process (mean (min-max) over processes); ms/token = ms/step / batch; "
         "ms/byte = ms/token / val bytes per token.", "",
         "| arm | head | vocab | batch | ms/step | min-max | ms/token | ms/byte | vs B384 (same batch) |", "|---|---|---|---|---|---|---|---|---|"]
    for B in sorted({r["batch"] for r in rows}):
        base = np.mean([r["ms_per_step"] for r in rows if r["arm"] == "B384" and r["batch"] == B])
        for a, (d, H, f, head, Vv, VH, bpt) in ARMS.items():
            t = [r["ms_per_step"] for r in rows if r["arm"] == a and r["batch"] == B]
            if t:
                L.append(f"| {a} | {head} | {VH:,} | {B} | {np.mean(t):.4f} | {min(t):.4f}-{max(t):.4f} | {np.mean(t) / B:.4f} | "
                         f"{np.mean(t) / B / bpt:.5f} | {np.mean(t) / base:.2f}x |")
    open(os.path.join(OUT, "decode.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        write_md()
