"""Round 9, Part F: exact jitted greedy decoding for the two-level heads (round-8 decode protocol).

exact_greedy(h): top-level log-probs lt over T = K + C; best singleton (if K > 0); expand the top 3 clusters
(their best word: lt[cluster] + within log-softmax). Certified if best >= lt of the 4th cluster (p(w) <= p(c(w)));
otherwise (lax.cond) the full flat head for that step: z = U h for all V words, per-cluster LSE by segment ops,
log p(w) = lt[slot(w)] + (tail ? z_w - LSE_c(w) : 0), argmax. So the result is always the exact argmax.

Timing (batch-1 KV-cache decode, 256 tokens from position 512, median of 5 interleaved repeats per process,
3 processes; random bf16 weights as round 8): random weights make p(c|h) flat, so the fallback fires at every
step. Two timed variants bracket it: exact_greedy_nofb (the top-3 path only) and exact_greedy_fb (the full
fallback every step); expected ms/token at the trained fallback rate r = nofb + r * (fb - nofb).
Rows: none, flat_greedy, hier_sample (round 8's) plus the two bracket variants; head = H2* 's shape
(K singletons + C clusters; tiny: K 4096 or 0, C 1024; 600M: same K, C 1024 over the remaining 65,536 - K).

python round9_decode.py fallback <label> <arm> <ckpt>   trained H2*: fallback rate + exactness on 4096 val positions
python round9_decode.py run <proc> <K>                 appends runs/round9/decode.jsonl
python round9_decode.py md                             -> runs/round9/decode.md
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round9")
F32, BF = jnp.float32, jnp.bfloat16
HEADS = ["none", "flat_greedy", "hier_sample", "exact_greedy_nofb", "exact_greedy_fb"]


def tables_for(V, K, C):
    """slot, tail, grp, members, mask, freq for V tokens: the first K ids are singletons, the rest round-robin."""
    ids = np.arange(V)
    tail = ids >= K
    grp = np.where(tail, (ids - K) % C, 0)
    L = -(-(V - K) // C)
    members = np.zeros((C, L), np.int32)
    mask = np.zeros((C, L), bool)
    fill = np.zeros(C, int)
    for w in ids[tail]:
        members[grp[w], fill[grp[w]]], mask[grp[w], fill[grp[w]]] = w, True
        fill[grp[w]] += 1
    slot = np.where(tail, K + grp, ids)
    return [jnp.asarray(a) for a in (slot, tail, grp.astype(np.int32), members, mask, ids[:K].astype(np.int32))]


def exact_greedy(h, U, Wc, bc, K, tb, mode="exact"):
    """h [d]; U [V, d] word matrix; Wc [C, d] class weights. Returns (token, fallback_fired)."""
    slot, tail, grp, members, mask, freq = tb
    C = Wc.shape[0]
    zc = (Wc @ h).astype(F32) + bc
    if K:
        zf = (U[freq] @ h).astype(F32)
        lt = jax.nn.log_softmax(jnp.concatenate([zf, zc]))
        lf, lcl = lt[:K], lt[K:]
        sbest, wbest = jnp.max(lf), freq[jnp.argmax(lf)]
    else:
        lcl = jax.nn.log_softmax(zc)
        lt = lcl
        sbest, wbest = jnp.float32(-jnp.inf), jnp.int32(-1)
    top, idx = jax.lax.top_k(lcl, 4)
    s, w = sbest, wbest
    for j in range(3):
        c = idx[j]
        mem = members[c]
        z = jnp.where(mask[c], (U[mem] @ h).astype(F32), -jnp.inf)
        lw = jax.nn.log_softmax(z)
        i = jnp.argmax(lw)
        better = top[j] + lw[i] > s
        s, w = jnp.where(better, top[j] + lw[i], s), jnp.where(better, mem[i], w)
    certified = s >= top[3]

    def full():
        z = (U @ h).astype(F32)
        seg = jnp.where(tail, grp, C)
        mx = jax.ops.segment_max(z, seg, num_segments=C + 1)
        lse = jnp.log(jax.ops.segment_sum(jnp.exp(z - mx[seg]), seg, num_segments=C + 1)) + mx
        return jnp.argmax(lt[slot] + jnp.where(tail, z - lse[seg], 0.0)).astype(jnp.int32)

    if mode == "nofb":
        return w.astype(jnp.int32), ~certified
    if mode == "fb":
        return full(), ~certified
    return jax.lax.cond(certified, lambda: w.astype(jnp.int32), full), ~certified


def fallback(label, arm, ckpt, n=4096):
    from arms import ARMS
    from data import val_rows
    from model import forward
    from round8_heads import load_params
    import round9_heads as R9
    from round8_heads import H, head_inputs, tables as tables8
    cfg = ARMS[arm]
    p = load_params(cfg, ckpt)
    val = val_rows(1024)
    rows = jnp.asarray(val[:: len(val) // (n // 1024)][: n // 1024])
    hs = jnp.concatenate([forward(p, rows[i:i + 1, :-1], cfg)[0] for i in range(len(rows))])
    if cfg.head == "hier9":
        slot, tail, grp, members, mask, pos, freq, count = R9.tables(cfg)
        Wt, bt = R9.top_matrix(p, cfg, R9.tables(cfg))
        K = cfg.head_adapt
        Wc, bc = Wt[K:], bt[K:]
        U = R9.word_matrix(p, cfg)
        ref = lambda x: R9.ref_logprobs_full(p, x, cfg)
    else:
        cls, members, mask, _ = tables8(cfg.head_classes, cfg.head_C)
        _, U, b, hp = head_inputs(p, hs, cfg)
        K, Wc, bc = 0, hp["Vc"].T, hp["bc"]
        tail, grp, slot, freq = jnp.ones(U.shape[0], bool), cls, cls, jnp.zeros(0, jnp.int32)
        ref = lambda x: H.hier_logprobs_full(x, U, b, hp["Vc"], hp["bc"], cls, cfg.head_C)
    tb = (slot, tail, grp, members, mask, freq)
    w, fired = jax.jit(jax.vmap(lambda h: exact_greedy(h, U, Wc, bc, K, tb)))(hs)
    exact = jnp.concatenate([jnp.argmax(ref(hs[i:i + 512]), -1) for i in range(0, len(hs), 512)])
    res = {"label": label, "ckpt": ckpt, "positions": int(len(hs)), "fallback_rate": float(fired.mean()),
           "exact_match": float((w == exact).mean())}
    json.dump(res, open(os.path.join(OUT, "decode_fallback.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def run(proc, K):
    """Times the round-8 backbone with each head; the head is applied to the final hidden state inside the scan."""
    os.makedirs(OUT, exist_ok=True)
    pre = D8.gpu_state()
    runs = {}
    for kind in ("tiny", "big"):
        c, p = D8.build(kind)
        tb = tables_for(c["Vh"], K, c["C"])
        Uh = p["emb"][: c["Vh"]]
        Wc = p["Vc"].T

        def headfn(head, hN, sk):
            if head == "none":
                return jnp.argmax(hN[:8]).astype(jnp.int32), jnp.bool_(False)
            if head == "flat_greedy":
                return jnp.argmax(jnp.einsum("d,vd->v", hN.astype(BF), p["emb"], preferred_element_type=F32)).astype(jnp.int32), jnp.bool_(False)
            if head == "hier_sample":
                return D8.H.hier_sample(sk, hN.astype(BF)[None], Uh, p["b"], p["Vc"], p["bc"], p["members"], p["mask"])[0].astype(jnp.int32), jnp.bool_(False)
            return exact_greedy(hN.astype(BF), Uh, Wc, p["bc"], K, tb, mode=head.split("_")[-1])
        for head in HEADS:
            step = D8_step_with_head(c, lambda hN, sk, head=head: headfn(head, hN, sk))
            carry0 = D8_carry0(c)
            comp = jax.jit(lambda p, carry: jax.lax.scan(lambda cc, x: step(p, cc, x), carry, None, length=D8.NEW)[0]
                           ).lower(p, carry0).compile()
            out = jax.block_until_ready(comp(p, carry0))
            runs[(kind, head)] = (comp, (p, carry0), {"fallback_fired": int(out[5])})
    times = {k: [] for k in runs}
    for _ in range(D8.REPS):
        for k, (comp, args, _) in runs.items():
            t = time.perf_counter()
            jax.block_until_ready(comp(*args))
            times[k].append(1000 * (time.perf_counter() - t) / D8.NEW)
    post = D8.gpu_state()
    with open(os.path.join(OUT, "decode.jsonl"), "a") as f:
        for (kind, head), ts in times.items():
            rec = {"proc": proc, "K": K, "config": kind, "head": head, "ms_per_token": statistics.median(ts), "all": ts,
                   **runs[(kind, head)][2], "gpu_before": pre, "gpu_after": post, "device": str(jax.devices()[0])}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k != "all"}), flush=True)


def D8_carry0(c):
    L, Hkv, hd = c["L"], c["Hkv"], c["hd"]
    Tmax = D8.PROMPT + D8.NEW
    return (jnp.int32(0), jnp.int32(D8.PROMPT), jnp.zeros((L, Tmax, Hkv, hd), BF), jnp.zeros((L, Tmax, Hkv, hd), BF),
            jax.random.PRNGKey(1), jnp.int32(0), jnp.int32(0))


def D8_step_with_head(c, headfn):
    """round8_decode.make_step's backbone (copied verbatim) with a pluggable head; carry[5] counts fallbacks."""
    L, Hq, Hkv, hd = c["L"], c["Hq"], c["Hkv"], c["hd"]
    G = Hq // Hkv
    Tmax = D8.PROMPT + D8.NEW
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
            xn = D8.rms(x, l["ln1"]).astype(BF)
            q = rope1(jnp.einsum("d,dnh->nh", xn, l["Wq"]).astype(F32), pos)
            k = rope1(jnp.einsum("d,dnh->nh", xn, l["Wk"]).astype(F32), pos)
            v = jnp.einsum("d,dnh->nh", xn, l["Wv"])
            Kl = jax.lax.dynamic_update_slice(K[li], k[None].astype(BF), (pos, 0, 0))
            Vl = jax.lax.dynamic_update_slice(Vv[li], v[None].astype(BF), (pos, 0, 0))
            s = jnp.einsum("kgh,tkh->kgt", q.reshape(Hkv, G, hd).astype(BF), Kl, preferred_element_type=F32) / math.sqrt(hd)
            s = jnp.where(jnp.arange(Tmax)[None, None] <= pos, s, -1e30)
            o = jnp.einsum("kgt,tkh->kgh", jax.nn.softmax(s, -1).astype(BF), Vl).reshape(Hq, hd)
            x = x + jnp.einsum("nh,nhd->d", o, l["Wo"]).astype(F32)
            xn = D8.rms(x, l["ln2"]).astype(BF)
            if "Wup" in l:
                y = jax.nn.gelu(xn @ l["Wup"]) @ l["Wdown"]
            else:
                y = (jax.nn.silu(xn @ l["Wg"]) * (xn @ l["Wu"])) @ l["Wd"]
            x = x + y.astype(F32)
            Ks.append(Kl)
            Vs.append(Vl)
        hN = D8.rms(x, p["ln_out"])
        key, sk = jax.random.split(key)
        nxt, fired = headfn(hN, sk)
        return (nxt.astype(jnp.int32), pos + 1, jnp.stack(Ks), jnp.stack(Vs), key, n3 + fired.astype(jnp.int32), nun), None
    return step


def write_md():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "decode.jsonl"))]
    procs = sorted({r["proc"] for r in rows})
    K = rows[0]["K"]
    fb = json.load(open(os.path.join(OUT, "decode_fallback.json"))) if os.path.exists(os.path.join(OUT, "decode_fallback.json")) else None
    r_ = fb["fallback_rate"] if fb else None
    L = ["# Round 9 Part F: exact jitted greedy (batch 1, random bf16 weights, round-8 protocol)", "",
         f"Head shape: K = {K} singletons + C = 1024 clusters. ms/token = median of 5 repeats per process, mean over "
         f"{len(procs)} processes. 'expected' = nofb + r·(fb − nofb) with the trained fallback rate r"
         + (f" = {r_:.4f} ({fb['label']}, {fb['positions']} val positions; exact match {fb['exact_match']:.4f})" if fb else "") + ".", ""]
    for kind in ("tiny", "big"):
        get = lambda hd: np.mean([r["ms_per_token"] for r in rows if r["config"] == kind and r["head"] == hd])
        fg = get("flat_greedy")
        L += [f"## {kind}", "", "| head | ms/token | vs flat_greedy |", "|---|---|---|"]
        for hd in HEADS:
            L.append(f"| {hd} | {get(hd):.4f} | {100 * (get(hd) / fg - 1):+.1f}% |")
        if r_ is not None:
            e = get("exact_greedy_nofb") + r_ * (get("exact_greedy_fb") - get("exact_greedy_nofb"))
            L.append(f"| exact_greedy, expected at r = {r_:.3f} | {e:.4f} | {100 * (e / fg - 1):+.1f}% |")
        L.append("")
    open(os.path.join(OUT, "decode.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fallback":
        fallback(*sys.argv[2:5])
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        write_md()
