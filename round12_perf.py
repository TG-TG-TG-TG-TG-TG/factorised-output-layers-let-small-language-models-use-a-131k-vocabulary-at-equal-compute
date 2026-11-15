"""Round 12 follow-up (2026-11-11, user's request): memory and speed of the five d=384 heads. Reported measurements, no
decision rule (nothing here is a verdict).

Arms: B (flat 50k), H3s384 (two-level 50k), F8k384 (flat 8k), F128k384 (flat 131k), H128k384 (two-level 131k); random
init (speed and memory do not depend on the weights), real validation rows of each arm's tokenizer as inputs.

  train <arm>   the exact train.py step (grad + global-norm clip + AdamW, one micro-batch, no accumulation) at micro
                16, 32, 64 rows of 1,024 tokens: tokens/s (median of 3 x 10 steps after 3 warm-up steps), peak GPU memory
                (jax peak_bytes_in_use; r12_perf.sh runs every config in its own fresh process, because configs sharing
                a process slowed each other down) and
                XLA's compiled estimate (arguments + outputs - aliased + temporaries).
  score <arm>   forward-only loss (lm_loss, no grad; what evaluation runs) at 64 and 128 rows: tokens/s, memory.
  decode        round12_decode.py's KV-cache decode (random bf16 weights, 256 new tokens from position 512) at batch
                1, 32, 128, all arms in one process: ms/step (median of 5) and XLA's compiled memory estimate.
  md            -> runs/round12/perf.md

Every record is appended to runs/round12/perf.jsonl; round12_analyze.py folds it into round12_results.json (key perf).
"""
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round12")
LOG = os.path.join(OUT, "perf.jsonl")
TRAIN_ARMS = {"B384": "B", "H3s384": "H3s384", "F8k384": "F8k384", "F128k384": "F128k384", "H128k384": "H128k384"}
MICROS = (16, 32, 64)
SCORE_ROWS = (64, 128)
DEC_BATCH = (1, 32, 128)  # one process: decode programs are small (KV cache < 1 GB)
GB = 2 ** 30


def write(rec):
    rec["device"] = str(jax.devices()[0])
    rec["t"] = round(time.time(), 1)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def mem_est(comp):
    m = comp.memory_analysis()
    if m is None:
        return None
    return (m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes + m.temp_size_in_bytes) / GB


def peak():
    return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / GB


def timed(fn, n_warm=3, n=10, reps=3):
    for _ in range(n_warm):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n):
            out = fn()
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t) / n)
    return statistics.median(ts)


def setup(arm):
    from arms import ARMS
    from data import val_rows
    from model import init_params
    cfg = replace(ARMS[TRAIN_ARMS[arm]], attn_impl="cudnn")
    params = init_params(cfg, jax.random.PRNGKey(0))
    val = np.asarray(val_rows(1024, source=cfg.data))
    return cfg, params, val


def train(arm, micros=MICROS):
    from model import decay_mask, lm_loss
    cfg, params, val = setup(arm)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(1.0, b1=0.9, b2=0.95, weight_decay=0.1, mask=decay_mask(params)))
    opt_state = opt.init(params)

    def step(params, opt_state, b, lr):
        l, g = jax.value_and_grad(lambda p: lm_loss(p, b, cfg))(params)
        upd, opt_state = opt.update(g, opt_state, params)
        upd = jax.tree.map(lambda u: lr * u, upd)
        return optax.apply_updates(params, upd), opt_state, l

    for m in micros:
        b = jnp.asarray(val[:m].astype(np.int32))
        rec = {"kind": "train", "arm": arm, "micro": m, "tokens_per_step": m * 1024}
        try:
            comp = jax.jit(step, donate_argnums=(0, 1)).lower(params, opt_state, b, jnp.float32(1e-4)).compile()
            rec["mem_est_gb"] = mem_est(comp)
            state = [params, opt_state]

            def run():
                state[0], state[1], l = comp(state[0], state[1], b, jnp.float32(1e-4))
                return l
            s = timed(run)
            params, opt_state = state
            rec.update(ok=True, s_per_step=s, tok_s=m * 1024 / s, peak_gb=peak())
        except Exception as e:  # OOM is a result, not a crash
            rec.update(ok=False, error=str(e).split("\n")[0][:300])
            write(rec)
            break
        write(rec)


def score(arm, rows=SCORE_ROWS):
    from model import lm_loss
    cfg, params, val = setup(arm)
    for n in rows:
        b = jnp.asarray(val[:n].astype(np.int32))
        rec = {"kind": "score", "arm": arm, "rows": n, "tokens": n * 1024}
        try:
            comp = jax.jit(lambda p, b: lm_loss(p, b, cfg)).lower(params, b).compile()
            rec["mem_est_gb"] = mem_est(comp)
            s = timed(lambda: comp(params, b), n=5)
            rec.update(ok=True, s_per_call=s, tok_s=n * 1024 / s, peak_gb=peak())
        except Exception as e:
            rec.update(ok=False, error=str(e).split("\n")[0][:300])
            write(rec)
            break
        write(rec)


def decode():
    import round11_decode as D11
    import round12_decode as D12
    for B in DEC_BATCH:
        for arm in D12.ARMS:
            c, p = D12.build(arm)
            step, carry0 = D11.make_step(c)
            if B > 1:
                ax = (0, 0, 0, None, 0, 0, 0)
                carry0 = tuple(x if a is None else jnp.broadcast_to(x, (B,) + x.shape) for x, a in zip(carry0, ax))
                carry0 = carry0[:6] + (jax.random.split(jax.random.PRNGKey(1), B),)
                vstep = jax.vmap(lambda pp, cc: step(pp, cc, None)[0], in_axes=(None, ax), out_axes=ax)
                body = lambda pp, cc, _: (vstep(pp, cc), None)
            else:
                body = step
            rec = {"kind": "decode", "arm": arm, "batch": B, "new": D11.NEW}
            try:
                comp = jax.jit(lambda p, carry, body=body: jax.lax.scan(lambda cc, x: body(p, cc, x), carry, None,
                                                                         length=D11.NEW)[0]).lower(p, carry0).compile()
                rec["mem_est_gb"] = mem_est(comp)
                s = timed(lambda: comp(p, carry0), n_warm=1, n=1, reps=5)
                ms = 1000 * s / D11.NEW
                rec.update(ok=True, ms_per_step=ms, ms_per_token=ms / B, tok_s=1000 * B / ms)
            except Exception as e:
                rec.update(ok=False, error=str(e).split("\n")[0][:300])
            write(rec)


def md():
    rows = [json.loads(l) for l in open(LOG)]
    last = {}
    for r in rows:  # the latest record per configuration wins (re-runs replace earlier ones)
        last[(r["kind"], r["arm"], r.get("micro") or r.get("rows") or r.get("batch"))] = r
    L = ["# Round 12 follow-up: memory and speed of the d=384 heads (random init, RTX 5070 Ti)", "",
         "Use peak GB for training and the XLA estimate for scoring and decode: the scoring peak is set before the "
         "scoring call (it is the same at 32 and 64 rows), so it does not measure the call.", ""]
    for kind, col, unit in (("train", "micro", "rows/step"), ("score", "rows", "rows"), ("decode", "batch", "batch")):
        L += [f"## {kind}", "", f"| arm | {unit} | tokens/s | peak GB | XLA estimate GB |", "|---|---|---|---|---|"]
        for (k, arm, n), r in sorted(last.items(), key=lambda kv: (kv[0][0], kv[0][2], list(TRAIN_ARMS).index(kv[0][1]))):
            if k != kind:
                continue
            if r.get("ok"):
                L.append(f"| {arm} | {n} | {r['tok_s']:,.0f} | {r.get('peak_gb', float('nan')):.2f} | {r.get('mem_est_gb') or float('nan'):.2f} |")
            else:
                L.append(f"| {arm} | {n} | failed: {r.get('error', '')[:80]} | - | {r.get('mem_est_gb') or float('nan'):.2f} |")
        L.append("")
    open(os.path.join(OUT, "perf.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sizes = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    {"train": lambda: train(sys.argv[2], sizes or MICROS), "score": lambda: score(sys.argv[2], sizes or SCORE_ROWS), "decode": decode, "md": md}[sys.argv[1]]()
