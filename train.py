"""Train one arm on the streamed FineWeb-Edu tokens and log everything to JSONL.

python train.py --arm B --seed 0 --lr 2e-3 --tokens 130e6 [--frac 0.25] [--name B_s0]
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from arms import ARMS, has_shared
from data import DATA_DIR, Batches, val_rows
from model import SHARED, count_params, decay_mask, flops_per_token, init_params, lm_loss, schedule

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of --tokens (LR sweep)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch", type=int, default=64, help="rows of 1024 tokens per step")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--n_evals", type=int, default=10)
    ap.add_argument("--eval_rows", type=int, default=976, help="periodic eval rows (~1M tokens)")
    ap.add_argument("--cos_every", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (calibration)")
    ap.add_argument("--attn_impl", default="cudnn")
    ap.add_argument("--remat", action="store_true")
    ap.add_argument("--keep_params", action="store_true")
    ap.add_argument("--keep_branch", default="", help="wsd: also save the decayed params of the branches at these token counts")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--runs_dir", default=RUNS)
    # WSD: warmup, constant peak LR to --tokens, decay branches forked from the stable phase
    ap.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--branch_at", default="", help="wsd: comma-separated token counts; the last must be --tokens")
    ap.add_argument("--decay_frac", type=float, default=0.1, help="wsd: branch length / steps so far")
    ap.add_argument("--warmup_tokens", type=float, default=3e6, help="wsd: warmup length")
    ap.add_argument("--neuron_every", type=float, default=25e6, help="gated arms: per-neuron stats every N tokens")
    ap.add_argument("--ng_lazy", action="store_true", help="round 10: row-sparse AdamW for the n-gram tables")
    return ap.parse_args()


def main():
    a = parse()
    from dataclasses import replace
    cfg = replace(ARMS[a.arm], attn_impl=a.attn_impl or None, remat=a.remat or ARMS[a.arm].remat)
    name = a.name or f"{a.arm}_s{a.seed}"
    RUNS = a.runs_dir
    os.makedirs(RUNS, exist_ok=True)
    log_path = os.path.join(RUNS, name + ".jsonl")
    if os.path.exists(os.path.join(RUNS, name + ".DONE")):
        sys.exit(f"{name} is finished (DONE marker); refusing to overwrite its log")
    if os.path.exists(log_path):
        os.remove(log_path)  # a run restarts from scratch; DONE marks finished ones
    logf = open(log_path, "a")

    def log(**kv):
        kv["t"] = round(time.time(), 2)
        logf.write(json.dumps(kv) + "\n")
        logf.flush()

    def say(msg):
        print(msg, flush=True)
        log(event="msg", msg=msg)

    tok_per_step = a.batch * a.seq
    steps = max(1, int(round(a.tokens * a.frac / tok_per_step)))
    if a.max_steps:
        steps_run = min(steps, a.max_steps)
    else:
        steps_run = steps
    accum = a.batch // a.micro
    assert accum * a.micro == a.batch

    params = init_params(cfg, jax.random.PRNGKey(a.seed))
    pc = count_params(params)
    fl = flops_per_token(cfg, a.seq)
    wsd = a.sched == "wsd"
    branches = {}
    if not wsd:
        warm = max(1, int(a.warmup * steps))
        sched = optax.join_schedules(
            [optax.linear_schedule(0.0, a.lr, warm),
             optax.cosine_decay_schedule(a.lr, steps - warm, alpha=0.1)], [warm])
    else:
        # the LR is an argument of the step (optax lr = 1, update scaled by lr), so the stable
        # phase and every decay branch share one compiled step
        warm = max(1, int(round(a.warmup_tokens / tok_per_step)))
        sched = lambda c: a.lr * min(1.0, c / warm)  # optax convention: step s uses sched(s - 1)
        for t in a.branch_at.split(","):
            S = int(round(float(t) / tok_per_step))
            branches[S] = max(1, int(round(a.decay_frac * S)))
        assert steps in branches, "the last branch must sit at the end of the stable phase"
    lazy = a.ng_lazy
    if not lazy:
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(1.0 if wsd else sched, b1=0.9, b2=0.95, weight_decay=a.wd,
                                      mask=decay_mask(params)))
        opt_state = opt.init(params)
    else:  # round 10 lazy n-gram tables: dense AdamW on the rest, row-sparse AdamW (same b1, b2, eps, wd) on ng*
        from model import ngram_ids
        from round10_lazy import adamw_rows, sparse_grads
        assert wsd and cfg.ngram, "--ng_lazy needs --sched wsd and an n-gram arm"
        NG = [f"ng{n}" for n in range(2, cfg.ngram + 1)]
        dense_of = lambda p: {k: v for k, v in p.items() if k not in NG}
        opt = optax.adamw(1.0, b1=0.9, b2=0.95, weight_decay=a.wd, mask=decay_mask(dense_of(params)))
        opt_state = (opt.init(dense_of(params)), {k: (jnp.zeros_like(params[k]), jnp.zeros_like(params[k])) for k in NG},
                     jnp.zeros((), jnp.int32))
    log(event="config", name=name, arm=a.arm, cfg=dict(cfg.__dict__), args=vars(a),
        steps=steps, steps_run=steps_run, tokens_per_step=tok_per_step, params=pc, flops_per_token=fl,
        device=str(jax.devices()[0]), warmup_steps=warm, branches=branches)
    say(f"{name}: {pc['non_embedding']:,} non-emb params, {steps} steps, accum {accum}x{a.micro}")

    def loss_fn(p, b):
        return lm_loss(p, b, cfg)

    def train_step(params, opt_state, batch, lr):
        mb = batch.reshape(accum, a.micro, -1)

        def body(g_acc, b):
            l, g = jax.value_and_grad(loss_fn)(params, b)
            return jax.tree.map(jnp.add, g_acc, g), l

        g0 = jax.tree.map(jnp.zeros_like, params)
        g, ls = jax.lax.scan(body, g0, mb)
        g = jax.tree.map(lambda x: x / accum, g)
        gnorm = optax.global_norm(g)
        upd, opt_state = opt.update(g, opt_state, params)
        if wsd:
            upd = jax.tree.map(lambda u: lr * u, upd)
        return optax.apply_updates(params, upd), opt_state, jnp.mean(ls), gnorm

    def train_step_lazy(params, opt_state, batch, lr):
        """Same maths as the dense step for every row a batch touches (global-norm clip over all grads, AdamW with
        bias correction by the global step count, decoupled decay); rows not in the batch are left alone (lazy)."""
        dst, sst, cnt = opt_state
        mb = batch.reshape(accum, a.micro, -1)
        dense = dense_of(params)

        def body(g_acc, b):
            ids = {k: ngram_ids(b[:, :-1], int(k[2:]), cfg) for k in NG}
            rows = {k + "_rows": params[k][ids[k]] for k in NG}
            l, (gd, gr) = jax.value_and_grad(lambda dd, rr: lm_loss({**dd, **rr}, b, cfg), argnums=(0, 1))(dense, rows)
            return jax.tree.map(jnp.add, g_acc, gd), (l, ids, gr)

        g0 = jax.tree.map(jnp.zeros_like, dense)
        gd, (ls, ids, gr) = jax.lax.scan(body, g0, mb)
        gd = jax.tree.map(lambda x: x / accum, gd)
        N = accum * a.micro * (mb.shape[-1] - 1)
        sparse = {k: sparse_grads(ids[k].reshape(N), gr[k + "_rows"].reshape(N, -1), cfg.ngram_M, accum) for k in NG}
        gnorm = jnp.sqrt(optax.global_norm(gd) ** 2 + sum(jnp.sum(g * g) for _, g in sparse.values()))
        scale = 1.0 / jnp.maximum(1.0, gnorm)           # optax.clip_by_global_norm(1.0), over dense + sparse grads
        upd, dst = opt.update(jax.tree.map(lambda x: x * scale, gd), dst, dense)
        new = optax.apply_updates(dense, jax.tree.map(lambda x: lr * x, upd))
        cnt = cnt + 1
        sst = dict(sst)
        for k in NG:
            u, g = sparse[k]
            new[k], m_, v_ = adamw_rows(params[k], *sst[k], u, g * scale, cnt, lr, a.wd)
            sst[k] = (m_, v_)
        return new, (dst, sst, cnt), jnp.mean(ls), gnorm

    train_step = jax.jit(train_step_lazy if lazy else train_step, donate_argnums=(0, 1))

    @jax.jit
    def eval_sum(params, rows):
        return lm_loss(params, rows, cfg) * rows[:, 1:].size

    @jax.jit
    def grad_cos(params, b):
        g_main = jax.grad(lambda p: lm_loss(p, b, cfg, sg_use=1))(params)
        g_sec = jax.grad(lambda p: lm_loss(p, b, cfg, sg_use=0))(params)
        out = {}
        for t in SHARED:
            cs = []
            for i in range(cfg.n_unique):
                x, y = g_main["blocks"][i][t].ravel(), g_sec["blocks"][i][t].ravel()
                cs.append(jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-20))
            out[t] = jnp.stack(cs)
        return out

    val = val_rows(a.seq, source=cfg.data)  # round 11: cfg.data selects the tokenizer / corpus ("" = GPT-2 FineWeb-Edu)

    def evaluate(rows, p=None):
        p = params if p is None else p
        tot, n = 0.0, 0
        eb = a.micro
        for i in range(0, len(rows) - eb + 1, eb):
            r = jnp.asarray(rows[i:i + eb])
            tot += float(eval_sum(p, r))
            n += r[:, 1:].size
        return tot / n

    eval_at = sorted({max(1, int(round(steps * k / a.n_evals))) for k in range(1, a.n_evals + 1)})
    data = Batches(a.batch, a.seq, seed=0, log=say, source=cfg.data)
    replay = collections.deque()  # batches a branch consumed; the stable phase sees them next

    def next_batch():
        if replay:
            data.wait_s = 0.0
            return replay.popleft()
        return data.next()

    def log_alpha(p, step):
        if "a_ffn" not in p["copies"][0]:
            return
        log(event="alpha", step=step, tokens=step * tok_per_step,
            **{k: [float(jnp.mean(jnp.abs(c[k]))) for c in p["copies"]] for k in ("a_attn", "a_ffn")})

    sched_list = schedule(cfg)
    neuron_at = {int(round(k * a.neuron_every / tok_per_step)) for k in range(1, int(a.tokens // a.neuron_every) + 1)}

    def log_neurons(p, step, **kv):
        """Per effective block: share of neurons switched off (beta < -2) and the gate means."""
        if not cfg.gate:
            return
        cps = [p["copies"][cp] for _, _, cp in sched_list]
        rec = {"block": [b for b, _, _ in sched_list], "use": [u for _, u, _ in sched_list],
               "beta_off": [float(jnp.mean(c["beta"] < -2)) for c in cps],
               "gam_mean": [float(jnp.mean(c["gam"])) for c in cps],
               "beta_mean": [float(jnp.mean(c["beta"])) for c in cps]}
        log(event="neurons", step=step, tokens=step * tok_per_step, **rec, **kv)

    def save_params(p, fname):
        ck = os.path.join(DATA_DIR, "ckpt")
        os.makedirs(ck, exist_ok=True)
        flat, _ = jax.tree_util.tree_flatten_with_path(p)
        np.savez(os.path.join(ck, fname + ".npz"), **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})

    keep_branch = {int(round(float(t) / tok_per_step)) for t in a.keep_branch.split(",") if t}

    def run_branch(S, p, st):
        """Linear decay from peak to 10% over D steps, from the stable state at step S."""
        D = branches[S]
        tb = time.time()
        buf = []
        for k in range(1, D + 1):
            b = next_batch()
            buf.append(b)
            lr_k = a.lr * (1.0 - 0.9 * k / D)
            p, st, loss, gnorm = train_step(p, st, jnp.asarray(b), jnp.float32(lr_k))
            if k % 50 == 0 or k == D:
                loss_f = float(loss)
                log(event="branch_train", branch=S, k=k, tokens=(S + k) * tok_per_step, loss=loss_f,
                    gnorm=float(gnorm), lr=lr_k)
                if not math.isfinite(loss_f):
                    say("non-finite loss in branch; aborting")
                    logf.close()
                    os._exit(1)
        jax.block_until_ready(p)
        train_s = time.time() - tb
        te = time.time()
        vf = evaluate(val, p)
        log(event="branch", branch_step=S, stable_tokens=S * tok_per_step, decay_steps=D,
            tokens=(S + D) * tok_per_step, val_loss_full=vf, val_loss=evaluate(val[: a.eval_rows], p),
            train_s=train_s, eval_s=time.time() - te)
        log_alpha(p, S + D)
        log_neurons(p, S + D, branch=S)
        if S in keep_branch and S < steps:  # the last branch is the final params (--keep_params)
            save_params(p, f"{name}_branch{S * tok_per_step / 1e6:.0f}M")
        say(f"  branch @ {S * tok_per_step / 1e6:.0f}M (+{D} decay steps): val_full {vf:.4f}")
        return p, st, buf, time.time() - tb, vf
    log_neurons(params, 0)
    t_start = time.time()
    t_mark, tok_mark, wait_acc = None, 0, 0.0
    train_time = 0.0
    for step in range(1, steps_run + 1):
        batch = next_batch()
        if step <= 3:  # round 8: data-stream identity across arms
            import hashlib
            log(event="data_hash", step=step, sha256=hashlib.sha256(np.ascontiguousarray(batch, np.int32).tobytes()).hexdigest())
        batch = jnp.asarray(batch)
        wait_acc += data.wait_s
        if step == 6:  # after compile and warm-up: start the throughput clock
            jax.block_until_ready(params)
            t_mark, tok_mark, wait_acc = time.time(), 0, 0.0
        params, opt_state, loss, gnorm = train_step(params, opt_state, batch, jnp.float32(sched(step - 1)))
        if t_mark is not None:
            tok_mark += tok_per_step
        if step % 20 == 0 or step == steps_run:
            loss_f = float(loss)
            now = time.time()
            rec = dict(event="train", step=step, tokens=step * tok_per_step, loss=loss_f,
                       gnorm=float(gnorm), lr=float(sched(step)), docs=data.docs)
            if t_mark is not None and now > t_mark:
                rec["tok_s"] = tok_mark / (now - t_mark)
                rec["data_wait_ms"] = 1000 * wait_acc / max(1, step - 5)
            ms = jax.devices()[0].memory_stats() or {}
            rec["peak_mem_gb"] = ms.get("peak_bytes_in_use", 0) / 2**30
            log(**rec)
            if step % 100 == 0 or step == steps_run:
                say(f"step {step}/{steps} loss {loss_f:.4f} {rec.get('tok_s', 0):,.0f} tok/s "
                    f"wait {rec.get('data_wait_ms', 0):.1f}ms mem {rec['peak_mem_gb']:.2f}GB")
            if not math.isfinite(loss_f):
                say("non-finite loss; aborting")
                break
        if has_shared(cfg) and a.cos_every and (step % a.cos_every == 0 or step == 1):
            jax.block_until_ready(params)
            tp = time.time()
            cs = grad_cos(params, batch[: a.micro])
            log(event="grad_cos", step=step, tokens=step * tok_per_step,
                **{t: [float(v) for v in cs[t]] for t in SHARED})
            if t_mark is not None:
                t_mark += time.time() - tp  # diagnostics don't count against throughput
        if step in eval_at:
            jax.block_until_ready(params)
            te = time.time()
            vl = evaluate(val[: a.eval_rows])
            if t_mark is not None:
                t_mark += time.time() - te
            log(event="eval", step=step, tokens=step * tok_per_step, val_loss=vl, eval_s=time.time() - te)
            say(f"  eval @ {step * tok_per_step / 1e6:.1f}M tok: val {vl:.4f}")
            log_alpha(params, step)
        if step in neuron_at:
            jax.block_until_ready(params)
            tn = time.time()
            log_neurons(params, step)
            if t_mark is not None:
                t_mark += time.time() - tn
        if step in branches and step < steps:
            jax.block_until_ready(params)
            _, _, buf, bt, _ = run_branch(step, jax.tree.map(jnp.copy, params), jax.tree.map(jnp.copy, opt_state))
            replay = collections.deque(buf + list(replay))
            if t_mark is not None:
                t_mark += bt  # branches don't count against stable-phase throughput
    jax.block_until_ready(params)
    tok_s = tok_mark / (time.time() - t_mark) if t_mark else None
    steps_done, v_last = steps_run, None
    if wsd and steps_run == steps:  # the last branch decays the stable run in place
        params, opt_state, _, _, v_last = run_branch(steps, params, opt_state)
        steps_done = steps + branches[steps]
    wall = time.time() - t_start
    summary = dict(event="final", name=name, arm=a.arm, seed=a.seed, lr=a.lr, steps=steps_done,
                   tokens=steps_done * tok_per_step, wall_s=wall, params=pc, flops_per_token=fl)
    if t_mark:
        summary["tok_s"] = tok_s
    summary["peak_mem_gb"] = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
    if steps_run == steps:
        te = time.time()
        summary["val_loss_full"] = evaluate(val) if v_last is None else v_last
        summary["final_eval_s"] = time.time() - te
    if a.zeroshot and steps_run == steps:
        from eval_zeroshot import run_all
        summary["zeroshot"] = run_all(params, cfg)
    if a.keep_params:
        save_params(params, name)
    log(**summary)
    say(f"done {name}: val {summary.get('val_loss_full', float('nan')):.4f} in {wall / 60:.1f} min")
    logf.close()
    if steps_run == steps:
        open(os.path.join(RUNS, name + ".DONE"), "w").write(json.dumps(summary))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
