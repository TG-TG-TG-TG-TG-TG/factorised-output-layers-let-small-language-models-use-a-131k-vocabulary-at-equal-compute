"""Round 8, steps 3-4: param / FLOPs accounting and the gates, on one real batch with each arm's
seed-0 init (fp32 head math; backbone in its normal bf16 compute). CPU is fine.

 1 H2 (and H2r): hier_nll_grouped == hier_nll == -hier_logprobs_full[y] (1e-5); rows of exp(full) sum to 1 (1e-5)
 2 H2: d(mean NLL)/dU, grouped vs gather (1e-5)
 3 H2: hier_greedy == argmax hier_logprobs_full on 256 positions; mean classes expanded
 4 every arm: backbone tensors at init bit-identical to tB's (seed 0)
 5 data stream: sha256 of the first 3 batches of data.Batches(64, seed 0) -> runs/round8/data_hash_ref.json;
   round8_smoke_check.py compares it with the data_hash events every round-8 run logs
 + wiring: model.lm_loss(tH2) (chunked gather = training path) == mean gather == mean grouped NLL; CB: cb_logits(fp32) == heads.class_bias_logits,
   bf16-path deviation reported

python round8_gates.py  -> runs/round8/gates.txt, gates.json
"""
import hashlib
import json
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import Batches
from model import count_params, flops_per_token, forward, init_params, lm_loss
from round8_heads import V_REAL, H, cb_logits, head_flops_fwd, head_inputs, tables

OUT = os.environ.get("ROUND8_OUT", "runs/round8")  # env: dry runs only
lines, res, fails = [], {}, []


def say(s):
    print(s, flush=True)
    lines.append(s)


def gate(name, ok, detail):
    say(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    res[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


data = Batches(64, 1024, seed=0, log=lambda m: None)
bat = [data.next() for _ in range(3)]
hashes = [hashlib.sha256(np.ascontiguousarray(b, np.int32).tobytes()).hexdigest() for b in bat]
json.dump({"sha256_first3": hashes}, open(os.path.join(OUT, "data_hash_ref.json"), "w"), indent=1)
say(f"data stream ref sha256 (first 3 batches): {[h[:16] for h in hashes]}")
rows = jnp.asarray(bat[0][:16])  # one micro-batch of the first real batch: 16 x 1024 targets

arms = ["tB", "tH2", "tH2r", "tCB", "tH2w"]
P = {k: init_params(ARMS[k], jax.random.PRNGKey(0)) for k in arms}
B0 = count_params(P["tB"])
acct = {}
for k in arms:
    cfg = ARMS[k]
    pc = count_params(P[k])
    new = pc["total"] - B0["total"]
    hf = head_flops_fwd(cfg)
    acct[k] = {"total_params": pc["total"], "new_params": new, "head_fwd_flops_per_token": hf,
               "head_ratio_to_B": hf / head_flops_fwd(ARMS["tB"]), "train_flops_per_token": flops_per_token(cfg)}
    say(f"{k}: params {pc['total']:,} (new {new:,}); head fwd FLOPs/token {hf:,} = {acct[k]['head_ratio_to_B']:.4f} x B; "
        f"train FLOPs/token {acct[k]['train_flops_per_token']['total']:,}")
res["accounting"] = acct

# gate 4
for k in arms[1:]:
    bb = {kk: v for kk, v in P[k].items() if kk != "head"}
    la, lb = jax.tree.leaves(bb), jax.tree.leaves(P["tB"])
    same = jax.tree.structure(bb) == jax.tree.structure(P["tB"]) and all(
        x.dtype == y.dtype and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))
    gate(f"4 backbone init bit-identical {k} vs tB", same, f"{len(la)} tensors")

h_all = forward(P["tH2"], rows[:, :-1], ARMS["tH2"]).reshape(-1, 128)
y_all = rows[:, 1:].reshape(-1)
NF = 2048  # positions for the full-vocabulary checks ([2048 x 50257] fp32)

for k in ("tH2", "tH2r"):
    cfg = ARMS[k]
    p = P[k]
    cls, members, mask, pos = tables(cfg.head_classes, cfg.head_C)
    hh, U, b, hp = head_inputs(p, h_all, cfg)
    g = H.hier_nll_grouped(hh, y_all, U, b, hp["Vc"], hp["bc"], cls, members, mask, pos)
    f = H.hier_nll(hh, y_all, U, b, hp["Vc"], hp["bc"], cls, members, mask, pos)
    lp = H.hier_logprobs_full(hh[:NF], U, b, hp["Vc"], hp["bc"], cls, cfg.head_C)
    full = -jnp.take_along_axis(lp, y_all[:NF, None], 1)[:, 0]
    d1 = float(jnp.abs(g - f).max())
    d2 = float(jnp.abs(g[:NF] - full).max())
    d3 = float(jnp.abs(jnp.exp(lp).sum(-1) - 1).max())
    gate(f"1 {k} grouped == gather (16384 tok)", d1 < 1e-5, f"max |diff| {d1:.2e}, mean NLL {float(g.mean()):.5f} (ln V = {np.log(V_REAL):.5f})")
    gate(f"1 {k} grouped == -full[y] ({NF} tok)", d2 < 1e-5, f"max |diff| {d2:.2e}")
    gate(f"1 {k} sum exp(full) == 1", d3 < 1e-5, f"max |sum - 1| {d3:.2e}")
    if k == "tH2":
        lm = float(lm_loss(p, rows, cfg))   # training path = chunked heads.hier_nll (gather)
        gm = float(H.hier_nll(hh, y_all, U, b, hp["Vc"], hp["bc"], cls, members, mask, pos).mean())
        gate("wiring lm_loss(tH2) == mean gather (training path)", abs(lm - gm) < 1e-5, f"{lm:.6f} vs {gm:.6f}")
        gate("wiring lm_loss(tH2) == mean grouped", abs(lm - float(g.mean())) < 1e-5, f"{lm:.6f} vs {float(g.mean()):.6f}")
        gU = jax.grad(lambda U: H.hier_nll_grouped(hh, y_all, U, b, hp["Vc"], hp["bc"], cls, members, mask, pos).mean())(U)
        fU = jax.grad(lambda U: H.hier_nll(hh, y_all, U, b, hp["Vc"], hp["bc"], cls, members, mask, pos).mean())(U)
        dg = float(jnp.abs(gU - fU).max())
        gate("2 tH2 dL/dU grouped == gather", dg < 1e-5, f"max |diff| {dg:.2e} (max |grad| {float(jnp.abs(fU).max()):.2e})")
        ok, ex = 0, []
        am = np.asarray(jnp.argmax(lp[:256], -1))
        for i in range(256):
            w, n = H.hier_greedy(hh[i], U, b, hp["Vc"], hp["bc"], members, mask)
            ok += int(w == am[i])
            ex.append(n)
        gate("3 tH2 hier_greedy == argmax full (256 pos)", ok == 256,
             f"{ok}/256 match; classes expanded mean {np.mean(ex):.2f}, max {max(ex)} (of {cfg.head_C}); init model")
        res["greedy_expanded_init"] = {"mean": float(np.mean(ex)), "max": int(max(ex))}

# CB wiring
cfg = ARMS["tCB"]
p = P["tCB"]
cls, *_ = tables(cfg.head_classes, cfg.head_C)
h = forward(p, rows[:, :-1], cfg).reshape(-1, 128)[:NF]
U = p["emb"][:V_REAL]
ref = H.class_bias_logits(h, U, jnp.zeros(V_REAL), p["head"]["Vg"], p["head"]["ag"], cls)
mine32 = cb_logits(h, U, p["head"]["Vg"], p["head"]["ag"], cls, replace(cfg, compute_dtype="float32"))
mine16 = cb_logits(h, U, p["head"]["Vg"], p["head"]["ag"], cls, cfg)
d32 = float(jnp.abs(ref - mine32).max())
gate("CB cb_logits(fp32) == heads.class_bias_logits", d32 < 1e-5, f"max |diff| {d32:.2e}")
lse = lambda z: jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, y_all[:NF, None], 1)[:, 0]
nll_ref, nll16 = float(lse(ref).mean()), float(lse(mine16).mean())
say(f"[info] CB bf16 path (as B's head): max |logit diff| {float(jnp.abs(ref - mine16).max()):.2e}, mean NLL {nll16:.6f} vs fp32 {nll_ref:.6f}")
# flat B for reference: the same bf16-in / fp32-out product
# 4 rows only: on 16 rows the CPU ragged_dot fallback for tH2w peaked at 22 GB and was OOM-killed (2026-11-06)
r4 = rows[:4]
lm_cb = float(lm_loss(p, r4, cfg))
say(f"[info] lm_loss at init (4 rows): tB {float(lm_loss(P['tB'], r4, ARMS['tB'])):.5f}, tCB {lm_cb:.5f}, "
    f"tH2 {float(lm_loss(P['tH2'], r4, ARMS['tH2'])):.5f}, tH2w {float(lm_loss(P['tH2w'], r4, ARMS['tH2w'])):.5f}")

say(f"\nALL GATES {'OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
res["fails"] = fails
open(os.path.join(OUT, "gates.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "gates.json"), "w"), indent=1)
