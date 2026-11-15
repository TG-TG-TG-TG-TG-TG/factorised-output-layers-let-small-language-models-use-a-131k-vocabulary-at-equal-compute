"""Round 10: per-frequency-bucket Δ vs B_r8 (round-9 A1 style) for two-level-head checkpoints (GPU, minutes).

Val set and B exactly as round9_partA.py (4,880 rows x 1,024 targets; B's NLL renormalised over the 50,257 real
tokens; bf16 h and E in, fp32 logits out). For each arm, every target y is split by THAT arm's partition:
  arm  top = -log p_top(slot(y) | h)  (singleton or tail cluster), within = -log p(y | cluster, h) (0 for singletons)
  B    top = LSE_real z - LSE_{w in group(y)} z  (group = {y} for the arm's singletons, the arm's cluster otherwise)
       within = LSE_{group(y)} z - z_y        so B_top + B_within = B's NLL for every arm
Δtop = arm top - B top, Δwithin = arm within - B within, per bucket of target rank (runs/round9/token_rank.npy):
[0,256) [256,1024) [1024,4096) [4096,16384) [16384,V). 'contrib' = bucket share x bucket mean Δ.
Per-bucket totals let Part D take Δ vs H3* (same targets, so Δ of bucket means = bucket mean of Δ).

python round10_buckets.py --tag partA --arms H2atm:tH2atm:wsd_H2atm_s0,H3k8:tH3k8:wsd_H3k8_s0,...
   (a checkpoint name without '/' is looked up in ~/lm-data/ckpt)        -> runs/round10/buckets_<tag>.json
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import val_rows
from model import forward
from round8_heads import V_REAL, load_params
import round9_heads as R9

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--arms", required=True, help="label:arm:ckpt,...")
a = ap.parse_args()
F32, BF = jnp.float32, jnp.bfloat16
val = val_rows(1024)[:4880]
rank = np.load("runs/round9/token_rank.npy")
EDGES = [0, 256, 1024, 4096, 16384, V_REAL]
BNAMES = [f"[{EDGES[i]},{EDGES[i + 1] if i < 4 else 'V'})" for i in range(5)]
CH = 4096
Y = val[:, 1:].reshape(-1)
bucket = np.searchsorted(EDGES, rank[Y], side="right") - 1
ARMLIST = [x.split(":") for x in a.arms.split(",") if x]


def run(f):
    return np.concatenate([np.asarray(f(jnp.asarray(val[i:i + 16]))) for i in range(0, len(val), 16)], 1)


# B_r8: nll_all (logged metric), nll (real rows), and for each arm's partition its LSE over group(y)
pB = load_params(ARMS["tB"], "wsd_B_r8_s0")
cB = ARMS["tB"]
E = pB["emb"].astype(BF)
TBS = [R9.tables(ARMS[arm]) for _, arm, _ in ARMLIST]


@jax.jit
def b_parts(rows):
    h = forward(pB, rows[:, :-1], cB).reshape(-1, cB.d)
    y = rows[:, 1:].reshape(-1)

    def piece(hc, yc):
        z = jnp.einsum("nd,vd->nv", hc.astype(BF), E, preferred_element_type=F32)
        zy = jnp.take_along_axis(z, yc[:, None], 1)[:, 0]
        zr = z[:, :V_REAL]
        lse = jax.nn.logsumexp(zr, -1)
        out = [jax.nn.logsumexp(z, -1) - zy, lse - zy]
        for slot, tail, grp, members, mask, pos, _, _ in TBS:
            c = grp[yc]
            zc = jnp.where(mask[c], jnp.take_along_axis(zr, members[c], 1), -jnp.inf)
            lse_g = jnp.where(tail[yc], jax.nn.logsumexp(zc, -1), zy)
            out.append(lse - lse_g)   # B top part under this arm's partition
        return jnp.stack(out)
    return jax.lax.map(lambda xs: piece(*xs), (h.reshape(-1, CH, cB.d), y.reshape(-1, CH))).transpose(1, 0, 2).reshape(2 + len(TBS), -1)


Bp = run(b_parts)
b_all, b_tot = Bp[0], Bp[1]
res = {"n_targets": int(Y.size), "buckets": BNAMES, "bucket_share": [float((bucket == k).mean()) for k in range(5)],
       "B": {"nll_all_mean": float(b_all.mean()), "logged": 4.122812896478371, "total": float(b_tot.mean()),
             "total_bucket": [float(b_tot[bucket == k].mean()) for k in range(5)]}, "arms": {}}
print(res["B"], flush=True)


def hier_parts(p, cfg):
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, _, _ = tb

    @jax.jit
    def f(rows):
        h = forward(p, rows[:, :-1], cfg).reshape(-1, cfg.d)
        y = rows[:, 1:].reshape(-1)
        Wt, bt = R9.top_matrix(p, cfg, tb)
        Uw = R9.word_matrix(p, cfg)

        def piece(hc, yc):
            lt = -jnp.take_along_axis(jax.nn.log_softmax(hc @ Wt.T + bt), slot[yc][:, None], 1)[:, 0]
            tot = R9.hier9_nll(hc, yc, Uw, Wt, bt, slot, tail, grp, members, mask, pos)
            return jnp.stack([tot, lt])
        return jax.lax.map(lambda xs: piece(*xs), (h.reshape(-1, CH, cfg.d), y.reshape(-1, CH))).transpose(1, 0, 2).reshape(2, -1)
    return f


for i, (lab, arm, ck) in enumerate(ARMLIST):
    cfg = ARMS[arm]
    p = load_params(cfg, ck)
    X = run(hier_parts(p, cfg))
    tot, top = X[0], X[1]
    win = tot - top
    b_top = Bp[2 + i]
    b_win = b_tot - b_top
    d, dt, dw = tot - b_tot, top - b_top, win - b_win
    sel = [bucket == k for k in range(5)]
    res["arms"][lab] = {
        "arm": arm, "ckpt": ck, "total": float(tot.mean()), "d_total": float(d.mean()),
        "top": float(top.mean()), "within": float(win.mean()), "d_top": float(dt.mean()), "d_within": float(dw.mean()),
        "total_bucket": [float(tot[s].mean()) for s in sel], "d_total_bucket": [float(d[s].mean()) for s in sel],
        "d_top_bucket": [float(dt[s].mean()) for s in sel], "d_within_bucket": [float(dw[s].mean()) for s in sel],
        "contrib_bucket": [float(d[s].sum() / Y.size) for s in sel],
        "B_top_bucket": [float(b_top[s].mean()) for s in sel], "B_within_bucket": [float(b_win[s].mean()) for s in sel]}
    print(lab, json.dumps({k: v for k, v in res["arms"][lab].items() if not k.startswith("B_")}), flush=True)
    del p, X
json.dump(res, open(f"runs/round10/buckets_{a.tag}.json", "w"), indent=1)
