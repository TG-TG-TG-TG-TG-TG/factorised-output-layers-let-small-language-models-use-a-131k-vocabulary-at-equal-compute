"""Round 9, Part A1 + A3 on the round-8 finals (GPU, ~minutes). A2 (decay gains) is read from the logs in
round9_analyze.py.

Val set: the same 4,880 rows x 1,024 targets (4,997,120 targets) that train.py's full eval averages over
(305 micro-batches of 16; the last 2 of the 4,882 cached rows are never evaluated), so the totals must
reproduce the logged val_loss_full (B_r8 4.1228, H2 4.2137).

A1, per val target y, class c = cls[y] (runs/round8/classes_kmeans1024.npy):
  B  (eval precision: bf16 h and E in, fp32 logits out)
     nll_all  = LSE over all 50,304 rows - z_y              (= the logged metric)
     nll      = LSE over the 50,257 real rows - z_y          (renormalized without the 47 pad rows; reported)
     class    = -(LSE_{w in c} z - LSE_real z),  within = nll - class = LSE_{w in c} z - z_y
     entropy  = entropy of the renormalized real-token distribution (terciles of it)
  H2 class = -log_softmax(h Vc + bc)[c], within = -log softmax over the class's words (fp32, heads.hier_nll's terms)
  extra arms (optional, any hier/hier9 checkpoint): total NLL per token, for the per-bucket Δtotal
Buckets: rank of y by training-set count (runs/round9/token_rank.npy): [0,256) [256,1024) [1024,4096) [4096,16384) [16384,V).

A3 on B_r8 and H2 finals:
  RMS of U = emb[:50257] entries per frequency bucket
  U = M + W with M = each row replaced by its k-means class mean: ||U||_F, ||M||_F, ||W||_F
  64 val batches (batch i = val rows 76i .. 76i+3, 4 x 1024 tokens): RMS of the input embedding x0 = emb[tokens],
  and of block 0's output after its alphas: a_attn*attn, a_ffn*ffn, and x1 - x0 (their sum)
  every per-copy alpha (mean |a|, min, max over the d channels)

python round9_partA.py [--extra H2tm:wsd_H2tm_s0,...]   -> runs/round9/partA.json
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import val_rows
from model import attention, ffn, forward, rmsnorm
from round8_heads import V_REAL, load_params, tables as tables8
import round9_heads as R9

ap = argparse.ArgumentParser()
ap.add_argument("--extra", default="", help="label:arm:ckpt,... more hier arms for per-bucket totals")
a = ap.parse_args()
F32, BF = jnp.float32, jnp.bfloat16
val = val_rows(1024)[:4880]
rank = np.load("runs/round9/token_rank.npy")
EDGES = [0, 256, 1024, 4096, 16384, V_REAL]
BNAMES = [f"[{EDGES[i]},{EDGES[i + 1] if i < 4 else 'V'})" for i in range(5)]
cls8, members8, mask8, _ = tables8("kmeans", 1024)
CH = 4096


def b_parts(p, cfg):
    E = p["emb"].astype(BF)

    @jax.jit
    def f(rows):
        h = forward(p, rows[:, :-1], cfg).reshape(-1, cfg.d)
        y = rows[:, 1:].reshape(-1)

        def piece(hc, yc):
            z = jnp.einsum("nd,vd->nv", hc.astype(BF), E, preferred_element_type=F32)
            zy = jnp.take_along_axis(z, yc[:, None], 1)[:, 0]
            lse_all = jax.nn.logsumexp(z, -1)
            zr = z[:, :V_REAL]
            lse = jax.nn.logsumexp(zr, -1)
            c = cls8[yc]
            zc = jnp.where(mask8[c], jnp.take_along_axis(zr, members8[c], 1), -jnp.inf)
            lse_c = jax.nn.logsumexp(zc, -1)
            pr = jax.nn.softmax(zr, -1)
            ent = lse - jnp.sum(pr * zr, -1)
            return jnp.stack([lse_all - zy, lse - zy, lse - lse_c, lse_c - zy, ent])

        return jax.lax.map(lambda xs: piece(*xs), (h.reshape(-1, CH, cfg.d), y.reshape(-1, CH))).transpose(1, 0, 2).reshape(5, -1)
    return f


def hier_parts(p, cfg):
    """(total, top/class part, within part) per token; hier (round 8) or hier9."""
    if cfg.head == "hier9":
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
                return jnp.stack([tot, lt, tot - lt])
            return jax.lax.map(lambda xs: piece(*xs), (h.reshape(-1, CH, cfg.d), y.reshape(-1, CH))).transpose(1, 0, 2).reshape(3, -1)
        return f
    cls, members, mask, pos = tables8(cfg.head_classes, cfg.head_C)
    U = p["emb"][:V_REAL].astype(F32)
    Vc, bc = p["head"]["Vc"], p["head"]["bc"]

    @jax.jit
    def f(rows):
        h = forward(p, rows[:, :-1], cfg).reshape(-1, cfg.d)
        y = rows[:, 1:].reshape(-1)

        def piece(hc, yc):
            c = cls[yc]
            lc = -jnp.take_along_axis(jax.nn.log_softmax(hc @ Vc + bc), c[:, None], 1)[:, 0]
            zc = jnp.where(mask[c], jnp.einsum("bd,bld->bl", hc, U[members[c]]), -jnp.inf)
            lw = -jnp.take_along_axis(jax.nn.log_softmax(zc, -1), pos[yc][:, None], 1)[:, 0]
            return jnp.stack([lc + lw, lc, lw])
        return jax.lax.map(lambda xs: piece(*xs), (h.reshape(-1, CH, cfg.d), y.reshape(-1, CH))).transpose(1, 0, 2).reshape(3, -1)
    return f


def run(f, n_out):
    out = [np.asarray(f(jnp.asarray(val[i:i + 16]))) for i in range(0, len(val), 16)]
    return np.concatenate(out, 1)


Y = val[:, 1:].reshape(-1)
bucket = np.searchsorted(EDGES, rank[Y], side="right") - 1
res = {"n_targets": int(Y.size), "buckets": BNAMES,
       "bucket_share": [float((bucket == k).mean()) for k in range(5)]}
pB = load_params(ARMS["tB"], "wsd_B_r8_s0")
B = run(b_parts(pB, ARMS["tB"]), 5)
pH = load_params(ARMS["tH2"], "wsd_H2_s0")
Hh = run(hier_parts(pH, ARMS["tH2"]), 3)
res["check"] = {"B_nll_all_mean": float(B[0].mean()), "B_logged": 4.122812896478371,
                "H2_mean": float(Hh[0].mean())}
print(res["check"], flush=True)
b_tot, b_cls, b_win, ent = B[1], B[2], B[3], B[4]
h_tot, h_cls, h_win = Hh


def stats(sel):
    return {"share": float(sel.mean()), "B_total": float(b_tot[sel].mean()), "B_class": float(b_cls[sel].mean()),
            "B_within": float(b_win[sel].mean()), "H2_total": float(h_tot[sel].mean()), "H2_class": float(h_cls[sel].mean()),
            "H2_within": float(h_win[sel].mean()), "d_total": float((h_tot - b_tot)[sel].mean()),
            "d_class": float((h_cls - b_cls)[sel].mean()), "d_within": float((h_win - b_win)[sel].mean()),
            # contribution to the overall mean Δ (share x mean Δ)
            "contrib_total": float((h_tot - b_tot)[sel].sum() / Y.size)}


res["A1_all"] = stats(np.ones(Y.size, bool))
res["A1_all"]["B_pad_effect"] = float((B[0] - B[1]).mean())
res["A1_bucket"] = [stats(bucket == k) for k in range(5)]
q = np.quantile(ent, [1 / 3, 2 / 3])
terc = np.searchsorted(q, ent)
res["entropy_tercile_edges"] = q.tolist()
res["A1_entropy"] = [dict(stats(terc == k), entropy_mean=float(ent[terc == k].mean())) for k in range(3)]
# per-bucket totals of extra arms (fix arms), Δ vs B
res["extra"] = {}
for item in [x for x in a.extra.split(",") if x]:
    lab, arm, ck = item.split(":")
    pe = load_params(ARMS[arm], ck)
    X = run(hier_parts(pe, ARMS[arm]), 3)
    res["extra"][lab] = {"total": float(X[0].mean()), "d_total": float((X[0] - b_tot).mean()),
                         "d_total_bucket": [float((X[0] - b_tot)[bucket == k].mean()) for k in range(5)],
                         "top_part": float(X[1].mean()), "within_part": float(X[2].mean())}
    print(lab, res["extra"][lab], flush=True)
    del pe, X

# ---- A3
A3 = {}
for lab, arm, p in (("B_r8", "tB", pB), ("H2", "tH2", pH)):
    cfg = ARMS[arm]
    U = np.asarray(p["emb"][:V_REAL], np.float64)
    r = {"rms_bucket": [float(np.sqrt((U[(np.searchsorted(EDGES, rank[:V_REAL], side="right") - 1) == k] ** 2).mean()))
                        for k in range(5)]}
    c = np.asarray(cls8)
    M = np.zeros((1024, U.shape[1]))
    np.add.at(M, c, U)
    M /= np.bincount(c, minlength=1024)[:, None]
    Mr = M[c]
    r["fro_U"], r["fro_classmean"], r["fro_within"] = (float(np.linalg.norm(x)) for x in (U, Mr, U - Mr))
    r["classmean_share_of_sq"] = r["fro_classmean"] ** 2 / r["fro_U"] ** 2
    b0, c0 = p["blocks"][0], p["copies"][0]

    @jax.jit
    def blk0(rows):
        x = p["emb"][rows]
        W = jax.tree.map(lambda w: w.astype(cfg.cdt), b0)
        ua = c0["a_attn"] * attention(rmsnorm(x, c0["ln_attn"]).astype(cfg.cdt), W["Wq"], W["Wk"], W["Wv"], W["Wo"], cfg).astype(F32)
        x1 = x + ua
        uf = c0["a_ffn"] * ffn(rmsnorm(x1, c0["ln_ffn"]).astype(cfg.cdt), W["Wup"], W["Wdown"], cfg).astype(F32)
        ms = lambda t: jnp.mean(t * t)
        return jnp.stack([ms(x), ms(ua), ms(uf), ms(ua + uf)])
    acc = np.mean([np.asarray(blk0(jnp.asarray(val[76 * i: 76 * i + 4, :-1]))) for i in range(64)], 0)
    r["rms_input_emb"], r["rms_blk0_attn_after_alpha"], r["rms_blk0_ffn_after_alpha"], r["rms_blk0_total"] = (float(np.sqrt(v)) for v in acc)
    r["blk0_over_emb"] = r["rms_blk0_total"] / r["rms_input_emb"]
    r["alphas"] = [{k: {"mean_abs": float(jnp.mean(jnp.abs(cp[k]))), "min": float(cp[k].min()), "max": float(cp[k].max())}
                    for k in ("a_attn", "a_ffn")} for cp in p["copies"]]
    A3[lab] = r
res["A3"] = A3
json.dump(res, open("runs/round9/partA.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "A3"}, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "alphas"} for k, v in A3.items()}, indent=1))
