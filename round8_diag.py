"""Round 8, step 1: gradient retention and U spectrum of a trained model.

grad_retention(U, S, p, y) = share of ||p - y||^2 inside col([U, (S,) 1]) (heads.grad_retention), on
64 val "batches". A batch here is 256 positions of one val row (row 76*i, every 4th position from 3),
because p - y is [positions x V]; 64 x 256 = 16,384 positions in all, spread over the val set.
p is the model's own distribution: flat softmax over all 50304 rows for tB (S has zero rows for the
47 pad tokens, which belong to no class); exp(heads.hier_logprobs_full) over 50257 for hier arms.
Spectrum: singular values of U = emb[:50257] (tied) or Uw (H2w); spectral entropy H of
sigma / sum(sigma) (Roy & Vetterli), effective rank exp(H).

python round8_diag.py --arm tB --ckpt wsd_B_r8_s0 [--classes kmeans] [--tag B]  -> runs/round8/diag_<tag>.json
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import val_rows
from model import forward
from round8_heads import V_REAL, H, head_inputs, load_params, tables

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--classes", default="kmeans", help="class table for the col([U, S, 1]) basis")
ap.add_argument("--tag", required=True)
ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--pos", type=int, default=256)
a = ap.parse_args()
cfg = ARMS[a.arm]
p = load_params(cfg, a.ckpt)
val = val_rows(1024)
cls, _, _, _ = tables(a.classes, cfg.head_C)
cls_np = np.asarray(cls)

if cfg.head:
    _, U, b, hp = head_inputs(p, jnp.zeros((1, cfg.d)), cfg)
    Vtot = V_REAL
else:
    U, Vtot = p["emb"].astype(jnp.float32), cfg.vocab
S = np.zeros((Vtot, cfg.head_C), np.float32)
S[np.arange(V_REAL), cls_np] = 1.0
S = jnp.asarray(S)


@jax.jit
def probs(rows):
    h = forward(p, rows[:, :-1], cfg)[0]  # [T, d]
    if not cfg.head:
        return jax.nn.softmax(h @ U.T, -1)
    hh, U_, b_, hp_ = head_inputs(p, h, cfg)
    return jnp.exp(H.hier_logprobs_full(hh, U_, b_, hp_["Vc"], hp_["bc"], cls, cfg.head_C))


def basis_q(Ub):  # the Q of heads.grad_retention, formed once instead of per batch
    return jnp.linalg.qr(jnp.concatenate([Ub, jnp.ones((Ub.shape[0], 1))], 1))[0]


@jax.jit
def retention(Q, pp, yo):  # heads.grad_retention with Q hoisted; checked against it on batch 0
    r = (pp - yo).T
    return jnp.sum((Q.T @ r) ** 2) / jnp.sum(r ** 2)


US = jnp.concatenate([U, S], 1)
Q_flat, Q_S = basis_q(U), basis_q(US)
stride = len(val) // a.batches
flat, withS, pad_mass = [], [], []
for i in range(a.batches):
    row = jnp.asarray(val[i * stride: i * stride + 1])
    pp = probs(row)
    idx = jnp.arange(3, 1024, 4)[: a.pos]
    pp = pp[idx]
    y = row[0, 1:][idx]
    yo = jax.nn.one_hot(y, Vtot)
    flat.append(float(retention(Q_flat, pp, yo)))
    withS.append(float(retention(Q_S, pp, yo)))
    pad_mass.append(float(pp[:, V_REAL:].sum(-1).mean()) if Vtot > V_REAL else 0.0)
    if i == 0:
        ref = H.grad_retention(U, None, pp, yo), H.grad_retention(U, S, pp, yo)
        assert abs(ref[0] - flat[0]) < 1e-5 and abs(ref[1] - withS[0]) < 1e-5, (ref, flat[0], withS[0])

sv = np.linalg.svd(np.asarray(U, np.float64), compute_uv=False)
q = sv / sv.sum()
Hs = float(-(q * np.log(q)).sum())
out = {"arm": a.arm, "ckpt": a.ckpt, "classes": a.classes, "batches": a.batches, "positions_per_batch": a.pos,
       "V_basis": int(Vtot), "basis_dims": {"flat": int(U.shape[1]) + 1, "with_classes": int(U.shape[1]) + cfg.head_C + 1},
       "chance_share": {"flat": (U.shape[1] + 1) / Vtot, "with_classes": (U.shape[1] + cfg.head_C + 1) / Vtot},
       "grad_retention_flat": {"mean": float(np.mean(flat)), "std": float(np.std(flat, ddof=1))},
       "grad_retention_with_classes": {"mean": float(np.mean(withS)), "std": float(np.std(withS, ddof=1))},
       "pad_prob_mass_mean": float(np.mean(pad_mass)),
       "per_batch": {"flat": flat, "with_classes": withS},
       "U_singular_values": sv.tolist(), "U_top5": sv[:5].tolist(),
       "U_spectral_entropy": Hs, "U_effective_rank": float(np.exp(Hs)), "U_sigma_max_over_min": float(sv[0] / sv[-1])}
os.makedirs("runs/round8", exist_ok=True)
json.dump(out, open(f"runs/round8/diag_{a.tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("per_batch", "U_singular_values")}, indent=1))
