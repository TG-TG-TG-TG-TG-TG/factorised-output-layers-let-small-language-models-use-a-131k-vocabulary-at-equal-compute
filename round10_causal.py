"""Round 10 leak check for the n-gram arms (CPU, seconds): nothing at position t may depend on tokens after t.

For tD1 / tD2 with random (non-zero) n-gram tables and the real init otherwise, on 4 val rows of 256 tokens:
  (a) ngram_ids(tokens)[:, :t+1] is unchanged when every token after t is replaced (t = 0, 17, 128, 254)
  (b) the hidden states h[:, :t+1] from forward() are unchanged under the same replacement (max |diff| == 0)
  (c) and they DO change at t+1 (the check is live)
  (d) the trained D1 / D1_lazy checkpoints (if present in ~/lm-data/ckpt) pass (b) as well
python round10_causal.py   (JAX_PLATFORMS=cpu)  -> runs/round10/causal.txt
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from arms import ARMS
from data import val_rows

rows = jnp.asarray(val_rows(1024)[:4, :256])
rng = np.random.default_rng(0)
out = []


def check(tag, p, cfg):
    ok_all = True
    for t in (0, 17, 128, 254):
        alt = np.asarray(rows).copy()
        alt[:, t + 1:] = rng.integers(0, 50257, size=alt[:, t + 1:].shape)
        alt = jnp.asarray(alt)
        h0, h1 = M.forward(p, rows, cfg), M.forward(p, alt, cfg)
        pre = float(jnp.abs(h0[:, :t + 1] - h1[:, :t + 1]).max())
        post = float(jnp.abs(h0[:, t + 1] - h1[:, t + 1]).max())
        ids_ok = all(bool(jnp.all(M.ngram_ids(rows, n, cfg)[:, :t + 1] == M.ngram_ids(alt, n, cfg)[:, :t + 1]))
                     for n in range(2, cfg.ngram + 1))
        ok = pre == 0.0 and post > 0 and ids_ok
        ok_all &= ok
        out.append(f"[{'OK' if ok else 'FAIL'}] {tag} t={t}: ids prefix same {ids_ok}, max|dh| up to t {pre:.1e}, at t+1 {post:.2e}")
        print(out[-1], flush=True)
    return ok_all


ok = True
for arm in ("tD1", "tD2"):
    cfg = ARMS[arm]
    p = M.init_params(cfg, jax.random.PRNGKey(0))
    p = dict(p, **{k: 0.05 * jax.random.normal(jax.random.PRNGKey(3 + i), p[k].shape) for i, k in enumerate(sorted(p)) if k.startswith("ng")})
    ok &= check(f"{arm} init+random tables", p, cfg)
try:
    from round8_heads import load_params
    for ck in ("wsd_D1_s0", "wsd_D1_lazy_s0"):
        if os.path.exists(os.path.expanduser(f"~/lm-data/ckpt/{ck}.npz")):
            ok &= check(f"{ck} trained", load_params(ARMS["tD1"], ck), ARMS["tD1"])
except Exception as e:  # report, don't hide
    out.append(f"[FAIL] trained-checkpoint check raised {type(e).__name__}: {e}")
    ok = False
out.append("ALL CAUSAL CHECKS OK" if ok else "CAUSAL CHECK FAILED")
print(out[-1])
open("runs/round10/causal.txt", "w").write("\n".join(out) + "\n")
