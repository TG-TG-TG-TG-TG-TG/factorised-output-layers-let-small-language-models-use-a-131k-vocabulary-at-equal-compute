"""Round 10 accounting for every arm (CPU, seconds): parameter split and FLOPs/token from the real init pytrees.

blocks    = the unique blocks' matrices (attention + FFN)
per_copy  = per-application RMSNorm gains, alphas, per-neuron gates (+ the final norm)
embedding = the tied input/output embedding (50,304 x d)
head      = the two-level head's own params (R, bc)
ngram     = hashed n-gram input tables (lookups: no FLOPs)
FLOPs/token = model.flops_per_token (training, 6N rule + attention; head = 3 x head forward), x tB (or x B at d=384).
Optimizer state (AdamW m + v) = 2 x params x 4 bytes; fp32 params = params x 4 bytes.

python round10_acct.py tB tH2atm tH3k8 ...   -> runs/round10/acct.json (merged with earlier entries)
"""
import json
import os
import sys

import jax

import model as M
from arms import ARMS

OUT = "runs/round10/acct.json"
n_ = lambda t: int(sum(x.size for x in jax.tree.leaves(t)))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
FB = {128: M.flops_per_token(ARMS["tB"])["total"], 384: M.flops_per_token(ARMS["B"])["total"]}
for arm in sys.argv[1:]:
    cfg = ARMS[arm]
    p = jax.eval_shape(lambda: M.init_params(cfg, jax.random.PRNGKey(0)))
    fl = M.flops_per_token(cfg)
    r = {"blocks": n_(p["blocks"]), "per_copy": n_(p["copies"]) + n_(p["ln_out"]), "embedding": int(p["emb"].size),
         "head": n_(p.get("head", {})), "ngram": sum(int(p[k].size) for k in p if k.startswith("ng")), "total": n_(p),
         "flops_per_token": fl, "flops_vs_B": fl["total"] / FB[384 if cfg.d == 384 else 128],
         "n_eff": cfg.n_eff, "d": cfg.d, "f": cfg.f, "H": cfg.H, "n_unique": cfg.n_unique}
    r["fp32_param_GB"] = 4 * r["total"] / 2 ** 30
    r["adam_state_GB"] = 8 * r["total"] / 2 ** 30
    r["ngram_param_plus_adam_GB"] = 12 * r["ngram"] / 2 ** 30
    res[arm] = r
    print(arm, {k: v for k, v in r.items() if k != "flops_per_token"}, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
