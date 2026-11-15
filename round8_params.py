"""Round 8 (2026-11-06, added after the results): parameter count per layer and memory per arm.

Tiny arms: counted from the real init pytree (model.init_params, seed 0), grouped by layer.
600M decode config (round8_decode.py CFG "big"): counted analytically from its shapes, flat vs two-level head.
Memory (analytic, decimal MB):
  training state = params + grads + AdamW m and v, all fp32 = 16 bytes/param (train.py: optax.adamw on
                   fp32 params; weights are cast to bf16 per matmul, not stored twice)
  inference      = bf16 weights = 2 bytes/param (round8_decode.py passes bf16 weights)
The measured training peak VRAM per arm is in runs/round8/smoke_check.json (200-step smoke runs);
round8_analyze.py puts both side by side.

JAX_PLATFORMS=cpu python round8_params.py  -> runs/round8/params_memory.json
"""
import json
import os

import jax

from arms import ARMS
from model import init_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "round8", "params_memory.json")
ARM_LIST = ["tB", "tH2", "tH2r", "tCB", "tH2w"]


def group_of(path):
    k = [getattr(p, "key", getattr(p, "idx", None)) for p in path]
    if k[0] == "emb":
        return "embedding (tied: input + flat output)", None
    if k[0] == "ln_out":
        return "final norm", None
    if k[0] == "head":
        return f"head: {k[1]}", None
    if k[0] == "blocks":
        part = "attention (Wq Wk Wv Wo)" if k[2] in ("Wq", "Wk", "Wv", "Wo") else "FFN (Wup Wdown)"
        return f"block {k[1]}", part
    if k[0] == "copies":
        return f"block {k[1]}", "per-copy norms + alphas"
    return "/".join(map(str, k)), None


def tiny(arm):
    cfg = ARMS[arm]
    p = init_params(cfg, jax.random.PRNGKey(0))
    rows = {}
    for path, v in jax.tree_util.tree_flatten_with_path(p)[0]:
        g, part = group_of(path)
        r = rows.setdefault(g, {"total": 0, "parts": {}})
        r["total"] += int(v.size)
        if part:
            r["parts"][part] = r["parts"].get(part, 0) + int(v.size)
    total = sum(r["total"] for r in rows.values())
    return {"config": {"d": cfg.d, "f": cfg.f, "H": cfg.H, "layers": cfg.n_unique, "vocab": cfg.vocab,
                       "head": cfg.head or "flat", "C": cfg.head_C if cfg.head else None,
                       "head_wide": cfg.head_wide or None},
            "layers": rows, "total": total,
            "train_state_MB": 16 * total / 1e6, "inference_bf16_MB": 2 * total / 1e6}


def big():
    """round8_decode.py CFG big: 28 x d1280, GQA 20 q / 4 kv heads, hd 64, SwiGLU 3456, tied V 65536,
    RMSNorm weights (2 per layer + final). Two-level head adds Vc (d x C) + bc (C), C = 1024."""
    L, d, hq, hkv, hd, f, V, C = 28, 1280, 20, 4, 64, 3456, 65536, 1024
    attn = d * hq * hd + 2 * d * hkv * hd + hq * hd * d
    ffn = 3 * d * f
    norms = 2 * d
    per_layer = attn + ffn + norms
    emb = V * d
    out = {}
    for head, extra in (("flat", 0), ("two-level C=1024", d * C + C)):
        total = L * per_layer + emb + d + extra
        out[head] = {"per_layer": {"attention": attn, "FFN (SwiGLU)": ffn, "norms": norms, "total": per_layer},
                     "layers": L, "embedding (tied)": emb, "final norm": d, "head extra": extra, "total": total,
                     "inference_bf16_MB": 2 * total / 1e6, "train_state_MB": 16 * total / 1e6,
                     # per decode step at batch 1 the head reads: flat all of E (V x d); two-level Vc + one class
                     "head_weights_read_per_token_MB_bf16": 2 * (V * d if extra == 0 else d * C + 64 * d) / 1e6,
                     "kv_cache_MB_per_1k_tokens_bf16": 2 * L * 2 * hkv * hd * 1000 / 1e6}
    return out


def main():
    res = {"tiny": {a: tiny(a) for a in ARM_LIST}, "big_600M": big()}
    json.dump(res, open(OUT, "w"), indent=1)
    for a, r in res["tiny"].items():
        print(f"{a:5s} total {r['total']:>11,}  train state {r['train_state_MB']:7.1f} MB  bf16 {r['inference_bf16_MB']:6.1f} MB")
    for h, r in res["big_600M"].items():
        print(f"600M {h:17s} total {r['total']:>13,}  bf16 {r['inference_bf16_MB']:7.1f} MB  "
              f"head read/token {r['head_weights_read_per_token_MB_bf16']:.2f} MB")


if __name__ == "__main__":
    main()
