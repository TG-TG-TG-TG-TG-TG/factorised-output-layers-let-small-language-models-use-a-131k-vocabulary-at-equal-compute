"""Round 12 P5 measurement (paper, theory section): the Jensen gap of the target's cluster on val contexts.

For a tail target y with cluster c = c(y) and member scores s_w = u_w . h (tied heads, no word bias, so b_w = 0):
  Delta_c(h) = LSE_{w in c}(s_w) - log n_c - mean_{w in c}(s_w)   (>= 0, = 0 iff all s_w in c are equal)
  offset_c(h) = z_c(h) - log n_c - mean_{w in c}(s_w), z_c = (ubar_c + R_c) . h + b_c the trained tied-mean cluster logit
  error_c(h) = z_c(h) - LSE_{w in c}(s_w) = offset - Delta  (what the top-level logit gets wrong vs the exact cluster mass)
Reported per GPT-2 rank bucket (round 9's token_rank.npy; [0,256) ... [16384,V)) on the first 1,000 FineWeb-Edu val rows,
and on E5a (round 11's copy task: [EOT] + 512 val tokens + the same 512 again, the first 2,000 rows) for copy 1 vs copy 2.
Frequent (singleton) targets have no cluster: counted in n_frequent, excluded from the means.

python round12_jensen.py [labels]   (default H3k8c4,H3s384_575M) -> runs/round12/jensen.json
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import round11_eval as E
import round9_heads as R9

OUT = os.path.join(E.HERE, "runs", "round12")
F32 = jnp.float32


def gaps(m, X, bs=16):
    """per-target (Delta, offset, error, is_tail) arrays [n_rows, T] for rows X [n, T+1]."""
    cfg, p = m.cfg, m.params
    tb = R9.tables(cfg)
    slot, tail, grp, members, mask, pos, freq, count = tb
    Wt, bt = R9.top_matrix(p, cfg, tb)
    U = R9.word_matrix(p, cfg)
    K = cfg.head_adapt

    @jax.jit
    def f(x, y):
        h = m._h(p, x).astype(F32)                                   # [b, T, d]
        c = grp[y]
        mem, mk = members[c], mask[c]                                # [b, T, L]
        s = jnp.einsum("btd,btld->btl", h, U[mem])
        n = mk.sum(-1)
        lse = jax.nn.logsumexp(jnp.where(mk, s, -jnp.inf), -1)
        mean = jnp.where(mk, s, 0.0).sum(-1) / n
        dl = lse - jnp.log(n) - mean
        zc = jnp.einsum("btd,btd->bt", h, Wt[K + c]) + bt[K + c]
        off = zc - jnp.log(n) - mean
        return dl, off, zc - lse, tail[y]

    out = [[], [], [], []]
    for i in range(0, len(X), bs):
        r = [np.asarray(a) for a in f(jnp.asarray(X[i:i + bs, :-1], jnp.int32), jnp.asarray(X[i:i + bs, 1:], jnp.int32))]
        for k in range(4):
            out[k].append(r[k])
    return [np.concatenate(o) for o in out]


def summarise(dl, off, err, tl, ranks):
    b = np.searchsorted(E.EDGES, ranks, side="right") - 1
    rec = {"n": int(tl.size), "n_tail": int(tl.sum()), "n_frequent": int((~tl).sum()),
           "delta_mean_tail": float(dl[tl].mean()), "delta_std_tail": float(dl[tl].std()),
           "offset_mean_tail": float(off[tl].mean()), "error_mean_tail": float(err[tl].mean()), "error_std_tail": float(err[tl].std()),
           "corr_offset_delta": float(np.corrcoef(off[tl], dl[tl])[0, 1]),
           "delta_per_target_all": float(np.where(tl, dl, 0.0).mean()), "buckets": {}}
    for i, nm in enumerate(E.BNAMES):
        s = (b == i) & tl
        rec["buckets"][nm] = {"n_tail": int(s.sum()), "n": int((b == i).sum()),
                              "delta_mean": float(dl[s].mean()) if s.any() else None,
                              "error_mean": float(err[s].mean()) if s.any() else None,
                              "error_abs_mean": float(np.abs(err[s]).mean()) if s.any() else None}
    return rec


def main(labels):
    fw = E.corpus("fineweb")
    res = {}
    for lab in labels:
        m = E.Model(lab)
        assert m.gpt2 and m.cfg.head == "hier9" and not m.cfg.head_rescore, lab
        X = fw[:1000]
        dl, off, err, tl = gaps(m, X)
        r = {"val": summarise(dl.ravel(), off.ravel(), err.ravel(), tl.ravel(), E.RANK[X[:, 1:]].ravel())}
        base = fw[:2000, 1:513]
        Xc = np.concatenate([np.full((len(base), 1), E.EOT), base, base], 1)
        dl, off, err, tl = gaps(m, Xc)
        rk = E.RANK[Xc[:, 1:]]
        r["E5a_copy1"] = summarise(dl[:, :512].ravel(), off[:, :512].ravel(), err[:, :512].ravel(), tl[:, :512].ravel(), rk[:, :512].ravel())
        r["E5a_copy2"] = summarise(dl[:, 512:].ravel(), off[:, 512:].ravel(), err[:, 512:].ravel(), tl[:, 512:].ravel(), rk[:, 512:].ravel())
        res[lab] = r
        v, c1, c2 = r["val"], r["E5a_copy1"], r["E5a_copy2"]
        print(f"[jensen] {lab}: val Delta {v['delta_mean_tail']:.3f} (std {v['delta_std_tail']:.3f}), offset {v['offset_mean_tail']:.3f}, "
              f"error {v['error_mean_tail']:+.3f} (std {v['error_std_tail']:.3f}); E5a Delta copy1 {c1['delta_mean_tail']:.3f} "
              f"copy2 {c2['delta_mean_tail']:.3f}; by bucket " +
              " ".join(f"{k} {b['delta_mean']:.3f}" for k, b in v["buckets"].items() if b["delta_mean"] is not None), flush=True)
    json.dump(res, open(os.path.join(OUT, "jensen.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["H3k8c4", "H3s384_575M"])
