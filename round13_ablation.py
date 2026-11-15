"""Round 13 (2026-11-12): compact design-choice ablation table from existing rounds 8-11 runs (no new runs; the reviewer
asked for one table with the arm names decoded). Every row is a d=128 model (6 unique blocks, the tiny B backbone) on the
same data order, read at the 1,200M-stable-token branch (1,320M tokens incl. the 10% decay) and at 600M, so the rows are
at EQUAL TOKENS, not equal FLOPs (each head's FLOPs are shown). BPB = logged val_loss_full x BPN (round 9). Seeds are
listed per row; where two seeds exist the mean is used and the spread shown. Each row changes one thing against the row
named in "vs"; nothing here was chosen after these numbers (all runs are rounds 8-11, read as logged).

python round13_ablation.py  -> runs/round13/ablation.md, ablation.json
"""
import json
import os

import numpy as np

from round9_analyze import BPN, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round13")

# key: (plain name, what changes, compared against, [(round, run name), ...])
ROWS = [
    ("flat", "flat tied softmax over all 50,257 tokens", None, [(8, "wsd_B_r8_s0"), (11, "wsd_B_s1")]),
    ("random", "two-level, 1,024 random clusters, no frequent list, free cluster vectors", None, [(8, "wsd_H2r_s0")]),
    ("kmeans", "clusters by balanced k-means on a trained embedding", "random", [(8, "wsd_H2_s0")]),
    ("kmeans+tied", "cluster logit tied to the members' mean embedding", "kmeans", [(9, "wsd_H2tm_s0")]),
    ("kmeans+freq4k", "4,096 most frequent tokens as top-level singletons", "kmeans", [(9, "wsd_H2a_s0")]),
    ("freq4k+tied", "both: frequent singletons and tied-mean cluster logits", "kmeans+freq4k", [(9, "wsd_H2atm_s0")]),
    ("freq8k+tied", "8,192 frequent singletons", "freq4k+tied", [(10, "wsd_H3k8_s0"), (10, "wsd_H3k8_s1")]),
    ("freq4k+tied+4kcl", "4,096 tail clusters instead of 1,024", "freq4k+tied", [(10, "wsd_H3c4_s0")]),
    ("final", "8,192 singletons and 4,096 tail clusters (the head used at d=384)", "freq8k+tied",
     [(10, "wsd_H3k8c4_s0"), (10, "wsd_H3k8c4_s1")]),
    ("final+rescore16", "exact re-scoring of the 16 top clusters", "final", [(11, "wsd_H3rs16_s0"), (11, "wsd_H3rs16_s1")]),
]
POINTS = (600_000_000, 1_200_000_000)


def branch(run, tok, tol=0.002):
    ks = [k for k in run["branch"] if abs(k - tok) <= tol * tok]
    return run["branch"][ks[0]] if ks else None


def main():
    res = {}
    for key, what, vs, runs in ROWS:
        r = {"what": what, "vs": vs, "runs": [n for _, n in runs], "points": {}}
        for tok in POINTS:
            bp, fl = [], []
            for rnd, name in runs:
                run = load_run(os.path.join(HERE, "runs", f"round{rnd}", name + ".jsonl"))
                b = branch(run, tok)
                assert b is not None, (name, tok)
                bp.append(b["loss"] * BPN)
                fl.append(run["final"]["flops_per_token"]["total"] * b["tokens"])
            r["points"][str(tok)] = {"bpb": bp, "mean": float(np.mean(bp)), "spread": float(max(bp) - min(bp)) if len(bp) > 1 else None,
                                     "flops": float(np.mean(fl))}
        res[key] = r
    for key, r in res.items():
        for tok in map(str, POINTS):
            p = r["points"][tok]
            p["delta_vs"] = None if r["vs"] is None else p["mean"] - res[r["vs"]]["points"][tok]["mean"]
            p["delta_vs_flat"] = p["mean"] - res["flat"]["points"][tok]["mean"]
    json.dump(res, open(os.path.join(OUT, "ablation.json"), "w"), indent=1)
    sp = [p["spread"] for r in res.values() for p in r["points"].values() if p["spread"] is not None]
    L = ["# Design-choice ablation (d=128, equal tokens, rounds 8-11 runs)", "",
         "BPB at the 1,200M-stable-token branch (600M in brackets). Δ = this row minus the row in 'vs' (two-seed means "
         "where two seeds exist). Equal tokens, not equal FLOPs: the flat row costs ~2.4-4x the two-level rows' FLOPs.", "",
         "| row | change | vs | seeds | BPB 1200M (600M) | Δ vs 1200M (600M) | FLOPs 1200M |", "|---|---|---|---|---|---|---|"]
    for key, r in res.items():
        a, b = r["points"][str(POINTS[1])], r["points"][str(POINTS[0])]
        dv = "—" if a["delta_vs"] is None else f"{a['delta_vs']:+.4f} ({b['delta_vs']:+.4f})"
        L.append(f"| {key} | {r['what']} | {r['vs'] or '—'} | {len(r['runs'])} | {a['mean']:.4f} ({b['mean']:.4f}) | {dv} | {a['flops']:.2e} |")
    L += ["", f"Seed spread where two seeds exist: {min(sp):.4f}-{max(sp):.4f} BPB, so a one-seed difference under ~0.002 "
          "is not distinguishable.", "", "Runs: " + "; ".join(f"{k} = {', '.join(r['runs'])}" for k, r in res.items()), ""]
    open(os.path.join(OUT, "ablation.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
