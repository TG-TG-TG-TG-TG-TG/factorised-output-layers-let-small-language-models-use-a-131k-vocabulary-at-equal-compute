"""Round-10 tables, plots, headline chart, social charts and verdicts (2026-11-07), from runs/round5, round8, round9,
round10. Windows Python (matplotlib):  PYTHONIOENCODING=utf8 python round10_analyze.py
  -> runs/round10/round10_summary.md, round10_results.json, loss_vs_tokens.png, delta_vs_B.png, buckets_partA.png,
     buckets_partD.png, round10_headline.png/.svg (+ runs/round10/social/ only if a verdict is positive)
Hand-written notes (surprises, deviations, bugs) come from runs/round10/round10_notes.md and are appended.
Parts that have not run yet are reported as "not run" (the script runs at any stage of the round).

Choices fixed before the results were seen (same conventions as round 9):
- Baseline B = B_r8 (round 8, seed 0). Val loss = val_loss_full of the branch; BPB = nats x round-8 bpb_per_nat.
  Tokens = stable + decay tokens. WSD branches at the same stable step are comparable across horizons (the stable
  phase has a constant LR), so B_r8's 75M..600M branches are the Part-C baselines.
- Wall-clock of a branch = its own-schedule wall time (round 9's load_run: run start to the end of that branch's
  decay, minus earlier inline branches' decay + eval and its own final eval).
- Training FLOPs of a branch = model.flops_per_token(cfg).total (runs/round10/acct.json) x tokens.
- H3* = the lowest 1200M val loss among H2atm (round 9) and the three Part-A arms (seed 0). Pinned 2026-11-07 22:10,
  after seed 0 and before any seed 1 (H3k8c4 4.1552 vs H3k8 4.1597, 0.0045 apart, inside the +-0.010 seed band):
  when an arm has seed 1, its score is the mean of seeds 0 and 1; H3* = the lower mean, even if the means differ by
  less than the seed-to-seed floor. The Part-C FLOPs rule plays no part in the pick.
- Part B: T = 300M x (H3*-384 smoke tok/s / B384 smoke tok/s, back to back), rounded to 25M. Comparisons:
  equal tokens (300M) and equal wall (H3*-384 at T vs B384's 300M, 35.2 min), with both actual walls.
- Part C: Δ vs B_r8 at every branch, Δ vs round-5 tCgcyc where branches overlap; spikes: max grad-norm after
  warmup <= 3x B_r8's over the same token range, and no train-loss jump > 0.5 nats above the previous 5 logs' mean.
- Part D: Δ vs H3* and vs B per branch; per-bucket Δ vs H3* (runs/round10/buckets_partD.json).
- Floor: seed-to-seed |s1 - s0| when seed 1 exists (max over the two arms), else 0.005.
- Verdict rules: the brief's (VERDICTS below).
"""
import glob
import json
import math
import os

import numpy as np

from round9_analyze import BPN, interp_log, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
R5, R8, R9, R10 = (os.path.join(HERE, "runs", r) for r in ("round5", "round8", "round9", "round10"))
TOK1200, TOK600 = 1_320_026_112, 660_013_056
B384_WALL300 = None  # filled from the log
A_ARMS = ["H2atm", "H3k8", "H3c4", "H3k8c4"]
KC = {"H2atm": (4096, 1024), "H3k8": (8192, 1024), "H3c4": (4096, 4096), "H3k8c4": (8192, 4096)}
ARM_OF = {"B": "tB", "H2atm": "tH2atm", "H3k8": "tH3k8", "H3c4": "tH3c4", "H3k8c4": "tH3k8c4", "tCgcyc": "tCgcyc",
          "Rloop": "Rloop", "Rdeep": "Rdeep", "Rwide": "Rwide", "RB24": "RB24", "D1": "tD1", "D2": "tD2",
          "B384": "B", "H2atm384": "H2atm384", "H3s384": "H3s384", "Rwide1200": "Rwide", "D1_lazy": "tD1", "D2_lazy": "tD2"}
NICE = {"B": "B (B_r8)", "H2atm": "H2atm (round 9)", "H3k8": "H3k8", "H3c4": "H3c4", "H3k8c4": "H3k8c4",
        "Rloop": "R-loop", "Rdeep": "R-deep", "Rwide": "R-wide", "RB24": "B24", "D1": "D1", "D2": "D2",
        "tCgcyc": "tCgcyc (round 5)", "B384": "B384", "H3s384": "H3*-384", "Rwide1200": "R-wide", "D1_lazy": "D1 lazy", "D2_lazy": "D2 lazy"}
G = "#7f7f7f"


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def js(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def lr_(dirn, name):
    return load_run(os.path.join(dirn, name + ".jsonl"))


def f4(x, s="+.4f"):
    return "-" if x is None else format(x, s)


def mins(s):
    return None if s is None else s / 60


def bkey(run, tok):
    """branch of a run whose stable token count is closest to tok (within 1%)."""
    if not run:
        return None
    ks = [k for k in run["branch"] if abs(k - tok) <= 0.01 * tok]
    return ks[0] if ks else None


def bloss(run, tok):
    k = bkey(run, tok)
    return run["branch"][k]["loss"] if k is not None else None


def gnorm_stats(run, max_tok, warm=3e6):
    tr = [r for r in run["config_recs"] if r.get("event") == "train" and warm < r["tokens"] <= max_tok]
    g = [r["gnorm"] for r in tr]
    ls = [r["loss"] for r in tr]
    jumps = [ls[i] - np.mean(ls[i - 5:i]) for i in range(5, len(ls))]
    return {"gnorm_max": max(g) if g else None, "gnorm_median": float(np.median(g)) if g else None,
            "max_loss_jump": float(max(jumps)) if jumps else None}


USER_COUNTS_LAZY = True  # user's call, 2026-11-08: lazy D1 meeting the D thresholds counts as 'input memory helps'
EXTRA_CMDS = []  # filled from runs/round10/commands_extra.txt (later chains), one command per line
if os.path.exists(os.path.join(R10, "commands_extra.txt")):
    EXTRA_CMDS = [l.rstrip("\n") for l in open(os.path.join(R10, "commands_extra.txt"), encoding="utf8") if l.strip()]


def med_tok(name):
    """median logged tok/s of a full run after step 400 (from its .out), None if absent."""
    f = os.path.join(R10, name + ".out")
    if not os.path.exists(f):
        return None
    v = []
    for line in open(f, encoding="utf8", errors="replace"):
        if line.startswith("step ") and "tok/s" in line:
            w = line.split()
            if int(w[1].split("/")[0]) > 400:
                v.append(float(w[4].replace(",", "")))
    return float(np.median(v)) if v else None


def with_recs(run, path):
    if run:
        run["config_recs"] = jl(path)
    return run


def main():
    res = {}
    L = ["# Round 10: fix the tail, confirm d=384, reinvest the saved compute, add free input memory (2026-11-07)", "",
         "Generated by `PYTHONIOENCODING=utf8 python round10_analyze.py` (Windows Python) from `runs/round5/`, "
         "`runs/round8/`, `runs/round9/` and `runs/round10/`; method choices are pinned in the script's docstring. "
         "Hand-written notes at the end come from `runs/round10/round10_notes.md`. Every run is seed 0 except the two Part-A seed-1 runs (the H3\\* tie-break).", ""]
    AC = js(os.path.join(R10, "acct.json"), {})
    SMA = js(os.path.join(R10, "smoke_check_partA.json"), {}).get("arms", {})
    SMR = js(os.path.join(R10, "smoke_check_rest.json"), {}).get("arms", {})
    SM = {**SMA, **SMR}
    SMD2 = js(os.path.join(R10, "smoke_check_D2.json"), {}).get("arms", {})  # D2 smokes, back to back with their own tH3k8c4
    GA = js(os.path.join(R10, "gates.json"), {})
    BK = js(os.path.join(R10, "buckets_partA.json"))
    BD = js(os.path.join(R10, "buckets_partD.json"))
    DEC = jl(os.path.join(R10, "decode.jsonl"))

    def P(dirn, name):
        return with_recs(lr_(dirn, name), os.path.join(dirn, name + ".jsonl"))

    RS = {"B": P(R8, "wsd_B_r8_s0"), "H2atm": P(R9, "wsd_H2atm_s0"), "tCgcyc": P(R5, "wsd_tCgcyc_s0"),
          "B384": P(R9, "wsd_B384_s0"), "H2atm384": P(R9, "wsd_H2atm384_s0")}
    for a in ("H3k8", "H3c4", "H3k8c4", "Rloop", "Rdeep", "Rwide", "RB24", "D1", "D2"):
        RS[a] = P(R10, f"wsd_{a}_s0")
    for a in ("Rloop", "Rdeep", "Rwide"):
        RS[a + "_lr1e-3"] = P(R10, f"wsd_{a}_lr0.001_s0")
    RS["D1_lazy"] = P(R10, "wsd_D1_lazy_s0")
    RS["D2_lazy"] = P(R10, "wsd_D2_lazy_s0")
    RS["Rwide1200"] = P(R10, "wsd_Rwide1200_s0")
    lb = sorted(glob.glob(os.path.join(R10, "wsd_H3s384_*M_s0.jsonl")))
    RS["H3s384"] = P(R10, os.path.basename(lb[-1])[:-6]) if lb else None
    RS = {k: v for k, v in RS.items() if v}
    B = RS["B"]
    stables = sorted(B["branch"])
    k1200 = max(stables)
    Bl = {k: B["branch"][k]["loss"] for k in stables}
    res["runs_present"] = sorted(RS)

    # ------------------------------------------------------------------ Part A
    have_A = [a for a in A_ARMS if a in RS and bkey(RS[a], 1200e6) is not None]
    S1 = {a: P(R10, f"wsd_{a}_s1") for a in A_ARMS[1:]}
    S1 = {a: r for a, r in S1.items() if r}
    res["partA_seed1"] = {a: bloss(r, 1200e6) for a, r in S1.items()}
    score = {a: (bloss(RS[a], 1200e6) + bloss(S1[a], 1200e6)) / 2 if a in S1 else bloss(RS[a], 1200e6) for a in have_A}
    res["H3star_score"] = score
    h3 = min(have_A, key=score.get) if len(have_A) == 4 else None
    res["H3star"] = h3
    res["H3star_complete"] = len(have_A) == 4
    L += ["## 1. Part A — tail fixes (d=128, 1200M, seed 0)", "",
          "All arms: round 9's general head (`hier9`) with tied-mean cluster logits (ū + R), K singleton tokens by "
          "training-set count, C' tail clusters by balanced k-means on B_r8's U rows of the tail tokens "
          "(`round10_classes.py`; K = 4096 / C' = 1024 is H2atm itself, round 9). Gates 1–5 "
          "(`runs/round10/gates.txt`): " + ("all OK." if GA and not GA.get("fails") else f"FAILED {GA.get('fails')}" if GA else "not run."), "",
          "### Branch table (val loss, nats/token; Δ vs B_r8)", "",
          "| arm | K | C' | " + " | ".join(f"{k / 1e6:.0f}M" for k in stables) + " | BPB @1200M |", "|---|---|---|" + "---|" * (len(stables) + 1),
          f"| {NICE['B']} | — | — | " + " | ".join(f"{Bl[k]:.4f}" for k in stables) + f" | {Bl[k1200] * BPN:.4f} |"]
    resA = {}
    for a in A_ARMS:
        r = RS.get(a)
        if not r:
            L.append(f"| {NICE[a]} | {KC[a][0]} | {KC[a][1]} | " + " | ".join("not run" for _ in stables) + " | - |")
            continue
        row = {}
        cells = []
        for k in stables:
            x = bloss(r, k)
            row[k] = x
            cells.append("-" if x is None else f"{x:.4f} ({x - Bl[k]:+.4f})")
        resA[a] = {str(k): {"loss": v, "delta": None if v is None else v - Bl[k]} for k, v in row.items()}
        star = " **= H3\\***" if a == h3 else ""
        L.append(f"| {NICE[a]}{star} | {KC[a][0]} | {KC[a][1]} | " + " | ".join(cells)
                 + f" | {f4(row.get(k1200) and row[k1200] * BPN, '.4f')} |")
    res["partA_branches"] = resA
    L += ["", f"**H3\\* = {NICE.get(h3, h3) if h3 else 'not decided yet (Part A incomplete)'}** (lowest 1200M val loss of the four; with seed 1, the mean of seeds 0 and 1 — rule pinned before seed 1 ran).", ""]
    if S1:
        L += ["Seed 1 at 1200M (run because the seed-0 gap between H3k8c4 and H3k8, 0.0045, was inside the ±0.010 band): "
              + "; ".join(f"{NICE[a]} s0 {bloss(RS[a], 1200e6):.4f} / s1 {bloss(S1[a], 1200e6):.4f} / mean {score[a]:.4f} ({score[a] - Bl[k1200]:+.4f} vs B)" for a in S1) + ".", ""]

    if BK:
        L += ["### Per-bucket Δ vs B (1200M finals; round-9 A1 style)", "",
              "Each cell: Δtotal (Δtop / Δwithin). top = −log p_top(slot(y)) (a singleton or the target's tail cluster), "
              "within = −log p(y | cluster); B's NLL is split by the same partition (B top = LSE_all − LSE_group, "
              "B within = LSE_group − z_y), so Δtop + Δwithin = Δtotal. Buckets = rank of the target by training-set "
              "count; share of val targets: " + ", ".join(f"{n} {s:.3f}" for n, s in zip(BK["buckets"], BK["bucket_share"])) + ". "
              f"Self-check: B here {BK['B']['nll_all_mean']:.5f} vs logged {BK['B']['logged']:.5f}.", "",
              "| arm | total | Δtotal | Δtop | Δwithin | " + " | ".join(BK["buckets"]) + " |", "|---|---|---|---|---|" + "---|" * 5]
        for lab, x in BK["arms"].items():
            L.append(f"| {NICE.get(lab, lab)} | {x['total']:.4f} | {x['d_total']:+.4f} | {x['d_top']:+.4f} | {x['d_within']:+.4f} | "
                     + " | ".join(f"{t:+.3f} ({tp:+.3f} / {w:+.3f})" for t, tp, w in zip(x["d_total_bucket"], x["d_top_bucket"], x["d_within_bucket"])) + " |")
        L += ["", "Contribution of each bucket to Δtotal (share × bucket mean Δ):", "",
              "| arm | " + " | ".join(BK["buckets"]) + " |", "|---|" + "---|" * 5]
        for lab, x in BK["arms"].items():
            L.append(f"| {NICE.get(lab, lab)} | " + " | ".join(f"{c:+.4f}" for c in x["contrib_bucket"]) + " |")
        L.append("")
        res["partA_buckets"] = BK

    L += ["### Cost table", "",
          "Smoke = 200 steps back to back with tB (`runs/round10/smoke_check_partA.json`); estimated full-run minutes = "
          "1.103 × 1,320M tokens / smoke tok/s (round 9's overhead factor). Own wall = own-schedule wall of the 1200M branch; "
          "whole run = the whole process including the four side branches.", "",
          "| arm | K | C' | capacity | params | head fwd FLOPs/tok (× B) | train FLOPs/tok (× B) | smoke tok/s | × B | step ms | "
          "peak VRAM GB | est. full min | own wall 1200M min | whole run min |", "|---|" * 14 + "|"[:0]]
    L[-1] = "|" + "---|" * 14
    for a in ["B"] + A_ARMS:
        arm = ARM_OF[a]
        sm = SM.get(arm, {})
        ac = AC.get(arm, {})
        r = RS.get(a)
        k = bkey(r, 1200e6) if r else None
        cap = "—" if a == "B" else str(-(-(50257 - KC[a][0]) // KC[a][1]))
        L.append(f"| {NICE[a]} | {'—' if a == 'B' else KC[a][0]} | {'—' if a == 'B' else KC[a][1]} | {cap} | "
                 f"{ac.get('total', 0):,} | " + (f"{sm['head_fwd_flops_per_token']:,.0f} ({sm['head_flops_vs_B']:.3f}) | " if sm else "- | ")
                 + (f"{ac['flops_per_token']['total']:,.0f} ({ac['flops_vs_B']:.3f}) | " if ac else "- | ")
                 + (f"{sm['tok_s']:,.0f} | {sm['speed_vs_B']:.2f} | {sm['step_ms']:.0f} | {sm['peak_mem_gb']:.2f} | {sm['est_full_min']:.0f} | " if sm else "- | - | - | - | - | ")
                 + (f"{r['branch'][k]['wall'] / 60:.1f} | {r['final']['wall_s'] / 60:.1f} |" if k is not None else "- | - |"))
    L += ["", "Smoke tok/s of H2atm was not re-measured this round (round 9: 3.5× B); its row uses round-9 logs only "
          "where marked.", ""]

    # ------------------------------------------------------------------ Part B
    L += ["## 2. Part B — d=384 at matched wall-clock", ""]
    B384 = RS.get("B384")
    b384_300 = bloss(B384, 300e6)
    b384_w300 = B384["branch"][bkey(B384, 300e6)]["wall"] if B384 else None
    H384 = RS.get("H3s384")
    resB = {"B384_300M": b384_300, "B384_wall300_min": mins(b384_w300)}
    if H384 and h3:
        T = max(H384["branch"])
        hT = H384["branch"][T]
        eq_tok_src = RS.get("H2atm384") if h3 == "H2atm" else H384
        e300 = bloss(eq_tok_src, 300e6)
        w300 = eq_tok_src["branch"][bkey(eq_tok_src, 300e6)]["wall"] if e300 is not None else None
        sm_b, sm_h = SM.get("B", {}), SM.get("H3s384", {})
        resB.update({"T_stable_tokens": T, "T_loss": hT["loss"], "T_wall_min": hT["wall"] / 60, "T_tokens": hT["tokens"],
                     "equal_tokens_300M": e300, "equal_tokens_delta": None if e300 is None else e300 - b384_300,
                     "equal_tokens_wall_min": mins(w300), "equal_wall_delta": hT["loss"] - b384_300,
                     "smoke_ratio": sm_h["tok_s"] / sm_b["tok_s"] if sm_h and sm_b else None,
                     "equal_tokens_source": "round 9 wsd_H2atm384_s0" if h3 == "H2atm" else os.path.basename(lb[-1])})
        L += [f"H3\\* = {NICE[h3]} on the round-5 Part-B d=384 shape (6 × d384, H 6, f 1536; classes and frequency split "
              f"from d=128). Smoke tok/s ratio H3\\*-384 / B384 (back to back) = {f4(resB['smoke_ratio'], '.3f')} → "
              f"T = {T / 1e6:.0f}M stable tokens.", "",
              "| comparison | H3*-384 | B384 (300M branch) | Δ | H3*-384 wall | B384 wall |", "|---|---|---|---|---|---|",
              f"| (i) equal tokens, 300M | {f4(e300, '.4f')} | {b384_300:.4f} | {f4(resB['equal_tokens_delta'])} | "
              f"{f4(mins(w300), '.1f')} min | {b384_w300 / 60:.1f} min |",
              f"| (ii) ≈ equal wall, {T / 1e6:.0f}M | {hT['loss']:.4f} | {b384_300:.4f} | {hT['loss'] - b384_300:+.4f} | "
              f"{hT['wall'] / 60:.1f} min | {b384_w300 / 60:.1f} min |", ""]
        if h3 == "H2atm":
            L += ["H3\\* is H2atm, so the equal-token point is round 9's 300M result and the new run has only the long branch.", ""]
    else:
        L += ["Not run yet.", ""]
    res["partB"] = resB

    # ------------------------------------------------------------------ Part C
    L += ["## 3. Part C — reinvest the saved head compute in the backbone (d=128, 600M)", ""]
    C_ARMS = [a for a in ("Rloop", "Rdeep", "Rwide", "RB24") if a in RS or ARM_OF[a] in AC]
    resC = {}
    cst = [k for k in stables if k <= 600e6 * 1.01]
    if C_ARMS:
        L += ["| arm | backbone | blocks / per-copy / emb / head params | train FLOPs/tok (× B) | smoke tok/s (× B) | peak VRAM GB | "
              "wall 600M min | " + " | ".join(f"{k / 1e6:.0f}M (Δ vs B)" for k in cst) + " | Δ vs tCgcyc @600M | max gnorm (× B) |",
              "|---|---|---|---|---|---|---|" + "---|" * len(cst) + "---|---|"]
        gB = gnorm_stats(B, 660e6)
        Tc = RS.get("tCgcyc")
        desc = {"Rloop": "6 blocks × 4 passes, cycle order, Cgcyc gates", "Rdeep": "24 unique blocks", "Rwide": "6 blocks, d 256, H 4, f 1024",
                "RB24": "24 unique blocks, flat head"}
        for a in C_ARMS:
            for variant in (a, a + "_lr1e-3"):
                r = RS.get(variant)
                if variant != a and not r:
                    continue
                ac, sm = AC.get(ARM_OF[a], {}), SM.get(ARM_OF[a], {})
                cells, row = [], {}
                for k in cst:
                    x = bloss(r, k) if r else None
                    row[str(k)] = x
                    cells.append("-" if x is None else f"{x:.4f} ({x - Bl[k]:+.4f})")
                k6 = bkey(r, 600e6) if r else None
                tcg = bloss(Tc, 600e6) if Tc else None
                gs = gnorm_stats(r, 660e6) if r else {}
                resC[variant] = {"loss": row, "delta_vs_B": {kk: (None if v is None else v - Bl[int(float(kk))]) for kk, v in row.items()},
                                 "delta_vs_tCgcyc_600M": None if (k6 is None or tcg is None) else r["branch"][k6]["loss"] - tcg,
                                 "wall600_min": None if k6 is None else r["branch"][k6]["wall"] / 60, "gnorm": gs, "gnorm_B": gB,
                                 "flops_vs_B": ac.get("flops_vs_B"), "params": {q: ac.get(q) for q in ("blocks", "per_copy", "embedding", "head", "total")}}
                L.append(f"| {NICE[a]}{' (LR 1e-3)' if variant != a else ''} | {desc[a]} | "
                         + (f"{ac['blocks']:,} / {ac['per_copy']:,} / {ac['embedding']:,} / {ac['head']:,} | {ac['flops_vs_B']:.3f} | " if ac else "- | - | ")
                         + (f"{sm['tok_s']:,.0f} ({sm['speed_vs_B']:.2f}) | {sm['peak_mem_gb']:.2f} | " if sm else "- | - | ")
                         + f"{f4(resC[variant]['wall600_min'], '.1f')} | " + " | ".join(cells)
                         + f" | {f4(resC[variant]['delta_vs_tCgcyc_600M'])} | "
                         + (f"{gs['gnorm_max']:.2f} ({gs['gnorm_max'] / gB['gnorm_max']:.2f}) |" if gs.get("gnorm_max") else "- |"))
        L += ["", f"B_r8 (flat head, 6 blocks): train FLOPs/tok {AC.get('tB', {}).get('flops_per_token', {}).get('total', 0):,.0f}; "
              f"max grad-norm after warmup to 660M tokens {gB['gnorm_max']:.2f}. tCgcyc (round 5, 6 blocks × 2 passes, flat head) "
              f"600M branch: {f4(bloss(Tc, 600e6) if Tc else None, '.4f')}.", ""]
        rw = RS.get("Rwide1200")
        if rw:
            k12 = bkey(rw, 1200e6)
            rep6 = bloss(rw, 600e6)
            resC["Rwide1200"] = {"loss": {str(k): bloss(rw, k) for k in stables}, "delta_vs_B": {str(k): (None if bloss(rw, k) is None else bloss(rw, k) - Bl[k]) for k in stables},
                                 "wall1200_min": rw["branch"][k12]["wall"] / 60 if k12 else None, "rep600_minus_600Mrun": None if rep6 is None or "Rwide" not in RS else rep6 - bloss(RS["Rwide"], 600e6),
                                 "flops_vs_B": AC.get("Rwide", {}).get("flops_vs_B")}
            L += ["**R-wide rerun to 1200M** (the brief's rule: the best Part-C arm is more than 0.020 below B at 600M; run `wsd_Rwide1200_s0`, same arm): "
                  + "; ".join(f"{k / 1e6:.0f}M {bloss(rw, k):.4f} ({bloss(rw, k) - Bl[k]:+.4f})" for k in stables if bloss(rw, k) is not None)
                  + f". Own wall to 1200M {f4(resC['Rwide1200']['wall1200_min'], '.1f')} min (B_r8: {B['branch'][k1200]['wall'] / 60:.1f} min). "
                  f"Its 600M branch minus the 600M run's: {f4(resC['Rwide1200']['rep600_minus_600Mrun'])} (same seed and data; GPU nondeterminism only).", ""]
    else:
        L += ["Not run yet.", ""]
    res["partC"] = resC

    # ------------------------------------------------------------------ Part D
    L += ["## 4. Part D — hashed n-gram input embeddings (d=128, 1200M)", ""]
    resD = {}
    h3_runs = [f"wsd_{h3}_s0", f"wsd_{h3}_s1"] if h3 else []
    h3_med = np.mean([m for m in (med_tok(n) for n in h3_runs) if m]) if h3 else None
    D_RUNS = [v for v in ("D1", "D1_lazy", "D2", "D2_lazy") if v in RS]
    if D_RUNS and h3:
        H = RS[h3]
        Hl = {k: bloss(H, k) for k in stables}
        L += ["| run | params blocks / emb / head / n-gram | n-gram params + Adam GB | train FLOPs/tok | smoke tok/s (× H3*) | peak VRAM GB | "
              + " | ".join(f"{k / 1e6:.0f}M" for k in stables) + " |", "|---|---|---|---|---|---|" + "---|" * len(stables)]
        for v in D_RUNS:
            r = RS[v]
            arm = ARM_OF[v.split("_")[0]]
            ac = AC.get(arm, {})
            src = SMD2 if v.startswith("D2") else SM
            sm = src.get(arm + ("_lazy" if v.endswith("_lazy") else ""), {})
            smh = src.get(ARM_OF[h3], {})
            cells, row = [], {}
            for k in stables:
                x = bloss(r, k)
                row[str(k)] = {"loss": x, "vs_H3": None if x is None or Hl[k] is None else x - Hl[k], "vs_B": None if x is None else x - Bl[k]}
                cells.append("-" if x is None else f"{x:.4f} ({row[str(k)]['vs_H3']:+.4f} / {row[str(k)]['vs_B']:+.4f})")
            resD[v] = {"branches": row, "tok_s": sm.get("tok_s"), "tok_s_vs_H3": sm["tok_s"] / smh["tok_s"] if sm and smh else None,
                       "fullrun_median_tok_s": med_tok(f"wsd_{v}_s0"), "fullrun_vs_H3": None if med_tok(f"wsd_{v}_s0") is None else med_tok(f"wsd_{v}_s0") / h3_med}
            L.append(f"| {v} | " + (f"{ac['blocks']:,} / {ac['embedding']:,} / {ac['head']:,} / {ac['ngram']:,} | {ac['ngram_param_plus_adam_GB']:.2f} | "
                                    f"{ac['flops_per_token']['total']:,.0f} | " if ac else "- | - | - | ")
                     + (f"{sm['tok_s']:,.0f} ({resD[v]['tok_s_vs_H3']:.2f}) | {sm['peak_mem_gb']:.2f} | " if sm and resD[v]['tok_s_vs_H3'] else "- | - | ")
                     + " | ".join(cells) + " |")
        L += ["", "Cells: loss (Δ vs H3\\* / Δ vs B). Full-run speed (median logged tok/s after step 400, from each run's .out; H3\\* = mean of "
              f"its two seeds, {h3_med:,.0f}): " + "; ".join(f"{v} {resD[v]['fullrun_median_tok_s']:,.0f} ({resD[v]['fullrun_vs_H3']:.2f}× H3\\*)" for v in D_RUNS if resD[v].get('fullrun_vs_H3')) + ".", "",
              "Leak check (`round10_causal.py` → `runs/round10/causal.txt`): replacing every token after position t leaves the n-gram ids and the hidden "
              "states up to t bit-identical (max |Δh| = 0) for tD1 / tD2 at init with random tables and for both trained D1 checkpoints, and changes them at t+1: "
              + ("all OK." if "ALL CAUSAL CHECKS OK" in open(os.path.join(R10, "causal.txt")).read() else "FAILED.") if os.path.exists(os.path.join(R10, "causal.txt")) else "not run.", ""]
        if BD:
            L += ["Per-bucket Δ vs H3\\* (1200M finals):", "", "| run | Δtotal | " + " | ".join(BD["buckets"]) + " |", "|---|---|" + "---|" * 5]
            hb = BD["arms"].get(h3)
            for lab, x in BD["arms"].items():
                if lab == h3 or not hb:
                    continue
                L.append(f"| {lab} | {x['total'] - hb['total']:+.4f} | " + " | ".join(f"{a - b:+.4f}" for a, b in zip(x["total_bucket"], hb["total_bucket"])) + " |")
            L.append("")
    else:
        L += ["Not run yet.", ""]
    res["partD"] = resD

    # ------------------------------------------------------------------ decode
    if DEC:
        L += ["## 5. Decode, batch 1 (round-8 tiny protocol, random bf16 weights)", ""]
        md = os.path.join(R10, "decode.md")
        if os.path.exists(md):
            L += [open(md, encoding="utf8").read().replace("# Round 10 decode", "Round 10 decode"), ""]
        res["decode"] = {a: float(np.mean([r["ms_per_token"] for r in DEC if r["arm"] == a])) for a in sorted({r["arm"] for r in DEC})}

    # ------------------------------------------------------------------ verdicts
    L += ["## 6. Verdicts", ""]
    V = {}
    if len(have_A) >= 1:
        best = min((a for a in have_A if a != "H2atm"), key=lambda a: bloss(RS[a], 1200e6), default=None)
        if best:
            d = bloss(RS[best], 1200e6) - Bl[k1200]
            V["tail_fixed"] = d <= 0.020
            L.append(f"- **Tail fixed** (a Part-A arm ≤ +0.020 vs B at 1200M): {'yes' if d <= 0.020 else 'no'} — best {NICE[best]} {d:+.4f}"
                     + ("" if len(have_A) == 4 else " (Part A incomplete)") + ".")
    if resB.get("T_loss") is not None:
        V["d384_confirmed"] = resB["T_loss"] <= b384_300 - 0.010
        L.append(f"- **d=384 confirmed** (H3*-384 at T ≥ 0.010 below B384's 300M {b384_300:.4f}): {'yes' if V['d384_confirmed'] else 'no'} — "
                 f"{resB['T_loss']:.4f} at {resB['T_stable_tokens'] / 1e6:.0f}M ({resB['equal_wall_delta']:+.4f}; wall {resB['T_wall_min']:.1f} vs {b384_w300 / 60:.1f} min).")
    if resC:
        k6 = str(min(cst, key=lambda k: abs(k - 600e6)))
        wins = {a: (v["delta_vs_B"].get(k6) is not None and v["delta_vs_B"][k6] < -0.020 and (v["flops_vs_B"] or 9) <= 1.1)
                for a, v in resC.items() if not a.startswith("RB24") and a != "Rwide1200"}
        V["reinvestment_wins"] = any(wins.values())
        V["Rloop_wins"] = wins.get("Rloop", False) or wins.get("Rloop_lr1e-3", False)
        L.append("- **Reinvestment wins** (a Part-C arm > 0.020 below B at 600M with ≤ 1.1× B's training FLOPs): "
                 f"{'yes' if V['reinvestment_wins'] else 'no'} — " + "; ".join(
                     f"{NICE.get(a.split('_')[0], a)}{' (LR 1e-3)' if a.endswith('lr1e-3') else ''} {f4(v['delta_vs_B'].get(k6))} at {f4(v['flops_vs_B'], '.3f')}× FLOPs"
                     for a, v in resC.items() if not a.startswith("RB24") and a != "Rwide1200") + " (the R-wide 1200M rerun is not part of this 600M verdict).")
        L.append(f"- **R-loop (iso-params) wins**: {'yes' if V['Rloop_wins'] else 'no'}.")
    if resD.get("D1") and h3:
        d1 = resD["D1"]["branches"].get(str(k1200), {})
        lz = resD.get("D1_lazy", {})
        use = "D1"
        if lz and lz["branches"].get(str(k1200), {}).get("loss") is not None and d1.get("loss") is not None:
            if abs(lz["branches"][str(k1200)]["loss"] - d1["loss"]) <= 0.003 and (lz.get("tok_s") or 0) > (resD["D1"].get("tok_s") or 0):
                use = "D1_lazy"
        x = resD[use]["branches"].get(str(k1200), {})
        sp = resD[use].get("tok_s_vs_H3")
        if x.get("vs_H3") is not None:
            V["input_memory_helps"] = x["vs_H3"] < -0.010 and (sp or 0) >= 0.85
            fr = resD[use].get("fullrun_vs_H3")
            L.append(f"- **Input memory helps** (D1 > 0.010 below H3* at 1200M, tok/s ≥ 0.85× H3*): {'yes' if V['input_memory_helps'] else 'no'} — "
                     f"{NICE.get(use, use)} (dense AdamW) {x['vs_H3']:+.4f} vs H3*, tok/s {f4(sp, '.2f')}× H3* (smoke), {f4(fr, '.2f')}× over the full run.")
            if lz and use == "D1":
                xl = lz["branches"].get(str(k1200), {})
                V["input_memory_helps_lazy"] = (xl.get("vs_H3") or 0) < -0.010 and (lz.get("tok_s_vs_H3") or 0) >= 0.85
                L.append(f"  - D1 with lazy (row-sparse) AdamW: {f4(xl.get('vs_H3'))} vs H3* at {f4(lz.get('tok_s_vs_H3'), '.2f')}× tok/s (smoke), "
                         f"{f4(lz.get('fullrun_vs_H3'), '.2f')}× (full run). It meets both numbers, but the brief swaps lazy in only if its loss "
                         f"matches dense within 0.003, and it is {f4(xl.get('loss') - d1.get('loss'))} from dense (better, not matching), so it does not "
                         "stand in for D1 by that rule.")
                if USER_COUNTS_LAZY and V["input_memory_helps_lazy"]:
                    V["input_memory_helps_dense"] = V["input_memory_helps"]
                    V["input_memory_helps"] = True
                    V["input_memory_helps_by"] = "D1_lazy (user decision 2026-11-08)"
                    L.append("  - **User decision (2026-11-08): lazy D1 counts. Input memory helps = yes (via lazy AdamW)**; the dense-only reading above stays on record.")
                    im = next(i for i, l in enumerate(L) if l.startswith("- **Input memory helps**"))
                    L[im] = L[im].replace("tok/s ≥ 0.85× H3*): no — ", "tok/s ≥ 0.85× H3*): **yes, via lazy AdamW (user decision)**. Dense-only reading: no — ")
    L.append("")
    res["verdicts"] = V

    # ------------------------------------------------------------------ all-arms table (inserted after the header)
    PART = {"B": "ref", "H2atm": "A (r9)", "H3k8": "A", "H3c4": "A", "H3k8c4": "A", "tCgcyc": "C ref (r5)", "Rloop": "C", "Rdeep": "C",
            "Rwide": "C", "Rwide1200": "C (to 1200M)", "RB24": "C opt", "D1": "D", "D1_lazy": "D", "D2": "D", "D2_lazy": "D",
            "B384": "B ref (r9)", "H2atm384": "B ref (r9)", "H3s384": "B"}
    allrows = [(l, RS[l], 0) for l in PART if l in RS] + [(l + "_s1", S1[l], 1) for l in S1]
    allrows += [(l, RS[l], 0) for l in RS if l.endswith("lr1e-3")]
    order = list(PART) + [a + "_s1" for a in A_ARMS] + [l for l in RS if l.endswith("lr1e-3")]
    allrows.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    B384r = RS.get("B384")
    TA = ["## 0. All arms", "",
          "Final branch of every run used in this round (val loss, nats/token). Δ is vs the same-width flat-softmax baseline at the same "
          "training tokens: B_r8 for d=128 arms, B384 for d=384 arms (H3*-384 at 575M has no same-token B384, so its Δ is vs B384 at 300M, "
          "equal wall-clock, marked †). Params include the n-gram tables; FLOPs are analytic training FLOPs/token (6N + attention) relative to the same-width flat baseline (B for d=128, B384 for d=384). "
          "Wall = own WSD schedule to that branch, RTX 5070 Ti.", "",
          "| run | Part | seed | params (M) | FLOPs/tok vs B | tokens | loss | Δ vs baseline | BPB | wall (min) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    res["all_arms"] = {}
    for l, r, s in allrows:
        base = l[:-3] if l.endswith("_s1") else l
        arm = ARM_OF.get(base.split("_lr")[0])
        ac = AC.get(arm, {})
        k = max(r["branch"])
        lo = r["branch"][k]["loss"]
        ref = B384r if base in ("B384", "H2atm384", "H3s384") else B
        kb = bkey(ref, k)
        mark = ""
        if base == "H3s384" and kb is None and B384r:
            kb, mark = bkey(B384r, 300e6), " †"
        dl = None if (kb is None or base in ("B", "B384")) else lo - ref["branch"][kb]["loss"]
        wall = r["branch"][k].get("wall")
        nm = NICE.get(base.split("_lr")[0], base) + (" (LR 1e-3)" if "lr1e-3" in base else "")
        TA.append(f"| {nm} | {PART.get(base, 'C')} | {s} | {f4((ac.get('total') or 0) / 1e6 if ac else None, '.2f')} | "
                  f"{f4(ac.get('flops_vs_B'), '.3f')} | {k / 1e6:.0f}M | {lo:.4f} | {f4(dl) + mark if dl is not None else '—'} | {lo * BPN:.4f} | "
                  f"{f4(mins(wall), '.1f')} |")
        res["all_arms"][l] = {"tokens": k, "loss": lo, "delta_vs_baseline": dl, "wall_min": mins(wall),
                              "params": ac.get("total"), "flops_vs_B": ac.get("flops_vs_B"), "seed": s}
    TA.append("")
    L[4:4] = TA

    # ------------------------------------------------------------------ figures (headline last)
    import round10_plots as RP
    figs = []
    RP.facets(RS, NICE, h3, figs, R10)
    RP.buckets(BK, BD, h3, NICE, figs, R10)

    def d_same(l):
        r = RS[l]
        k = max(r["branch"])
        kb = bkey(B, k)
        return None if kb is None else r["branch"][k]["loss"] - Bl[kb]
    cand = {l: d_same(l) for l in RS if l in ARM_OF and l not in ("B", "B384", "H2atm384", "H3s384", "tCgcyc") and ARM_OF[l] in AC}
    cand = {l: v for l, v in cand.items() if v is not None}
    best = min(cand, key=cand.get) if cand else None
    res["headline_best_arm"] = best
    social_files = []
    tweets = []
    FOOT = ("{size}; FineWeb (streamed), seed 0, one RTX 5070 Ti; val loss on 5M held-out tokens. "
            "@ProjectTEST-AI · caveats: one seed, small models, {extra}")
    if V.get("tail_fixed"):
        a = min((x for x in have_A if x != "H2atm"), key=lambda x: bloss(RS[x], 1200e6))
        r = RS[a]
        k = bkey(r, 1200e6)
        d = r["branch"][k]["loss"] - Bl[k1200]
        rt = B["branch"][k1200]["wall"] / r["branch"][k]["wall"]
        ser = [("B: full softmax", [p[0] / 60 for p in RP.pts(B, "wall")], [p[1] for p in RP.pts(B, "wall")], RP.GREY, False),
               (f"{NICE[a]}: two-level head", [p[0] / 60 for p in RP.pts(r, "wall")], [p[1] for p in RP.pts(r, "wall")], RP.BLUE, True)]
        t = f"Two-level output head: {d:+.3f} nats vs full softmax at 1.3B tokens, in {rt:.1f}× less training time"
        social_files += RP.social("tail_fixed", ser, "training wall-clock, minutes (RTX 5070 Ti, log scale)", t,
                                  FOOT.format(size="7.8M-param GPT (d=128, 6 layers, 50k vocab)", extra="wall-clock includes eval-free own schedule"), R10)
        tweets.append((t, "social/tail_fixed_1600x900.png", f"A two-level softmax (K={KC[a][0]} frequent tokens + {KC[a][1]} clusters) "
                       f"trained a 7.8M-param GPT to {d:+.3f} nats of the full-softmax loss at 1.3B tokens, {rt:.1f}× faster on one RTX 5070 Ti. One seed, small model."))
    if V.get("d384_confirmed"):
        bw = [p[0] / 60 for p in RP.pts(B384, "wall")]
        hw = [p[0] / 60 for p in RP.pts(H384, "wall")]
        ser = [("B384: full softmax", bw, [p[1] for p in RP.pts(B384, "wall")], RP.GREY, False),
               ("two-level head", hw, [p[1] for p in RP.pts(H384, "wall")], RP.BLUE, True)]
        dw = resB["equal_wall_delta"]
        NL = chr(10)
        t = f"Same GPU time, lower loss: {dw:+.3f} nats".replace("-", "−") + NL + "with a two-level output head (d=384 GPT)"
        foot = (f"30M-param GPT (d=384, 6 layers, 50k vocab), FineWeb (streamed), seed 0, one RTX 5070 Ti; wall-clock = own training schedule." + NL
                + f"Caveat: at equal tokens (300M) the two-level head is {resB['equal_tokens_delta']:+.3f} nats worse; the gain comes from "
                f"{resB['smoke_ratio']:.1f}× training throughput ({resB['T_stable_tokens'] / 1e6:.0f}M vs 300M tokens). One seed, one model size." + NL
                + "@ProjectTEST-AI")
        social_files += RP.social("d384_wall", ser, "training wall-clock (minutes, RTX 5070 Ti)", t, foot, R10, xlog=False, xzero=True,
                                  note=(bw[-1], b384_300, resB["T_loss"], f"{dw:+.3f}".replace("-", "−")))
    if V.get("reinvestment_wins"):
        k6 = str(min(cst, key=lambda k: abs(k - 600e6)))
        a = min((x for x, v in resC.items() if not x.startswith("RB24") and v["delta_vs_B"].get(k6) is not None), key=lambda x: resC[x]["delta_vs_B"][k6])
        r = RS[a]
        fa, fb = AC[ARM_OF[a.split("_")[0]]]["flops_per_token"]["total"], AC["tB"]["flops_per_token"]["total"]
        ser = [("B", [p[0] * fb for p in RP.pts(B, "tokens") if p[0] < 700e6], [p[1] for p in RP.pts(B, "tokens") if p[0] < 700e6], RP.GREY, False),
               (NICE[a.split("_")[0]], [p[0] * fa for p in RP.pts(r, "tokens")], [p[1] for p in RP.pts(r, "tokens")], RP.BLUE, True)]
        dd = resC[a]["delta_vs_B"][k6]
        t = f"Spend the head's saved compute on the backbone: {dd:+.3f} nats at {resC[a]['flops_vs_B']:.2f}× the training FLOPs"
        social_files += RP.social("reinvest", ser, "training FLOPs (log scale)", t,
                                  FOOT.format(size="d=128 GPT, 600M tokens", extra="FLOPs are analytic (6N + attention)"), R10)
        tweets.append((t, "social/reinvest_1600x900.png", f"Cheaper output head, deeper backbone: {NICE[a.split('_')[0]]} reaches {dd:+.3f} nats vs the flat-softmax "
                       f"baseline at 600M tokens for {resC[a]['flops_vs_B']:.2f}× its training FLOPs. One seed, small model."))
    if V.get("input_memory_helps"):
        u = "D1_lazy" if V.get("input_memory_helps_by", "").startswith("D1_lazy") else "D1"
        r, H = RS[u], RS[h3]
        x = resD[u]["branches"][str(k1200)]
        NL = chr(10)
        hx, rx = [p[0] / 1e9 for p in RP.pts(H, "tokens")], [p[0] / 1e9 for p in RP.pts(r, "tokens")]
        ser = [("two-level-head GPT", hx, [p[1] for p in RP.pts(H, "tokens")], RP.GREY, False),
               ("+ hashed bigram table", rx, [p[1] for p in RP.pts(r, "tokens")], RP.BLUE, True)]
        dv = f"{x['vs_H3']:+.3f}".replace("-", "−")
        t = f"Hashed bigram inputs: {dv} nats" + NL + "at the same speed, 8M-param GPT"
        dd = resD["D1"]["branches"][str(k1200)]["vs_H3"]
        foot = (f"d=128, 6-layer GPT with a two-level output head; + 2^20 x 128 hashed bigram input table (134M lookup params, ~0 extra FLOPs), "
                f"row-sparse (lazy) AdamW, {f4(resD[u].get('fullrun_vs_H3'), '.2f')}x the baseline's tok/s. FineWeb (streamed), seed 0, one RTX 5070 Ti; "
                "val loss on 5M held-out tokens." + NL
                + f"Caveats: with dense AdamW the gain is {dd:+.3f} but training is 20% slower; one seed, one small model; lookup params are not compute." + NL
                + "@ProjectTEST-AI")
        social_files += RP.social("input_memory", ser, "training tokens (billions)", t, foot, R10, xlog=False, xzero=True,
                                  note=(hx[-1], bloss(H, 1200e6), x["loss"], dv))
    res["social"] = social_files
    # tweets: hand-written in runs/round10/social/tweets.md from round10_results.json (not generated)
    cap = ("Δ = vs B (flat softmax, 6 × d128) at the same training tokens; d=384 arms vs B384 at 300M (equal wall-clock). "
           "One seed. Wall-clock on one RTX 5070 Ti, own schedule.")
    RP.headline(RS, AC, ARM_OF, NICE, best, BPN, cap, R10, figs)
    res["figures"] = figs
    L += ["## 7. Figures", "", "Colour-blind check: `validate_palette.js` (light, all pairs) passes for blue / yellow / aqua / red; "
          "red vs aqua is 6.9 under deuteranopia, so arms also differ by marker and direct label. No figure shows more than four "
          "coloured arms plus grey.", ""] + [f"- `runs/round10/{f}`" for f in figs] + \
         (["", "Social charts (a verdict is positive): " + ", ".join(f"`runs/round10/{f}`" for f in social_files) + "; tweets in `runs/round10/social/tweets.md`."]
          if social_files else ["", "Social charts: none (no verdict is positive)."]) + [""]

    L += ["## 8. Exact commands", "",
          "All GPU steps in WSL (`wsl -d AEON-Training --cd <repo> bash r10chain.sh <steps>`; a step's words are joined by `+`; "
          "`r10chain.sh` and every launcher set `ulimit -c 0`). Analysis in Windows Python. In order:", "",
          "```",
          "python round10_classes.py                                   # class splits -> runs/round10/adapt_*.npy (once, before the chains)",
          "bash r10chain.sh cpu+round10_compat.py cpu+round10_gates.py smoke+tB+tH3k8+tH3c4+tH3k8c4 \\",
          "  py+round10_smoke_check.py+--tag+partA+--arms+tB,tH3k8,tH3c4,tH3k8c4 one+tiny+tH3k8:0,tH3c4:0,tH3k8c4:0         # chainA.out",
          "bash r10chain.sh one+tiny+tH3k8c4:1 pytry+round10_buckets.py+--tag+partA+--arms+H2atm:tH2atm:wsd_H2atm_s0,H3k8:tH3k8:wsd_H3k8_s0,H3c4:tH3c4:wsd_H3c4_s0,H3k8c4:tH3k8c4:wsd_H3k8c4_s0   # chainA2.out",
          "bash r10chain.sh one+tiny+tH3k8:1                                                                               # chainA3.out",
          "bash r10chain.sh cpu+round10_acct.py+Rloop+Rdeep+Rwide+RB24+tD1+tD2+H3s384 cpu+round10_gates_cd.py+Rloop+Rdeep+Rwide+RB24+tD1+tD2+H3s384  # chainBCD0.out (gates_cd crashed on RB24; fixed)",
          "bash r10chain.sh cpu+round10_gates_cd.py+Rloop+Rdeep+Rwide+RB24+tD1+tD2+H3s384 smoke+tB+tH3k8c4+Rloop+Rdeep+Rwide+RB24+tD1+tD1@lazy+B+H3s384 \\",
          "  py+round10_smoke_check.py+--tag+rest+--arms+tB,tH3k8c4,Rloop,Rdeep,Rwide,RB24,tD1,tD1_lazy,B,H3s384 \\",
          "  py+round10_decode.py+run+0+8192+4096 py+round10_decode.py+run+1+8192+4096 py+round10_decode.py+run+2+8192+4096 py+round10_decode.py+md   # chainBCD0b.out",
          "bash r10chain.sh one+c600+Rloop:0,Rdeep:0,Rwide:0 one+d384+H3s384:0+--tokens+575e6+--branch_at+300e6,575e6 one+tiny+tD1:0 one+tiny+tD1:0+--extra=--ng_lazy   # chainBCD1.out",
          "bash r10chain.sh one+tiny+Rwide1200:0                                                                           # chainC2.out",
          "bash r10chain.sh cpu+round10_causal.py                                                                          # causal.txt",
          ] + [c for c in EXTRA_CMDS] + [
          "PYTHONIOENCODING=utf8 python round10_analyze.py              # Windows Python: this summary, round10_results.json, every figure (headline last)",
          "```", "", "Ledger of GPU time: `runs/round10/budget.json`. Run logs: `runs/round10/wsd_*.jsonl` / `.out`; smokes: `runs/round10/smoke/`.", ""]

    # ------------------------------------------------------------------ notes, json
    notes = os.path.join(R10, "round10_notes.md")
    if os.path.exists(notes):
        L += [open(notes, encoding="utf8").read(), ""]
    open(os.path.join(R10, "round10_summary.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(R10, "round10_results.json"), "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
