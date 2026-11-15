"""Round-9 tables, plots, headline chart and verdicts (2026-11-07), from runs/round8 + runs/round9.
Windows Python (matplotlib):  PYTHONIOENCODING=utf8 python round9_analyze.py
  -> runs/round9/round9_summary.md, round9_results.json, loss_vs_tokens.png, delta_vs_B.png,
     decomposition_by_bucket.png, round9_headline.png/.svg (+ round9_headline_d384.png/.svg)
Hand-written notes (surprises, deviations) come from runs/round9/round9_notes.md and are appended.

Choices fixed before the fix-arm results were seen:
- Baseline B = B_r8 (round 8), seed 0 (+ seed 1 if run). Δ vs B paired per seed.
- Val loss = val_loss_full of the branch; BPB = nats x runs/round8/valbytes.json bpb_per_nat.
- Tokens = stable + decay tokens ("tokens seen").
- Wall-clock of a branch result = its own-schedule wall time: time from the run's start to the end of that
  branch's decay, minus the time the run spent in earlier inline branches (their decay + eval) and minus that
  branch's own final eval. So B_r8's 1200M point is its run time without the 75M..600M side branches.
  Periodic stable-phase evals stay in (small, the same for every arm).
- Equal wall-clock, two ways: (raw) the arm's loss interpolated log-linearly in own-schedule wall time at B's
  1200M wall; (smoke-normalised) at the token count the arm would reach in B's 1200M tokens' time at today's
  back-to-back smoke tok/s ratio, T = 1,320M x (arm smoke tok/s / tB smoke tok/s), log-linear in tokens.
  Outside the arm's measured range -> "not reached" (no extrapolation).
- Equal FLOPs: B_r8's training FLOPs (flops_per_token.total x 1,320M tokens) / the arm's FLOPs per token
  -> token count, the arm's loss there log-linear in tokens (no extrapolation).
- Floor: seed-to-seed |s1 - s0| at 1200M for B and the arm when seed 1 exists (max), else 0.005.
- H2* = the lowest 1200M val loss among round-8 H2, H2tm, H2a, H2atm (seed 0).
- Verdict rules: the brief's (pinned in VERDICTS below).
- ADDED AFTER Part E came in (the pinned interpolation was undefined: H2*-384 finished its 300M branch before
  B384 reached its first, 150M, branch, so the wall-clock curves do not overlap): an assumption-free bound,
  H2*-384's last result vs B384's first result (reached with MORE wall-clock). Assumes a decayed branch's loss
  falls with tokens. The d=384 regime verdict rests on this bound.
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R8 = os.path.join(HERE, "runs", "round8")
R9 = os.path.join(HERE, "runs", "round9")
VB = json.load(open(os.path.join(R8, "valbytes.json")))
BPN = VB["bpb_per_nat"]
TOK1200 = 1_320_026_112
NICE = {"B_r8": "B (B_r8)", "H2": "H2 (round 8)", "H2r": "H2r (round 8)", "CB": "CB (round 8)",
        "H2tm": "H2tm", "H2a": "H2a", "H2atm": "H2atm", "H2u": "H2u", "B384": "B384",
        "H2_384": "H2-384", "H2tm384": "H2tm-384", "H2a384": "H2a-384", "H2atm384": "H2atm-384"}
ARM_OF = {"B_r8": "tB", "H2": "tH2", "H2r": "tH2r", "CB": "tCB", "H2tm": "tH2tm", "H2a": "tH2a", "H2atm": "tH2atm",
          "H2u": "tH2u", "B384": "B", "H2_384": "H2_384", "H2tm384": "H2tm384", "H2a384": "H2a384", "H2atm384": "H2atm384"}
FIX = ["H2tm", "H2a", "H2atm", "H2u"]
MARK = {"H2": "s", "H2r": "P", "CB": "X", "H2tm": "^", "H2a": "v", "H2u": "*"}  # B and H2* keep "o"


def jl(p):
    return [json.loads(l) for l in open(p)]


def load_run(path):
    """branches {stable: dict(tokens, loss, wall)} with own-schedule wall times, + final record."""
    if not (os.path.exists(path) and os.path.exists(path[:-6] + ".DONE")):
        return None
    recs = jl(path)
    t0 = next(r["t"] for r in recs if "t" in r)
    br = sorted((r for r in recs if r.get("event") == "branch"), key=lambda r: r["t"])
    side, out = 0.0, {}
    for r in br:
        own = r["t"] - t0 - side - r["eval_s"]
        out[r["stable_tokens"]] = {"tokens": r["tokens"], "loss": r["val_loss_full"], "wall": own,
                                   "decay_train_s": r["train_s"], "branch_step": r["branch_step"]}
        side += r["train_s"] + r["eval_s"]
    train = {r["step"]: (r["tokens"], r["loss"], r["t"] - t0) for r in recs if r.get("event") == "train"}
    hashes = [r["sha256"] for r in sorted((r for r in recs if r.get("event") == "data_hash"), key=lambda r: r["step"])]
    msgs = [r["msg"] for r in recs if r.get("event") == "msg"]
    return {"branch": out, "final": next(r for r in recs if r.get("event") == "final"), "train": train,
            "config": recs[0], "hashes": hashes, "stream_restarts": sum("exhausted" in m for m in msgs)}


def run_of(label, seed=0, suffix=""):
    for base in ((R8, R9) if label in ("B_r8", "H2", "H2r", "CB") else (R9,)):  # round-9 seed-1 reruns live in R9
        r = load_run(os.path.join(base, f"wsd_{label}{suffix}_s{seed}.jsonl"))
        if r:
            return r
    return None


def interp_log(pts, x):
    """y at x, log-linear in x between measured points; None outside the range."""
    pts = sorted(pts)
    if not pts or x < pts[0][0] * (1 - 1e-9) or x > pts[-1][0] * (1 + 1e-9):
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (math.log(x) - math.log(x0)) * (y1 - y0) / (math.log(x1) - math.log(x0))
    return pts[-1][1]


def smooth_at(train, step):
    """5-point centred mean (100 steps) of logged train loss around the logged step nearest to `step`."""
    st = sorted(train)
    i = int(np.argmin([abs(s - step) for s in st]))
    lo, hi = max(0, i - 2), min(len(st), i + 3)
    return float(np.mean([train[s][1] for s in st[lo:hi]])), st[i]


def main():
    res = {"valbytes": VB}
    L = ["# Round 9: why the two-level head falls behind, and can we fix it? (2026-11-07)", "",
         "Generated by `PYTHONIOENCODING=utf8 python round9_analyze.py` (Windows Python) from `runs/round8/` and "
         "`runs/round9/`; method choices are pinned in the script's docstring. Hand-written notes at the end come "
         "from `runs/round9/round9_notes.md`. Every arm is seed 0 unless a seed list says otherwise.", ""]
    PA = json.load(open(os.path.join(R9, "partA.json"))) if os.path.exists(os.path.join(R9, "partA.json")) else None
    FR = json.load(open(os.path.join(R9, "freq_report.json")))
    AR = json.load(open(os.path.join(R9, "adapt_report.json")))
    G = json.load(open(os.path.join(R9, "gates.json"))) if os.path.exists(os.path.join(R9, "gates.json")) else {}
    SM = json.load(open(os.path.join(R9, "smoke_check_tiny.json")))["arms"] if os.path.exists(os.path.join(R9, "smoke_check_tiny.json")) else {}
    SM384 = json.load(open(os.path.join(R9, "smoke_check_d384.json")))["arms"] if os.path.exists(os.path.join(R9, "smoke_check_d384.json")) else {}
    _g = os.path.join(R9, "gates_H2tm384_H2a384_H2atm384.json")  # written by round9_gates.py H2tm384 H2a384 H2atm384
    G384 = json.load(open(_g)) if os.path.exists(_g) else {}

    RS = {l: run_of(l) for l in ["B_r8", "H2", "H2r", "CB"] + FIX}
    RS = {l: v for l, v in RS.items() if v}
    RS1 = {l: run_of(l, 1) for l in ["B_r8"] + ["H2", "H2tm", "H2a", "H2atm", "H2u"]}
    RS1 = {l: v for l, v in RS1.items() if v}
    B = RS["B_r8"]
    stables = sorted(B["branch"])
    last = max(stables)

    # ------------------------------------------------------------------ Part A
    if PA:
        a1 = PA["A1_all"]
        L += ["## 1. Part A — analyses of the round-8 finals (no training)", "",
              f"### A1. Loss decomposition (full val set: {PA['n_targets']:,} targets, the rows train.py's full eval uses)", "",
              f"Self-check: B's 50,304-row NLL here {PA['check']['B_nll_all_mean']:.5f} vs logged {PA['check']['B_logged']:.5f}; "
              f"H2 {PA['check']['H2_mean']:.5f} vs logged 4.21367. B below is renormalised over the 50,257 real tokens "
              f"(changes B by {a1['B_pad_effect']:+.6f}). Class part = −log p(c_y | h), within = −log p(y | c_y, h), "
              "classes = round 8's k-means 1024. Δ = H2 − B.", "",
              "| | total | class part | within part |", "|---|---|---|---|",
              f"| B | {a1['B_total']:.4f} | {a1['B_class']:.4f} | {a1['B_within']:.4f} |",
              f"| H2 | {a1['H2_total']:.4f} | {a1['H2_class']:.4f} | {a1['H2_within']:.4f} |",
              f"| Δ (H2 − B) | {a1['d_total']:+.4f} | {a1['d_class']:+.4f} | {a1['d_within']:+.4f} |", "",
              f"Per target-frequency bucket (rank by training-set count over the 1,320M-token stream the 1200M runs see; "
              f"`round9_freq.py`). 'share of Δ' = the bucket's contribution to the overall Δtotal.", "",
              "| bucket (rank) | share of val targets | B total | H2 total | Δtotal | Δclass | Δwithin | share of Δ |",
              "|---|---|---|---|---|---|---|---|"]
        for name, s in zip(PA["buckets"], PA["A1_bucket"]):
            L.append(f"| {name} | {s['share']:.3f} | {s['B_total']:.4f} | {s['H2_total']:.4f} | {s['d_total']:+.4f} | "
                     f"{s['d_class']:+.4f} | {s['d_within']:+.4f} | {s['contrib_total'] / a1['d_total']:.0%} |")
        L += ["", "Per tercile of B's predictive entropy (nats; real-token distribution):", "",
              "| tercile | mean entropy | B total | Δtotal | Δclass | Δwithin |", "|---|---|---|---|---|---|"]
        for k, s in enumerate(PA["A1_entropy"]):
            L.append(f"| {['low', 'mid', 'high'][k]} | {s['entropy_mean']:.3f} | {s['B_total']:.4f} | {s['d_total']:+.4f} | "
                     f"{s['d_class']:+.4f} | {s['d_within']:+.4f} |")
        if PA.get("extra"):
            L += ["", "Fix arms (1200M finals), Δtotal vs B per bucket (top part = the top-level softmax term, "
                  "within = the within-cluster term; for H2a/H2atm frequent targets have no within term):", "",
                  "| arm | total | Δtotal | " + " | ".join(PA["buckets"]) + " |", "|---|---|---|" + "---|" * 5]
            for lab, x in PA["extra"].items():
                L.append(f"| {NICE.get(lab, lab)} | {x['total']:.4f} | {x['d_total']:+.4f} | " + " | ".join(f"{v:+.4f}" for v in x["d_total_bucket"]) + " |")
        L.append("")
        res["A1"] = PA

    # A2: decay gains
    L += ["### A2. Decay gains", "",
          "gain = (5-point smoothed train loss at the step the branch's decay starts) − (that branch's decayed val "
          "loss). Train loss is logged every 20 steps; the nearest logged step is used (branch steps 1144, 2289, 4578, "
          "9155, 18311 → logged 1140, 2280, 4580, 9160, 18320, i.e. ≤ 9 steps away; the 5-point window spans ±40 "
          "steps). Before = paired smoothed train Δ (arm − B) at that step; after = decayed val Δ.", "",
          "| arm | " + " | ".join(f"gain @{k / 1e6:.0f}M" for k in stables) + " |", "|---|" + "---|" * len(stables)]
    A2 = {}
    for l in ["B_r8", "H2", "H2r", "CB"] + [f for f in FIX if f in RS]:
        r = RS.get(l)
        if not r:
            continue
        g = {}
        for k in stables:
            if k not in r["branch"]:
                continue
            sm, used = smooth_at(r["train"], r["branch"][k]["branch_step"])
            g[k] = {"gain": sm - r["branch"][k]["loss"], "train_smoothed": sm, "logged_step": used}
        A2[l] = g
        L.append(f"| {NICE[l]} | " + " | ".join(f"{g[k]['gain']:.4f}" if k in g else "-" for k in stables) + " |")
    L += ["", "| arm | quantity | " + " | ".join(f"{k / 1e6:.0f}M" for k in stables) + " |", "|---|---|" + "---|" * len(stables)]
    gapH2 = {}
    for l in A2:
        if l == "B_r8":
            continue
        bef = [A2[l][k]["train_smoothed"] - A2["B_r8"][k]["train_smoothed"] for k in stables if k in A2[l]]
        aft = [RS[l]["branch"][k]["loss"] - B["branch"][k]["loss"] for k in stables if k in A2[l]]
        gg = [A2["B_r8"][k]["gain"] - A2[l][k]["gain"] for k in stables if k in A2[l]]
        L.append(f"| {NICE[l]} | Δ before decay (train) | " + " | ".join(f"{x:+.4f}" for x in bef) + " |")
        L.append(f"| {NICE[l]} | Δ after decay (val) | " + " | ".join(f"{x:+.4f}" for x in aft) + " |")
        L.append(f"| {NICE[l]} | gain_B − gain_arm | " + " | ".join(f"{x:+.4f}" for x in gg) + " |")
        if l == "H2":
            gapH2 = dict(zip(stables, gg))
    n_big = sum(v > 0.03 for v in gapH2.values())
    L += ["", f"Part C trigger (gain_B − gain_H2 > 0.03 at ≥ 2 branches): {n_big} branch(es) → "
          f"{'Part C runs' if n_big >= 2 else 'Part C not run'}.", ""]
    res["A2"] = {l: {str(k): v for k, v in g.items()} for l, g in A2.items()}
    res["A2_gap_H2"] = {str(k): v for k, v in gapH2.items()}
    res["partC_triggered"] = n_big >= 2

    if PA:
        A3 = PA["A3"]
        L += ["### A3. Embedding statistics (1200M finals)", "",
              "| | " + " | ".join(f"RMS U {b}" for b in PA["buckets"]) + " | ‖U‖_F | ‖class-mean part‖_F | ‖within part‖_F | class-mean share of ‖U‖² |",
              "|---|" + "---|" * 9]
        for lab in ("B_r8", "H2"):
            x = A3[lab]
            L.append(f"| {NICE[lab]} | " + " | ".join(f"{v:.4f}" for v in x["rms_bucket"]) +
                     f" | {x['fro_U']:.1f} | {x['fro_classmean']:.1f} | {x['fro_within']:.1f} | {x['classmean_share_of_sq']:.3f} |")
        L += ["", "64 val batches (4 rows each): RMS of the input embedding and of block 0's update after its alphas "
              "(attention, FFN, and their sum x1 − x0):", "",
              "| | RMS emb | RMS a_attn·attn | RMS a_ffn·ffn | RMS (x1 − x0) | (x1 − x0) / emb |", "|---|---|---|---|---|---|"]
        for lab in ("B_r8", "H2"):
            x = A3[lab]
            L.append(f"| {NICE[lab]} | {x['rms_input_emb']:.4f} | {x['rms_blk0_attn_after_alpha']:.4f} | "
                     f"{x['rms_blk0_ffn_after_alpha']:.4f} | {x['rms_blk0_total']:.4f} | {x['blk0_over_emb']:.2f} |")
        L += ["", "Per-copy alphas, mean |a| over the 128 channels (init 1/√12 = 0.2887):", "",
              "| | " + " | ".join(f"blk{i} attn / ffn" for i in range(6)) + " |", "|---|" + "---|" * 6]
        for lab in ("B_r8", "H2"):
            L.append(f"| {NICE[lab]} | " + " | ".join(f"{c['a_attn']['mean_abs']:.3f} / {c['a_ffn']['mean_abs']:.3f}" for c in A3[lab]["alphas"]) + " |")
        L.append("")

    # ------------------------------------------------------------------ branch table
    floor_sp = {}
    for l in RS1:
        if l in RS:
            floor_sp[l] = abs(RS1[l]["branch"][last]["loss"] - RS[l]["branch"][last]["loss"])
    floor = max(floor_sp.values()) if floor_sp else 0.005
    res["floor"] = {"value": floor, "from": floor_sp or "default 0.005 (one seed)"}

    def dB(l, k, R1=False):
        ds = [RS[l]["branch"][k]["loss"] - B["branch"][k]["loss"]]
        if l in RS1 and "B_r8" in RS1 and k in RS1[l]["branch"]:
            ds.append(RS1[l]["branch"][k]["loss"] - RS1["B_r8"]["branch"][k]["loss"])
        return float(np.mean(ds)), ds

    L += ["## 2. Branch table (tiny proxy, d=128)", "",
          "Val loss nats/token (BPB in brackets). Δ vs B paired per seed (mean over the seeds both have; per-seed values "
          "in brackets when 2). Round-8 rows are references, unchanged.", "",
          "| arm | " + " | ".join(f"{k / 1e6:.0f}M" for k in stables) + " |", "|---|" + "---|" * len(stables)]
    res["arms"] = {}
    for l in RS:
        L.append(f"| {NICE[l]} | " + " | ".join(f"{RS[l]['branch'][k]['loss']:.4f} ({RS[l]['branch'][k]['loss'] * BPN:.4f})"
                                                if k in RS[l]["branch"] else "-" for k in stables) + " |")
        res["arms"][l] = {"loss": {str(k): v["loss"] for k, v in RS[l]["branch"].items()},
                          "bpb": {str(k): v["loss"] * BPN for k, v in RS[l]["branch"].items()},
                          "wall_own_s": {str(k): v["wall"] for k, v in RS[l]["branch"].items()},
                          "tokens": {str(k): v["tokens"] for k, v in RS[l]["branch"].items()},
                          "final": RS[l]["final"]}
    for l in RS1:
        L.append(f"| {NICE[l]} seed 1 | " + " | ".join(f"{RS1[l]['branch'][k]['loss']:.4f}" if k in RS1[l]["branch"] else "-" for k in stables) + " |")
        res["arms"].setdefault(l, {})["loss_s1"] = {str(k): v["loss"] for k, v in RS1[l]["branch"].items()}
    for l in RS:
        if l == "B_r8":
            continue
        cells = []
        for k in stables:
            if k not in RS[l]["branch"]:
                cells.append("-")
                continue
            m, ds = dB(l, k)
            cells.append(f"{m:+.4f}" + (" (" + ", ".join(f"{x:+.4f}" for x in ds) + ")" if len(ds) > 1 else ""))
        L.append(f"| Δ {NICE[l]} − B | " + " | ".join(cells) + " |")
        res["arms"][l]["delta_vs_B"] = {str(k): dB(l, k)[0] for k in stables if k in RS[l]["branch"]}
    L.append("")
    cand = [l for l in ("H2", "H2tm", "H2a", "H2atm") if l in RS and last in RS[l]["branch"]]
    H2S = min(cand, key=lambda l: RS[l]["branch"][last]["loss"])
    res["H2_star"] = H2S
    L += [f"**H2\\* = {NICE[H2S]}** (lowest 1200M val loss among {', '.join(NICE[c] for c in cand)}: "
          + ", ".join(f"{NICE[c]} {RS[c]['branch'][last]['loss']:.4f}" for c in cand) + ").", ""]

    # ------------------------------------------------------------------ cost table
    acct = {k.split(" ", 1)[1]: v for k, v in G.items() if k.startswith("acct ")}
    L += ["## 3. Cost per arm (tiny)", "",
          "Head forward FLOPs/token: B 2·V·d (V = 50,304), H2-shape heads 2·(C + L)·d with L the largest cluster, "
          "H2a/H2atm 2·(4096 + 1024 + 46)·d; tied-mean arms add the ū segment sum (V·d adds per 16,384-token "
          "micro-batch, ~393 FLOPs/token). Smoke = 200-step runs back to back today (`r9smoke.sh`); wall = the "
          "1200M branch's own-schedule wall (see docstring) and the whole run's wall.", "",
          "| arm | params | new params | head fwd FLOPs/tok | × B | train FLOPs/tok | smoke tok/s | × B | step ms | peak VRAM GB | smoke est. full-run min | own wall min (1200M) | whole-run min |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    from_r8 = json.load(open(os.path.join(R8, "gates.json")))["accounting"]
    for l in RS:
        a = ARM_OF[l]
        ac = acct.get(a) or from_r8.get(a)
        sm = SM.get(a)
        if not ac:
            continue
        hf = ac["head_fwd_flops_per_token"]
        hb = from_r8["tB"]["head_fwd_flops_per_token"]
        L.append(f"| {NICE[l]} | {ac['total_params']:,} | {ac['new_params']:,} | {hf:,.0f} | {hf / hb:.4f} | "
                 f"{ac['train_flops_per_token']['total']:,.0f} | "
                 + (f"{sm['tok_s']:,.0f} | {sm['speed_vs_B']:.2f} | {sm['step_ms']:.0f} | {sm['peak_mem_gb']:.2f} | {sm['est_full_min']:.0f} | " if sm else "- | - | - | - | - | ")
                 + f"{RS[l]['branch'][last]['wall'] / 60:.1f} | {RS[l]['final']['wall_s'] / 60:.1f} |")
        res["arms"][l]["cost"] = {"accounting": ac, "smoke": sm, "head_ratio": hf / hb}
    L.append("")
    head_ratio = {l: res["arms"][l].get("cost", {}).get("head_ratio") for l in RS}

    # ------------------------------------------------------------------ Part C
    L += ["## 4. Parts C, D, E", ""]
    rc = run_of(H2S, 0, "_dec2")
    if res["partC_triggered"]:
        L += ["### Part C — decay twice as long (20% of the steps so far), H2*", ""]
        if rc:
            L += ["| branch | normal decay | 2× decay | change | gap to recover (gain_B − gain_H2) |", "|---|---|---|---|---|"]
            for k in sorted(rc["branch"]):
                n = RS[H2S]["branch"][k]["loss"]
                L.append(f"| {k / 1e6:.0f}M | {n:.4f} | {rc['branch'][k]['loss']:.4f} | {rc['branch'][k]['loss'] - n:+.4f} | {gapH2.get(k, float('nan')):+.4f} |")
            res["partC"] = {str(k): {"normal": RS[H2S]["branch"][k]["loss"], "dec2": v["loss"]} for k, v in rc["branch"].items()}
        else:
            L.append("Triggered but not run (see notes).")
        L.append("")
    else:
        L += ["### Part C — not run (A2 trigger not met)", ""]

    # ------------------------------------------------------------------ Part D
    rd = run_of(H2S, 0, "_4800M")
    Bw = B["branch"][last]["wall"]
    Bfl = B["final"]["flops_per_token"]["total"] * B["branch"][last]["tokens"]
    if rd:
        fpt = rd["final"]["flops_per_token"]["total"]
        L += [f"### Part D — matched compute: {NICE[H2S]} with its stable phase to 4,800M", "",
              f"Unique training tokens: {FR['train_docs_total']:,} documents in train shards 000–012 (parquet footers) × "
              f"{FR['tokens_per_doc_incl_eot']:.1f} tokens/doc = {FR['est_unique_train_tokens'] / 1e9:.2f}B "
              f"(the run needs {FR['partD_tokens_needed'] / 1e9:.2f}B incl. the final decay); stream restarts logged in "
              f"the run: {rd['stream_restarts']}.", "",
              "| branch | tokens | val loss | own wall min | Δ vs B_r8 1200M (4.1228) |", "|---|---|---|---|---|"]
        for k in sorted(rd["branch"]):
            v = rd["branch"][k]
            L.append(f"| {k / 1e6:.0f}M | {v['tokens'] / 1e6:.0f}M | {v['loss']:.4f} | {v['wall'] / 60:.1f} | {v['loss'] - B['branch'][last]['loss']:+.4f} |")
        L += ["", f"Rerun check: the 1200M branch of this run repeats {NICE[H2S]}'s Part-B 1200M result on the same "
              f"data and recipe: {rd['branch'][last]['loss']:.4f} vs {RS[H2S]['branch'][last]['loss']:.4f} "
              f"(|Δ| {abs(rd['branch'][last]['loss'] - RS[H2S]['branch'][last]['loss']):.4f}, GPU nondeterminism).", ""]
        ptsW = [(v["wall"], v["loss"]) for v in rd["branch"].values()]
        ptsT = [(v["tokens"], v["loss"]) for v in rd["branch"].values()]
        eq_w = interp_log(ptsW, Bw)
        sm_ratio = (SM.get(ARM_OF[H2S], {}).get("tok_s") or 0) / (SM.get("tB", {}).get("tok_s") or 1)
        eq_wn = interp_log(ptsT, TOK1200 * sm_ratio) if sm_ratio else None
        eq_f = interp_log(ptsT, Bfl / fpt)
        Bl = B["branch"][last]["loss"]
        D = {"equal_tokens": RS[H2S]["branch"][last]["loss"] - Bl,
             "equal_wall_raw": None if eq_w is None else eq_w - Bl,
             "equal_wall_smoke_norm": None if eq_wn is None else eq_wn - Bl,
             "equal_flops": None if eq_f is None else eq_f - Bl,
             "B_wall_own_min": Bw / 60, "B_flops": Bfl, "arm_flops_per_token": fpt,
             "tokens_at_B_flops": Bfl / fpt, "smoke_ratio": sm_ratio, "tokens_at_B_time_smoke": TOK1200 * sm_ratio}
        res["partD"] = D
        f = lambda x: "not reached" if x is None else f"{x:+.4f}"
        L += ["| comparison vs B_r8 at 1200M (4.1228) | H2* loss | Δ |", "|---|---|---|",
              f"| (i) equal tokens (1,320M) | {RS[H2S]['branch'][last]['loss']:.4f} | {D['equal_tokens']:+.4f} |",
              f"| (ii) equal wall-clock, raw (B's own wall {Bw / 60:.1f} min) | {'-' if eq_w is None else f'{eq_w:.4f}'} | {f(D['equal_wall_raw'])} |",
              f"| (ii') equal wall-clock, smoke-normalised ({TOK1200 * sm_ratio / 1e6:,.0f}M tokens at {sm_ratio:.2f}× B's tok/s) | {'-' if eq_wn is None else f'{eq_wn:.4f}'} | {f(D['equal_wall_smoke_norm'])} |",
              f"| (iii) equal training FLOPs ({Bfl:.3e} → {Bfl / fpt / 1e6:,.0f}M tokens) | {'-' if eq_f is None else f'{eq_f:.4f}'} | {f(D['equal_flops'])} |", ""]
        res["partD"]["branches"] = {str(k): v for k, v in rd["branch"].items()}
    else:
        L += ["### Part D — not finished", ""]

    # ------------------------------------------------------------------ Part E
    b384 = run_of("B384")
    e_lab = {"H2": "H2_384", "H2tm": "H2tm384", "H2a": "H2a384", "H2atm": "H2atm384"}[H2S]
    h384 = run_of(e_lab)
    if b384:
        L += ["### Part E — d=384 (round-5 part-B B shape: 6 unique blocks, tied, WSD 300M, branches 150M/300M)", "",
              f"Classes and frequency split are the d=128 ones (from B_r8), not rebuilt. Arm {NICE[e_lab]} = H2*'s head on the B384 backbone.", "",
              "| arm | 150M | 300M | smoke tok/s | × B | whole-run tok/s | own wall min (300M) |", "|---|---|---|---|---|---|---|"]
        E = {}
        for lab, r in (("B384", b384), (e_lab, h384)):
            if not r:
                continue
            sm = SM384.get(ARM_OF[lab], {})
            L.append(f"| {NICE[lab]} | " + " | ".join(f"{r['branch'][k]['loss']:.4f}" for k in sorted(r["branch"])) +
                     f" | {sm.get('tok_s', 0):,.0f} | {sm.get('speed_vs_B') or 0:.2f} | {r['final']['tok_s']:,.0f} | {r['branch'][max(r['branch'])]['wall'] / 60:.1f} |")
            E[lab] = {str(k): v for k, v in r["branch"].items()}
        if h384:
            ks = sorted(b384["branch"])
            dE = {k: h384["branch"][k]["loss"] - b384["branch"][k]["loss"] for k in ks}
            L.append(f"| Δ | " + " | ".join(f"{dE[k]:+.4f}" for k in ks) + " | | | | |")
            bw = b384["branch"][max(ks)]["wall"]
            ptsB = [(v["wall"], v["loss"]) for v in b384["branch"].values()]
            hw = h384["branch"][max(ks)]["wall"]
            bl_at = interp_log(ptsB, hw)  # B384's loss at H2*384's wall (H2*384 is faster, so inside B's range)
            eqw = None if bl_at is None else h384["branch"][max(ks)]["loss"] - bl_at
            hl = h384["branch"][max(ks)]["loss"]
            b_first = b384["branch"][min(ks)]
            # if H2*384 finishes before B384's first branch, no interpolation exists; give the assumption-free
            # comparison instead: H2*384's final vs B384's first branch, which B reached LATER (more wall-clock).
            bound = hl - b_first["loss"] if (bl_at is None and hw <= b_first["wall"]) else None
            L += ["", f"Head forward FLOPs/token × B384: {G384.get('acct ' + e_lab, {}).get('head_ratio_to_B', float('nan')):.4f} "
                  f"(`gates_H2tm384_H2a384_H2atm384.json`). Own-schedule wall: B384 150M {b_first['wall'] / 60:.2f} min, 300M {bw / 60:.2f} min; "
                  f"{NICE[e_lab]} 300M {hw / 60:.2f} min.", "",
                  f"Equal wall-clock: in {NICE[e_lab]}'s {hw / 60:.1f} min B384 reaches "
                  + (f"a loss of {bl_at:.4f} (log-linear in its own-schedule wall between its 150M and 300M branches) → "
                     f"Δ {eqw:+.4f}." if bl_at is not None else
                     f"less than its 150M branch (outside its measured range; no extrapolation). Assumption-free comparison: "
                     f"{NICE[e_lab]}'s 300M result ({hl:.4f}, {hw / 60:.1f} min) vs B384's 150M result ({b_first['loss']:.4f}, "
                     f"{b_first['wall'] / 60:.1f} min, i.e. B had more time): Δ {bound:+.4f}. Since a decayed branch taken earlier "
                     f"has seen fewer tokens, B384 at {hw / 60:.1f} min would be worse still, so the equal-wall Δ is ≤ {bound:+.4f} "
                     "(a bound, assuming loss falls with tokens)."), ""]
            res["partE"] = {"runs": E, "delta": {str(k): v for k, v in dE.items()}, "equal_wall_delta": eqw,
                            "equal_wall_delta_bound": bound, "arm": e_lab,
                            "wall_min": {"B384": {str(k): v["wall"] / 60 for k, v in b384["branch"].items()},
                                         e_lab: {str(k): v["wall"] / 60 for k, v in h384["branch"].items()}}}
    else:
        L += ["### Part E — not run", ""]

    # ------------------------------------------------------------------ verdicts
    V = []
    f = lambda x: "not reached" if x is None else f"{x:+.4f}"
    d1200 = {l: res["arms"][l]["delta_vs_B"][str(last)] for l in RS if l != "B_r8" and str(last) in res["arms"][l].get("delta_vs_B", {})}
    closed = [l for l in FIX if l in d1200 and d1200[l] <= 0.010 and (head_ratio.get(l) or 9) <= 0.15]
    V.append(f"- **Gap closed** (Δ at 1200M ≤ +0.010 with head FLOPs ≤ 0.15× B): " + (", ".join(NICE[l] for l in closed) if closed else "no arm") + ".")
    h2 = RS["H2"]["branch"][last]["loss"]
    part = [l for l in FIX if l in RS and h2 - RS[l]["branch"][last]["loss"] > 0.02]
    V.append(f"- **Partial fix** (beats round-8 H2 by > 0.02 at 1200M): " + (", ".join(f"{NICE[l]} ({RS[l]['branch'][last]['loss'] - h2:+.4f})" for l in part) if part else "no arm")
             + "; vs H2: " + ", ".join(f"{NICE[l]} {RS[l]['branch'][last]['loss'] - h2:+.4f}" for l in FIX if l in RS) + ".")
    if "partD" in res:
        D = res["partD"]
        win = D["equal_wall_raw"] is not None and D["equal_wall_raw"] <= -0.010
        V.append(f"- **Matched-compute win** (H2* at B_r8's wall-clock ≥ 0.010 below B): {'yes' if win else 'no'} "
                 f"(raw {f(D['equal_wall_raw'])}, smoke-normalised {f(D['equal_wall_smoke_norm'])}; equal FLOPs {f(D['equal_flops'])}).")
    if "partE" in res:
        E = res["partE"]
        d300 = E["delta"].get(str(max(int(k) for k in E["delta"])))
        bd = E.get("equal_wall_delta_bound")
        by_bound = E["equal_wall_delta"] is None and bd is not None and bd < -0.010
        rw = (d300 is not None and d300 <= 0.010) or (E["equal_wall_delta"] is not None and E["equal_wall_delta"] < -0.010) or by_bound
        V.append(f"- **Regime win** (d=384: Δ at 300M ≤ +0.010, or equal-wall-clock win > 0.010): {'yes' if rw else 'no'} "
                 f"(Δ300M {d300:+.4f}; equal wall " + (f"{E['equal_wall_delta']:+.4f}" if E["equal_wall_delta"] is not None else
                 (f"≤ {bd:+.4f}, a bound: {NICE[E['arm']]}'s 300M result vs B384's 150M result, which B reached with more wall-clock" if bd is not None else "n/a"))
                 + (f". The equal-token gap ({d300:+.3f}) is about the d=128 one ({d1200.get(H2S, float('nan')):+.3f}); the win comes from "
                    f"{SM384.get(ARM_OF[E['arm']], {}).get('speed_vs_B') or float('nan'):.1f}× tok/s at a horizon where B384's 150M→300M doubling is worth "
                    f"{float(E['runs']['B384'][str(max(int(k) for k in E['runs']['B384']))]['loss']) - float(E['runs']['B384'][str(min(int(k) for k in E['runs']['B384']))]['loss']):+.3f} nats"
                    if by_bound else "") + ").")
    if PA:
        a1 = PA["A1_all"]
        within_most = a1["d_within"] > 0.5 * a1["d_total"]
        freq_share = sum(s["contrib_total"] for s in PA["A1_bucket"][:3]) / a1["d_total"]
        tm_gain = h2 - RS["H2tm"]["branch"][last]["loss"] if "H2tm" in RS else None
        a_gain = h2 - RS["H2a"]["branch"][last]["loss"] if "H2a" in RS else None
        Ha = (tm_gain is not None and tm_gain > 0.02) or within_most
        Hb = (a_gain is not None and a_gain > 0.02) or freq_share > 0.5
        V.append(f"- **H-a (tied embeddings)**: {'supported' if Ha else 'not supported'} — H2tm vs H2 "
                 f"{'n/a' if tm_gain is None else f'{-tm_gain:+.4f}'} (needs < −0.02); within part = {a1['d_within'] / a1['d_total']:.0%} of A1's Δ (needs > 50%).")
        V.append(f"- **H-b (frequent tokens)**: {'supported' if Hb else 'not supported'} — H2a vs H2 "
                 f"{'n/a' if a_gain is None else f'{-a_gain:+.4f}'} (needs < −0.02); top-4096 buckets carry {freq_share:.0%} of A1's Δ (needs > 50%).")
        res["hyp"] = {"H-a": Ha, "H-b": Hb, "within_share": a1["d_within"] / a1["d_total"], "freq_share": freq_share}
    gapmax = sum(v > 0.03 for v in gapH2.values())
    if res["partC_triggered"] and "partC" in res:
        rec = np.mean([gapH2[int(k)] and (v["normal"] - v["dec2"]) / gapH2[int(k)] for k, v in res["partC"].items()])
        Hc = rec > 0.5
        V.append(f"- **H-c (LR decay)**: {'supported' if Hc else 'not supported'} — the decay-gain gap exceeds 0.03 at {gapmax} branch(es); "
                 f"2× decay recovers {rec:.0%} of it on average (needs > 50%).")
    else:
        V.append(f"- **H-c (LR decay)**: not supported — the decay-gain gap exceeds 0.03 at {gapmax} branch(es) (needs ≥ 2, "
                 "then Part C recovering > half); " + ", ".join(f"{k / 1e6:.0f}M {v:+.3f}" for k, v in gapH2.items()) + ".")
    res["verdicts"] = V
    # headline captions (data-driven, so they cannot hide a bad result)
    hs = res["arms"].get(H2S, {})
    dt = hs.get("delta_vs_B", {}).get(str(last))
    cap = (f"Tiny proxy (d=128, 6 blocks x 2, WSD, seed 0). H2* = {NICE[H2S]}, the best two-level head at 1200M. "
           f"At equal tokens (1,320M): Δ {dt:+.3f} nats vs B_r8 ({B['branch'][last]['loss']:.3f}).")
    if "partD" in res:
        D = res["partD"]
        cap += (f"\nAt B's own wall-clock ({D['B_wall_own_min']:.0f} min): {f(D['equal_wall_raw'])} (raw), "
                f"{f(D['equal_wall_smoke_norm'])} (smoke-normalised tok/s); at equal training FLOPs: {f(D['equal_flops'])}. "
                "Negative = H2* better.")
    res["headline_caption"] = cap
    if "partE" in res:
        E = res["partE"]
        k3 = str(max(int(k) for k in E["delta"]))
        res["headline_caption_d384"] = (f"d=384 (6 unique blocks, tied, WSD 300M, seed 0). {NICE[E['arm']]} vs B384: "
                                        f"{E['delta'][k3]:+.3f} nats at equal tokens (300M);\nat {NICE[E['arm']]}'s wall-clock B384 is "
                                        + ((f"outside its measured range. {NICE[E['arm']]}'s 300M result is {E['equal_wall_delta_bound']:+.3f}\nvs B384's 150M result, which B reached with more wall-clock "
                                            "(so the equal-wall Δ is at most this). Negative = H2*-384 better." if E.get("equal_wall_delta_bound") is not None else "outside its measured range.")
                                           if E["equal_wall_delta"] is None else f"{-E['equal_wall_delta']:+.3f} vs H2*-384 "
                                           f"(i.e. H2*-384 {E['equal_wall_delta']:+.3f} at equal wall-clock). Negative = H2*-384 better."))
    L = L[:4] + ["## Verdicts", ""] + V + [f"\nRerun floor: {floor:.4f} ({res['floor']['from']}).", ""] + L[4:]

    # ------------------------------------------------------------------ Part F
    dm = os.path.join(R9, "decode.md")
    if os.path.exists(dm):
        txt = open(dm, encoding="utf8").read().replace("\n## tiny", "\n#### tiny (d=128 proxy)").replace(
            "\n## big", "\n#### 600M systems config (28 x d1280, GQA 20/4, V 65,536; round 8's 'big')").replace("# Round 9 Part F", "Round 9 Part F")
        L += ["### Part F — exact greedy decoding with the two-level head", "", txt, ""]
        if os.path.exists(os.path.join(R9, "decode_fallback.json")):
            res["partF_fallback"] = json.load(open(os.path.join(R9, "decode_fallback.json")))
    else:
        L += ["### Part F — not run", ""]

    # ------------------------------------------------------------------ plots
    plots(RS, B, H2S, rd, b384, h384, e_lab, PA, res, SM, SM384, stables, last)
    L += ["## 5. Plots", "", "`loss_vs_tokens.png`, `delta_vs_B.png`, `decomposition_by_bucket.png`; headline `round9_headline.png` "
          "(+ `.svg`)" + (", `round9_headline_d384.png`" if h384 else "") + ".", "",
          "## 6. Commands", "", "```",
          "# WSL (AEON-Training), from the project folder; launch scripts set ulimit -c 0",
          "JAX_PLATFORMS=cpu python round9_freq.py            # train_counts.npy, freq_report.json (unique-token check)",
          "JAX_PLATFORMS=cpu python round9_classes.py         # adapt_freq4096.npy, adapt_tail4096_1024.npy, token_rank.npy",
          "JAX_PLATFORMS=cpu python round9_gates.py           # gates.txt / gates.json",
          "JAX_PLATFORMS=cpu python round9_gates.py H2tm384 H2a384 H2atm384   # d=384 accounting + gate 4 (gates_H2tm384_H2a384_H2atm384.json)",
          "bash r9smoke.sh tB tH2 tH2tm tH2a tH2atm; python round9_smoke_check.py --tag tiny --arms tB,tH2,tH2tm,tH2a,tH2atm",
          "bash r9one.sh d384 B384:0                       # Part E baseline (37 min)",
          "python round9_partA.py                          # A1 + A3 (round-8 finals) -> partA.json",
          "bash r9one.sh tiny H2tm:0,H2a:0,H2atm:0         # Part B fix arms",
          "bash r9one.sh D H2atm:0                         # Part D (H2* = H2atm), 87 min",
          "bash r9smoke.sh B H2atm384; python round9_smoke_check.py --tag d384 --arms B,H2atm384",
          "bash r9one.sh d384 H2atm384:0                   # Part E arm",
          "python round9_partA.py --extra H2tm:tH2tm:wsd_H2tm_s0,H2a:tH2a:wsd_H2a_s0,H2atm:tH2atm:wsd_H2atm_s0   # rerun: + fix-arm buckets",
          "python round9_decode.py fallback H2atm tH2atm wsd_H2atm_s0   # Part F fallback rate + exactness",
          "python round9_decode.py run 0 4096; python round9_decode.py run 1 4096; python round9_decode.py run 2 4096",
          "python round9_decode.py md                      # -> decode.md",
          "# (all of the above chained in one go: bash r9chain.sh one+D+H2atm:0 waitidle smoke+B+H2atm384 ... — see queue.log)",
          "# Windows Python",
          "PYTHONIOENCODING=utf8 python round9_analyze.py", "```", ""]
    notes = os.path.join(R9, "round9_notes.md")
    if os.path.exists(notes):
        L += [open(notes, encoding="utf8").read()]
    open(os.path.join(R9, "round9_summary.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(R9, "round9_results.json"), "w"), indent=1, default=float)
    print("\n".join(V))


def plots(RS, B, H2S, rd, b384, h384, e_lab, PA, res, SM, SM384, stables, last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GREY, STRONG = "#6f6e69", "#2a78d6"
    # validated (dataviz validate_palette.js --pairs all, light): blue / yellow / aqua / red pass the normal-vision floor;
    # worst CVD pair 6.9 (deutan, red vs aqua) is in the 6-8 band, so every muted arm also gets its own marker + direct label.
    MUTED = {"H2": "#eda100", "H2r": "#e87ba4", "CB": "#6250d6", "H2tm": "#1baf7a", "H2a": "#e34948", "H2atm": "#008300", "H2u": "#8a5a00"}
    col = lambda l: GREY if l == "B_r8" else (STRONG if l == H2S else MUTED.get(l, "#999999"))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # loss vs tokens
    fig, ax = plt.subplots(figsize=(8, 5))
    for l, r in RS.items():
        ks = sorted(r["branch"])
        ax.plot([r["branch"][k]["tokens"] for k in ks], [r["branch"][k]["loss"] for k in ks], MARK.get(l, "o") + "-", color=col(l),
                lw=2.5 if l in ("B_r8", H2S) else 1.5, ms=5, label=NICE[l])
    if rd:
        ks = sorted(rd["branch"])
        ax.plot([rd["branch"][k]["tokens"] for k in ks], [rd["branch"][k]["loss"] for k in ks], "s--", color=STRONG, lw=2, ms=5,
                label=f"{NICE[H2S]}, 4800M run")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("val loss (nats/token)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R9, "loss_vs_tokens.png"), dpi=130)
    plt.close(fig)
    # delta vs B
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=GREY, lw=1)
    for l, r in RS.items():
        if l == "B_r8":
            continue
        ks = [k for k in stables if k in r["branch"]]
        ax.plot([B["branch"][k]["tokens"] for k in ks], [r["branch"][k]["loss"] - B["branch"][k]["loss"] for k in ks], MARK.get(l, "o") + "-",
                color=col(l), lw=2.5 if l == H2S else 1.5, label=NICE[l])
    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("Δ val loss vs B (nats/token)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R9, "delta_vs_B.png"), dpi=130)
    plt.close(fig)
    # decomposition by bucket
    if PA:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(5)
        dc = [s["d_class"] for s in PA["A1_bucket"]]
        dw = [s["d_within"] for s in PA["A1_bucket"]]
        ax.bar(x - 0.2, dc, 0.38, color="#2a78d6", label="Δ class part")
        ax.bar(x + 0.2, dw, 0.38, color="#eb6834", label="Δ within part")
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xticks(x, [f"{b}\n{s['share']:.0%} of targets" for b, s in zip(PA["buckets"], PA["A1_bucket"])], fontsize=9)
        ax.set_ylabel("H2 − B (nats/token)")
        ax.set_xlabel("target frequency rank bucket")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(R9, "decomposition_by_bucket.png"), dpi=130)
        plt.close(fig)
    headline(RS, B, H2S, rd, res, SM, stables, last, "round9_headline", col, ARMSET=[l for l in RS if l not in ("H2r", "CB")], cap=res.get("headline_caption", ""))
    if b384 and h384:
        headline({"B_r8": b384, H2S: h384}, b384, H2S, None, res, SM384, sorted(b384["branch"]), max(b384["branch"]),
                 "round9_headline_d384", col, ARMSET=["B_r8", H2S], d384=(e_lab,), cap=res.get("headline_caption_d384", ""))


def headline(RS, B, H2S, rd, res, SM, stables, last, fname, col, ARMSET, d384=None, cap=""):
    NLC = chr(10)
    import matplotlib.pyplot as plt
    nice = dict(NICE)
    if d384:
        nice = {"B_r8": "B384", H2S: NICE[d384[0]]}
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1.7], hspace=0.34, wspace=0.30)
    axW, axT = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], sharey=None)
    Bl, Bw = B["branch"][last]["loss"], B["branch"][last]["wall"]
    curves = {}
    for l in ARMSET:
        r = RS[l]
        ks = sorted(r["branch"])
        pts = [(r["branch"][k]["wall"] / 60, r["branch"][k]["tokens"], r["branch"][k]["loss"]) for k in ks]
        if l == H2S and rd:
            pts += [(rd["branch"][k]["wall"] / 60, rd["branch"][k]["tokens"], rd["branch"][k]["loss"]) for k in sorted(rd["branch"]) if k > last]
        curves[l] = pts
        strong = l in ("B_r8", H2S)
        for ax, i in ((axW, 0), (axT, 1)):
            ax.plot([p[i] for p in pts], [p[2] for p in pts], MARK.get(l, "o") + "-", color=col(l), lw=2.6 if strong else 1.4,
                    ms=6 if strong else 4, alpha=1 if strong else 0.75, zorder=3 if strong else 2, label=nice.get(l, l))
    # shade where H2* is below B at equal minutes
    bw = [(p[0], p[2]) for p in curves["B_r8"]]
    hw = [(p[0], p[2]) for p in curves[H2S]]
    lo, hi = max(min(x for x, _ in bw), min(x for x, _ in hw)), min(max(x for x, _ in bw), max(x for x, _ in hw))
    if hi > lo:
        xs = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
        yb = np.array([interp_log(bw, x) for x in xs])
        yh = np.array([interp_log(hw, x) for x in xs])
        axW.fill_between(xs, yh, yb, where=yh < yb, color=col(H2S), alpha=0.12, lw=0, label=f"{nice[H2S]} below B at equal minutes")
        sgn = np.sign(yh - yb)
        cr = [xs[i] for i in range(1, len(xs)) if sgn[i] != sgn[i - 1]]
        for c in cr[:1]:
            axW.axvline(c, color=col(H2S), lw=0.8, ls=":")
            axW.annotate(f"crossover {c:.0f} min", (c, interp_log(bw, c)), xytext=(6, 18),
                         textcoords="offset points", fontsize=10, color="#333")
        if not cr:
            below = bool((yh < yb).all())
            axW.text(0.02, 0.04, f"no crossover in {lo:.0f}–{hi:.0f} min (the range both reached):" + NLC + f"{nice[H2S]} is "
                     + ("below" if below else "above") + " B throughout", transform=axW.transAxes, fontsize=10, color="#333", ha="left", va="bottom")
    elif d384:
        # no overlap in wall-clock: show the assumption-free pair (H2*'s last point vs B's first, reached later)
        hl_, bl_ = curves[H2S][-1], curves["B_r8"][0]
        axW.annotate("", xy=(hl_[0], hl_[2]), xytext=(bl_[0], bl_[2]),
                     arrowprops=dict(arrowstyle="<->", color="#333", lw=1, shrinkA=6, shrinkB=6))
        axW.annotate(f"{nice[H2S]} at {hl_[0]:.0f} min is {hl_[2] - bl_[2]:+.3f}\nvs B384 at {bl_[0]:.0f} min (more time)",
                     ((hl_[0] * bl_[0]) ** 0.5, (hl_[2] + bl_[2]) / 2), xytext=(0, 14), textcoords="offset points",
                     ha="center", fontsize=10.5, fontweight="bold", color=col(H2S))
    # B's final point and H2* at B's wall
    h_at = interp_log(hw, Bw / 60)
    if h_at is not None:
        dd = h_at - Bl
        axW.plot([Bw / 60], [h_at], "D", color=col(H2S), ms=9, zorder=4, mec="white", mew=1.2)
        axW.text(0.98, 0.56, f"B final: {Bl:.4f} at {Bw / 60:.0f} min", transform=axW.transAxes, ha="right", fontsize=10.5, color="#333")
        axW.text(0.98, 0.50, f"◆ {nice[H2S]} at B's {Bw / 60:.0f} min: {h_at:.4f} ({dd:+.4f})", transform=axW.transAxes, ha="right",
                 fontsize=11.5, fontweight="bold", color=col(H2S))
    else:
        axW.annotate(f"B final: {Bl:.4f} at {Bw / 60:.0f} min", (Bw / 60, Bl), xytext=(-14, 16), textcoords="offset points",
                     ha="right", va="bottom", fontsize=10, color="#333")
        axW.annotate(f"{nice[H2S]} stopped at {max(x for x, _ in hw):.0f} min (before B's {Bw / 60:.0f})", (Bw / 60, Bl),
                     xytext=(-14, -8), ha="right", va="top", textcoords="offset points", fontsize=10, color="#333")
    # Δ at the end of each line (right panel), vertically de-overlapped
    ends = []
    for l in ARMSET:
        r = RS[l]
        if last in r["branch"]:
            v = r["branch"][last]["loss"]
            ends.append([v, v, l, (f"B {Bl:.3f}" if l == "B_r8" else f"{v - Bl:+.3f}")])
    ends.sort()
    gap = 0.04 * (max(p[2] for c in curves.values() for p in c) - min(p[2] for c in curves.values() for p in c) + 0.06)
    for a, b in zip(ends, ends[1:]):
        if b[1] - a[1] < gap:
            b[1] = a[1] + gap
    for v, ypos, l, txt in ends:
        axT.annotate(txt, (B["branch"][last]["tokens"], v), xytext=(B["branch"][last]["tokens"] * 1.06, ypos),
                     textcoords="data", fontsize=9.5, color=col(l), va="center", fontweight="bold" if l == H2S else "normal",
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=5)
    if rd:
        kx = max(rd["branch"])
        axT.annotate(f"{kx / 1e6:.0f}M: {rd['branch'][kx]['loss']:.3f}\n({rd['branch'][kx]['loss'] - Bl:+.3f} vs B at 1200M)",
                     (rd["branch"][kx]["tokens"], rd["branch"][kx]["loss"]), xytext=(0, -12), textcoords="offset points",
                     ha="right", va="top", fontsize=9.5, color=col(H2S))
    ylo = min(p[2] for c in curves.values() for p in c) - 0.09
    yhi = max(p[2] for c in curves.values() for p in c) + 0.03
    for ax, title, xl in ((axW, "loss vs wall-clock", "training wall-clock (minutes, own schedule)"),
                          (axT, "loss vs tokens", "training tokens")):
        ax.set_xscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.22)
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * BPN, lambda v: v / BPN))
        sec.set_ylabel("bits per byte (same scale)")
    x0, x1 = axT.get_xlim()
    axT.set_xlim(x0, x1 * 1.35)
    axW.set_ylabel("val loss (nats/token)")
    axW.legend(frameon=False, fontsize=9.5, loc="upper right")
    # caption + table
    axC = fig.add_subplot(gs[1, :])
    axC.axis("off")
    axC.text(0, 1.02, cap, fontsize=12, va="top")
    rows = []
    for l in ARMSET:
        a = ARM_OF[d384[0]] if (d384 and l == H2S) else ("B" if (d384 and l == "B_r8") else ARM_OF[l])
        sm = SM.get(a, {})
        hr = sm.get("head_flops_vs_B")
        rows.append([nice.get(l, l), f"{hr:.3f}×" if hr else "-", f"{sm['speed_vs_B']:.2f}×" if sm.get("speed_vs_B") else "-"])
    tb = axC.table(cellText=rows, colLabels=["arm", "head FLOPs/token (× B)", "training tok/s (× B, smoke)"],
                   loc="lower left", cellLoc="left", colLoc="left", bbox=[0, -0.12, 0.55, 0.66])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    for c in tb.get_celld().values():
        c.set_edgecolor("#dddddd")
    fig.savefig(os.path.join(R9, fname + ".png"), dpi=112, bbox_inches="tight")
    fig.savefig(os.path.join(R9, fname + ".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
