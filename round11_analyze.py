"""Round 11 analysis: is the two-level head cheap AND good? Writes runs/round11/round11_summary.md, round11_results.json and
every figure (headline last). Windows Python:  PYTHONIOENCODING=utf8 python round11_analyze.py

Decision rules, pinned 2026-11-09 18:20 before any run they depend on had finished (anything decided after seeing a
result is labelled post-hoc in the summary):

Accounting (Part A and everywhere): training FLOPs of a reported branch = the run's logged flops_per_token["total"]
  (model.flops_per_token: 6N rule + causal attention; the flat head counts the padded vocab, 6 * d * 50,304 for GPT-2
  arms; two-level heads count 3x their forward FLOPs; re-scored heads add 3 * 2 * M * L * d) x (stable + decay tokens)
  = flops_per_token x (branch_step + decay_steps) x 65,536. The user's table used stable tokens only; Part A reports both
  and says so. D arms (round-10 n-gram) are excluded from Part A by the user's scope decision.
Part B horizons: F8k-384 stable = 9,633 steps (631,308,288 tokens), branch at 5,026 steps (329,383,936); F4k-384 stable
  10,674 steps (699,531,264), branch 5,570 steps (365,035,520). Each = the step count S whose S + round(0.1 S) total steps
  x 65,536 x fpt(arm) is closest to H3*-384's 575M run (8,774 + 877 steps) or its 300M branch (4,578 + 458 steps) at
  fpt(H3s384) (ties rounded up). Matched FLOPs = within 0.02% (stated per run).
  Metric: BPB on the shared FineWeb-Edu val bytes (round11_bytes.py). GPT-2 arms: the logged val loss x tokens /
  (bytes x ln 2) over the 23,160,213 scored bytes; other tokenizers: the byte-aligned evaluator (each GPT-2 val row's
  first token is context, the rest is scored; windows of 1,025 with >= 512 tokens of in-row context). Any byte the
  evaluator cannot score (a row cut inside a UTF-8 character) is reported.
  Verdict: two-level wins if BPB(H3*-384) <= BPB(F8k-384) - 0.003 at matched FLOPs (575M run vs the F8k final branch;
  also reported at the 300M-matched branch), small vocab wins if the reverse, else tie (then decode ms per byte).
Part C: tH3rs16 (M = 16) first. Run M = 64 only if tH3rs16 beats H3k8c4 (wsd_H3k8c4_s0, 4.1552) by more than 0.010 at
  1200M. Best M = the lower 1200M loss among the M arms run. The best M then runs at d=384 (H3rs<M>_384, 575M,
  branches 300M / 575M). Verdicts: gap closed if Δ vs B_r8 at 1200M <= +0.010 with head FLOPs <= 0.30x B's head;
  partial if better than H3k8c4 by more than 0.010; no help otherwise. Hit fraction = share of tail targets (rank >= the
  K singletons) whose cluster is among the top M, over the 4,880 x 1,024 val targets.
Part D (optional): H3s512 (or H3rs<best M>_512 if Part C is "partial" or better) stable 5,484 steps (359,399,424 tokens).
Part E models: d=384: B384 (wsd_B384_300M_s0 re-run, weights kept), H3*-384 at 300M (wsd_H3s384_300M_s0 re-run) and 575M
  (round-10 wsd_H3s384_575M_s0), F8k-384 (both branches), the best Part-C d=384 arm if run. d=128: B_r8 and H3k8c4 at 1200M
  (+ tH3rs arms in E5). Part A/B tables use the original logged losses; Part E uses the re-run checkpoints (both are shown
  where they overlap; a re-run more than 0.01 from its original is stopped and reported).
  Every E comparison at equal tokens (H3*-384 300M vs B384 300M) and at equal FLOPs (H3*-384 575M vs B384 300M; their
  FLOPs differ by 5%: 6.72e16 vs 6.39e16, stated). F8k-384 is compared at equal FLOPs only (equal tokens is not defined
  across tokenizers) and is left out of every per-GPT-2-token-bucket table.
  E1: BPB on FineWeb-Edu val, WikiText-103 test (raw, documents = articles split at " = Title = " headings, joined by EOT),
      PG-19 test (emozilla/pg19-test mirror; books in dataset order until >= 1,000,000 GPT-2 tokens); rows of 1,025 GPT-2
      tokens, contiguous, EOT between documents; other tokenizers byte-aligned as Part B. OOD holds if H3*-384 (575M) BPB <=
      B384 BPB - 0.003 on both OOD sets.
  E2: BLiMP, all 67 subsets; a pair is correct if sum log p(good) > sum log p(bad), each sentence scored after one EOT, no
      length normalisation; groups = the dataset's linguistics_term field. A difference > 1.0 point overall is real.
  E3: LAMBADA (EleutherAI/lambada_openai test): context = text up to the last space, target = " " + last word. Accuracy =
      every target token is the argmax of the full next-token distribution (teacher forced = greedy until the word is
      complete; lm-eval-harness is_greedy). Word NLL in bits per UTF-8 byte of the target string. Split by the training-set
      rank of the target's first GPT-2 token (< 4,096 vs >= 4,096), for every model.
  E4: bios (round11_bios.py). Exposures = (stable + decay steps) x 64 x 1,025 stream tokens / (N x 55.636 tokens per bio
      incl. EOT). N = 20,000: 1,542 stable steps; N = 100,000: 7,710 stable steps (100 exposures each); H3k8c4 equal-FLOPs at
      N = 100,000: 18,303 stable steps (237.4 exposures = 100 x fpt(tB) / fpt(tH3k8c4)). Knowledge bits = sum over people
      and attributes of max(0, log2|A| - NLL_bits(value tokens | canonical probe)); greedy exact match = teacher-forced
      argmax of the full distribution at every value token. All N people unless eval > 10 min (then a fixed 5,000 sample,
      scaled, stated). No knowledge penalty if H3k8c4 >= 95% of B's bits at equal exposures for both N; more tokens -> more
      knowledge if the equal-FLOPs H3k8c4 run stores more bits than B at N = 100,000.
  E5: L = 1,024. E5a: the first 2,000 FineWeb-Edu val rows, tokens 1..512 repeated (row = 512 + 512 tokens, after one EOT
      context token); E5b: 2,000 rows of 512 tokens drawn uniformly (np.random.default_rng(11)) from training ranks >= 4,096,
      repeated. NLL per token on copy 1 and copy 2 by round-9 A1 rank bucket; Δ = two-level - flat at equal tokens and at
      equal FLOPs. Prior-only if the rank >= 4,096 Δ on copy 2 <= 0.02 nats and <= 1/3 of the copy-1 Δ (E5a); can't copy if
      the E5b copy-2 Δ > 0.05 nats (then the tH3rs arms are checked).
Seeds: a verdict within ±0.003 BPB (B, E1), ±0.010 nats (C, D), ±1 point (E2) or ±5% of bits (E4) of its threshold runs
  seed 1 for both sides; the seed spread is then the floor.
Colours: round 10's validated palette (blue #2a78d6, yellow #eda100, aqua #1baf7a, red #e34948, grey #6f6e69 for flat
  GPT-2 baselines), at most four coloured series per figure.
"""
import glob
import json
import math
import os

import numpy as np

from round9_analyze import BPN, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
RD = {r: os.path.join(HERE, "runs", f"round{r}") for r in (5, 8, 9, 10, 11)}
OUT = RD[11]
GREY, BLUE, YEL, AQUA, RED = "#6f6e69", "#2a78d6", "#eda100", "#1baf7a", "#e34948"
LN2 = math.log(2)
VAL_BYTES = 23_160_213


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf8")] if os.path.exists(p) else []


def js(p, default=None):
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else default


def f4(x, s="+.4f"):
    return "-" if x is None else format(x, s)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    return plt


# ------------------------------------------------------------------ Part A: equal-compute frontier (no training)
# head family -> (legend label, colour); width -> marker. Seed-0 runs of rounds 8-10, D arms excluded (user's scope).
FAMILY = {"flat": ("flat tied head (50,304)", GREY), "r8": ("round-8 class heads (CB, H2, H2r)", YEL),
          "r9": ("round-9 two-level, K=4096 C=1024", BLUE), "r10": ("round-10 two-level, K=8192 / C=4096 family", AQUA),
          "small": ("flat head, small BPE vocab (round 11)", RED), "rs": ("two-level + exact re-scoring (round 11)", RED)}
WMARK = {128: "o", 256: "^", 384: "s", 512: "D"}
UNLABELLED = {"H2", "H2a", "H2tm", "H2atm", "H3k8", "H3c4"}


def family(cfg, rnd):
    h = cfg.get("head") or ""
    if not h:
        return "small" if cfg.get("vocab", 50304) < 50000 else "flat"
    if cfg.get("head_rescore"):
        return "rs"
    if h in ("cb", "hier"):
        return "r8"
    if h == "hier9" and cfg.get("head_adapt") in (0, 4096) and cfg.get("head_C") == 1024:
        return "r9"
    return "r10"


def run_points(rnds=(8, 9, 10), include=None):
    """[{name, round, arm, d, family, fpt, branches: [(stable, tokens, loss, wall_s)]}] for every DONE seed-0 run."""
    out = []
    for r in rnds:
        for done in sorted(glob.glob(os.path.join(RD[r], "*_s0.DONE"))):
            name = os.path.basename(done)[:-5]
            if r == 10 and name.startswith("wsd_D"):
                continue  # round-10 n-gram arms: dropped by the user's scope decision
            if include is not None and name not in include:
                continue
            run = load_run(done[:-5] + ".jsonl")
            if not run:
                continue
            cfg = run["config"]["cfg"]
            fpt = run["final"]["flops_per_token"]["total"]
            br = [(k, v["tokens"], v["loss"], v["wall"]) for k, v in sorted(run["branch"].items())]
            out.append({"name": name, "round": r, "arm": run["config"]["arm"], "d": cfg["d"], "family": family(cfg, r),
                        "fpt": fpt, "branches": br, "data": cfg.get("data", ""), "run": run})
    return out


def lower_envelope(pts):
    """pts [(x, y, label)] -> the points not dominated (no other point with x' <= x and y' < y), sorted by x."""
    env, best = [], float("inf")
    for x, y, lab in sorted(pts):
        if y < best - 1e-12:
            env.append((x, y, lab))
            best = y
    return env


def short(name):
    s = name.replace("wsd_", "").replace("_s0", "")
    return {"B_r8": "B", "H2atm_4800M": "H2atm 4.8B", "Rwide1200": "R-wide", "Rwide": "R-wide (600M run)", "Rdeep": "R-deep",
            "Rloop": "R-loop", "H3s384_575M": "H3*-384", "B384": "B384", "H2atm384": "H2atm-384"}.get(s, s)


def part_a(runs, L, res):
    rows = []
    for r in runs:
        for k, tok, loss, wall in r["branches"]:
            rows.append({"run": r["name"], "round": r["round"], "d": r["d"], "family": r["family"], "fpt": r["fpt"],
                         "stable_tokens": k, "tokens": tok, "flops_stable": r["fpt"] * k, "flops": r["fpt"] * tok,
                         "loss": loss, "bpb": loss * BPN, "wall_s": wall})
    res["partA"] = {"rows": rows}
    chk = [("B (B_r8) 1200M", "wsd_B_r8_s0", 1200e6, 6.0e16), ("R-wide 1200M", "wsd_Rwide1200_s0", 1200e6, 6.8e16),
           ("B384 300M", "wsd_B384_s0", 300e6, 5.8e16), ("H3*-384 575M", "wsd_H3s384_575M_s0", 575e6, 6.1e16)]
    L += ["## Part A — equal-compute frontier (no training)", "",
          "Training FLOPs = the run's logged `flops_per_token.total` × tokens. The user's table counts stable tokens only; "
          "the frontier uses stable + decay tokens (every reported branch includes its 10% decay).", "",
          "| run | FLOPs/token | user's table | ours, stable tokens | ours, stable + decay | matches (within 1%) |", "|---|---|---|---|---|---|"]
    res["partA"]["check"] = []
    for lab, nm, k, user in chk:
        row = next((x for x in rows if x["run"] == nm and abs(x["stable_tokens"] - k) <= 0.01 * k), None)
        if row is None:
            L.append(f"| {lab} | - | {user:.1e} | missing | missing | - |")
            continue
        ok = abs(row["flops_stable"] / user - 1) <= 0.01  # the table's values are 2 s.f. (6.05e16 written 6.0e16)
        res["partA"]["check"].append({"run": lab, "user": user, "stable": row["flops_stable"], "total": row["flops"], "match": ok})
        L.append(f"| {lab} | {row['fpt'] / 1e6:.2f}M | {user:.1e} | {row['flops_stable']:.3e} | {row['flops']:.3e} | {'yes' if ok else 'NO'} |")
    L += ["", f"Every branch (loss = full-val nats per GPT-2 token; BPB = nats × {BPN:.5f}):", "",
          "| run | d | head family | stable tokens | tokens | FLOPs | wall min | loss | BPB |", "|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: x["flops"]):
        L.append(f"| {short(x['run'])} | {x['d']} | {x['family']} | {x['stable_tokens'] / 1e6:.0f}M | {x['tokens'] / 1e6:.0f}M | "
                 f"{x['flops']:.2e} | {x['wall_s'] / 60:.1f} | {x['loss']:.4f} | {x['bpb']:.4f} |")
    for key, fname, xl in (("flops", "frontier_loss_vs_flops.png", "total training FLOPs (stable + decay tokens, log scale)"),
                           ("wall_s", "frontier_loss_vs_wall.png", "own-schedule wall-clock, minutes (log scale, RTX 5070 Ti)")):
        env = lower_envelope([(x[key], x["loss"], short(x["run"]) + f" {x['stable_tokens'] / 1e6:.0f}M") for x in rows])
        res["partA"][f"envelope_{key}"] = [{"x": a, "loss": b, "point": c} for a, b, c in env]
        frontier_plot(runs, key, os.path.join(OUT, fname), xl, env)
        L += ["", f"Lower envelope vs {'FLOPs' if key == 'flops' else 'wall-clock'}: " +
              ", ".join(f"{c} ({b:.4f})" for a, b, c in env) + f". Figure: `runs/round11/{fname}`."]
    L.append("")


def frontier_plot(runs, key, path, xlabel, env, ax=None, title=None):
    plt = _plt()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 6.6))
    seen_f = set()
    for r in runs:
        lab, col = FAMILY[r["family"]]
        xs = [(r["fpt"] * tok if key == "flops" else wall / 60) for k, tok, loss, wall in r["branches"]]
        ys = [loss for k, tok, loss, wall in r["branches"]]
        ax.plot(xs, ys, "-", color=col, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(xs, ys, WMARK.get(r["d"], "o"), color=col, ms=7, mec="white", mew=1.2, zorder=3,
                label=lab if r["family"] not in seen_f else None)
        seen_f.add(r["family"])
        if short(r["name"]) not in UNLABELLED:
            ax.annotate(short(r["name"]), (xs[-1], ys[-1]), xytext=(6, -3), textcoords="offset points", fontsize=8, color="#333")
    ex = [(a if key == "flops" else a / 60) for a, b, c in env]
    ax.step(ex, [b for a, b, c in env], where="post", color="#222", lw=1.0, ls="--", zorder=1, label="lower envelope")
    for w, mk in WMARK.items():
        if any(r["d"] == w for r in runs):
            ax.plot([], [], mk, color="#555", ms=7, ls="none", label=f"width d={w}")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("val loss, nats per GPT-2 token")
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * BPN, lambda v: v / BPN))
    sec.set_ylabel("bits per byte")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(title or f"Loss vs {'training FLOPs' if key == 'flops' else 'wall-clock'}: every round 8–10 seed-0 run (branches joined)",
                 loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    if own:
        fig.text(0.01, 0.01, "Colour = head family, marker = width: runs of different widths are compared. "
                 "Round-10 n-gram (D) arms excluded by the user's scope decision.\nUnlabelled d=128 lines: "
                 + ", ".join(sorted(UNLABELLED)) + ".", fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(path, dpi=130)
        plt.close(fig)



# ------------------------------------------------------------------ shared inputs
GPT2_BPT = 4_999_168 / VAL_BYTES          # GPT-2 val tokens per byte (4,882 rows x 1,024 targets)
H3_575 = ("round10", "wsd_H3s384_575M_s0")
BUCKETS = ["[0,256)", "[256,1024)", "[1024,4096)", "[4096,16384)", "[16384,V)"]
BLIMP_GROUP = {"s-selection": "argument_structure"}   # see E2 note: HF labels 2 argument-structure paradigms s-selection


def R(rnd, name):
    run = load_run(os.path.join(RD[rnd], name + ".jsonl"))
    if run:
        run["name"] = name
    return run


def bkey(run, tok, tol=0.01):
    if not run:
        return None
    ks = [k for k in run["branch"] if abs(k - tok) <= tol * tok]
    return ks[0] if ks else None


def br(run, tok):
    k = bkey(run, tok)
    return None if k is None else run["branch"][k]


def flops(run, tok):
    b = br(run, tok)
    return None if b is None else run["final"]["flops_per_token"]["total"] * b["tokens"]


def last(rows, **kw):
    """last jsonl record matching every key=value."""
    m = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return m[-1] if m else None


def seed_flag(delta, thresholds, band):
    """True if delta is within +-band of any verdict threshold (the pinned seed-1 rule)."""
    return delta is not None and any(abs(delta - t) <= band + 1e-12 for t in thresholds)


def decode_ms():
    rows = jl(os.path.join(OUT, "decode.jsonl"))
    out = {}
    for a in {r["arm"] for r in rows}:
        out[a] = float(np.mean([r["ms_per_token"] for r in rows if r["arm"] == a]))
    return out


# ------------------------------------------------------------------ Part B
def part_b(L, res, EB):
    H = R(10, "wsd_H3s384_575M_s0")
    B384 = R(9, "wsd_B384_s0")
    DEC = decode_ms()
    bpe = {V: js(os.path.join(OUT, f"bpe{V}.json"), {}) for V in (8192, 4096)}
    L += ["## Part B — small-vocabulary control (flat head, BPE 8k / 4k, d=384)", "",
          "Tokenizers: byte-level BPE trained on the first 2.0 GB of the training stream (GPT-2 pre-tokenisation, EOT kept as id V−1); "
          "round trip exact on all 4,895 val docs (`runs/round11/bpe8192.json`, `bpe4096.json`). BPB on the same 23,160,213 FineWeb-Edu "
          "val bytes: GPT-2 arms = logged val loss × tokens / (bytes × ln 2); BPE arms = the byte-aligned evaluator (`round11_eval.py`; "
          "the docstring's `round11_bytes.py` is implemented there).", ""]
    for V in (8192, 4096):
        if bpe[V]:
            L.append(f"- BPE-{V // 1024}k: {bpe[V]['val_bytes_per_token_docs']:.3f} val bytes/token (GPT-2: {1 / GPT2_BPT:.3f}), "
                     f"round-trip failures {bpe[V]['gate_roundtrip_failures']}, sha256 {bpe[V]['sha256'][:12]}…")
    V = jl(os.path.join(OUT, "eval_validate.json")) if False else js(os.path.join(OUT, "eval_validate.json"), {})
    if V:
        L.append("- Evaluator check in GPT-2 mode (re-encode with tiktoken through the byte-aligned path vs direct scoring vs the logged "
                 "val loss): " + "; ".join(f"{k}: direct−logged {v['direct_minus_logged']:+.1e} nats, aligned−direct "
                                            f"{v['aligned_bpb'] - v['direct_bpb']:+.1e} BPB, gate {'OK' if v['gate_ok'] else 'FAIL'}" for k, v in V.items()))
    L += ["", "| arm | branch | tokens | FLOPs | FLOPs vs H3*-384 | BPB | wall min | params | bytes/token | decode ms/token | decode ms/byte |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}

    def add(lab, run, tok, bpb, bpt, dec, match=None):
        b = br(run, tok)
        if b is None:
            return
        f = flops(run, tok)
        rows[lab] = {"tokens": b["tokens"], "flops": f, "bpb": bpb, "wall_min": b["wall"] / 60, "params": run["final"]["params"]["total"],
                     "bytes_per_token": bpt, "decode_ms_token": dec, "decode_ms_byte": None if dec is None else dec / bpt,
                     "flops_vs_match": None if match is None else f / match}
        L.append(f"| {lab} | {tok / 1e6:.0f}M stable | {b['tokens'] / 1e6:.0f}M | {f:.3e} | {f4(rows[lab]['flops_vs_match'], '.4f')} | "
                 f"{f4(bpb, '.4f')} | {b['wall'] / 60:.1f} | {run['final']['params']['total'] / 1e6:.2f}M | {bpt:.3f} | {f4(dec, '.3f')} | "
                 f"{f4(rows[lab]['decode_ms_byte'], '.4f')} |")
    hm = {300: flops(H, 300e6), 575: flops(H, 575e6)}
    for t in (300, 575):
        add(f"H3*-384 ({t}M)", H, t * 1e6, br(H, t * 1e6)["loss"] * BPN if br(H, t * 1e6) else None, 1 / GPT2_BPT, DEC.get("H3s384"), hm[t])
    add("B384 (300M)", B384, 300e6, br(B384, 300e6)["loss"] * BPN, 1 / GPT2_BPT, DEC.get("B384"), hm[575])
    for arm, nm, (b1, b2), V in (("F8k-384", "wsd_F8k384_631M_s0", (329383936, 631308288), 8192),
                                 ("F4k-384", "wsd_F4k384_700M_s0", (365035520, 699531264), 4096)):
        run = R(11, nm)
        if not run:
            L.append(f"| {arm} | not run yet | | | | | | | | | |")
            continue
        for tok, match, lab in ((b1, hm[300], f"{arm[:3]}384_{round(b1 / 1e6)}M"), (b2, hm[575], f"{arm[:3]}384_{round(b2 / 1e6)}M")):
            e = last(EB, label=lab, corpus="fineweb")
            add(f"{arm} ({tok / 1e6:.0f}M)", run, tok, None if e is None else e["bpb"], bpe[V].get("val_bytes_per_token_docs", float("nan")),
                DEC.get(arm.replace("-", "")), match)
            if e and e["unscored_bytes"]:
                L.append(f"| | {e['unscored_bytes']} val bytes unscored (rows cut inside a UTF-8 character) | | | | | | | | | |")
    res["partB"] = {"rows": rows}
    h, f = rows.get("H3*-384 (575M)"), rows.get("F8k-384 (631M)")
    if h and f and f["bpb"] is not None:
        d = h["bpb"] - f["bpb"]
        v = "two-level wins" if d <= -0.003 else "small vocab wins" if d >= 0.003 else "tie"
        if v == "tie" and f["decode_ms_byte"] and h["decode_ms_byte"]:
            v += f" (decode cost per byte: H3*-384 {h['decode_ms_byte']:.4f} ms vs F8k-384 {f['decode_ms_byte']:.4f} ms → " + \
                 ("two-level cheaper)" if h["decode_ms_byte"] < f["decode_ms_byte"] else "small vocab cheaper)")
        h3, f3 = rows.get("H3*-384 (300M)"), rows.get("F8k-384 (329M)")
        d300 = None if not (h3 and f3 and f3["bpb"]) else h3["bpb"] - f3["bpb"]
        res["partB"].update({"delta_bpb_575": d, "delta_bpb_300": d300, "verdict": v, "seed1_needed": seed_flag(d, (-0.003, 0.003), 0.003)})
        L += ["", f"**Verdict (pinned rule, matched FLOPs, H3*-384 575M vs F8k-384 631M): BPB(H3*) − BPB(F8k) = {d:+.4f} → {v}.** "
              f"At the 300M-matched branch: {f4(d300)}." + (" Within ±0.003 of a threshold: the seed-1 rule applies." if res["partB"]["seed1_needed"] else "")]
    L.append("")


# ------------------------------------------------------------------ Part C
def part_c(L, res, SM):
    B = R(8, "wsd_B_r8_s0")
    H = R(10, "wsd_H3k8c4_s0")
    rs = {M: R(11, f"wsd_H3rs{M}_s0") for M in (16, 64)}
    BK = js(os.path.join(OUT, "buckets_partC.json"))
    BK64 = js(os.path.join(OUT, "buckets_partC64.json"))
    if BK and BK64:
        BK["arms"].update(BK64["arms"])
    rule = js(os.path.join(OUT, "partC_rule.json"))
    DEC = decode_ms()
    toks = [75e6, 150e6, 300e6, 600e6, 1200e6]
    L += ["## Part C — exact re-scoring of the top-M tail clusters (tH3rs, d=128)", "",
          "Gates 1–4 (`runs/round11/gates_partC.txt`): " + ("ALL OK" if (js(os.path.join(OUT, "gates_partC.json"), {}).get("fails") == []) else "see file") + ".", "",
          "| arm | " + " | ".join(f"{t / 1e6:.0f}M" for t in toks) + " | Δ vs B @1200M | Δ vs H3k8c4 @1200M |", "|---" * 8 + "|"]
    out = {}
    for lab, run in (("B (B_r8)", B), ("H3k8c4", H), ("tH3rs16", rs[16]), ("tH3rs64", rs[64])):
        if not run:
            if lab.startswith("tH3rs"):
                L.append(f"| {lab} | " + " | ".join("-" for _ in toks) + " | not run | |")
            continue
        ls = [br(run, t)["loss"] if br(run, t) else None for t in toks]
        dB = None if ls[-1] is None else ls[-1] - br(B, 1200e6)["loss"]
        dH = None if ls[-1] is None else ls[-1] - br(H, 1200e6)["loss"]
        out[lab] = {"loss": dict(zip([int(t / 1e6) for t in toks], ls)), "d_vs_B_1200": dB, "d_vs_H3k8c4_1200": dH}
        L.append(f"| {lab} | " + " | ".join(f4(x, ".4f") for x in ls) + f" | {f4(dB)} | {f4(dH)} |")
    L.append("")
    # cost table
    L += ["| arm | head fwd FLOPs/token | head FLOPs vs B | train FLOPs vs B | smoke tok/s | own wall to 1200M | decode ms/token |", "|---|---|---|---|---|---|---|"]
    for lab, arm, run, dk in (("B (B_r8)", "tB", B, "B"), ("H3k8c4", "tH3k8c4", H, "H3s"), ("tH3rs16", "tH3rs16", rs[16], "H3rs16"), ("tH3rs64", "tH3rs64", rs[64], "H3rs64")):
        s = SM.get(arm, {})
        if not s and not run:
            continue
        hf = s.get("head_fwd_flops_per_token")
        L.append(f"| {lab} | {f4(hf, ',.0f')} | {f4(s.get('head_flops_vs_B'), '.3f')} | {f4(s.get('train_flops_vs_B'), '.3f')} | "
                 f"{f4(s.get('tok_s'), ',.0f')} | {f4(br(run, 1200e6)['wall'] / 60 if run and br(run, 1200e6) else None, '.1f')} min | {f4(DEC.get(dk), '.3f')} |")
        if lab in out:
            out[lab]["head_flops_vs_B"] = s.get("head_flops_vs_B")
    if BK:
        L += ["", f"Per-bucket Δ vs B_r8 at 1200M, nats (top = top-level part, within = within-cluster part; `runs/round11/buckets_partC.json`, "
                  f"{BK['n_targets']:,} val targets):", "", "| arm | quantity | " + " | ".join(BK["buckets"]) + " | all |", "|---" * 8 + "|"]
        for lab, a in BK["arms"].items():
            for q in ("d_total", "d_top", "d_within"):
                L.append(f"| {lab} | Δ{q[2:]} | " + " | ".join(f"{x:+.4f}" for x in a[q + "_bucket"]) + f" | {a[q]:+.4f} |")
            if a.get("hit_fraction") is not None:
                L.append(f"| {lab} | hit fraction (tail targets whose cluster is re-scored) | " + " | ".join(f4(x, '.3f') for x in a["hit_fraction_bucket"])
                         + f" | {a['hit_fraction']:.3f} |")
                out.setdefault("hits", {})[lab] = {"hit_fraction": a["hit_fraction"], "tail_share": a["tail_share"],
                                                  "nll_tail_hit": a["nll_tail_hit"], "nll_tail_miss": a["nll_tail_miss"]}
    # verdict on the best M
    cand = [(out[l]["loss"][1200], l) for l in ("tH3rs16", "tH3rs64") if l in out and out[l]["loss"][1200] is not None]
    if cand:
        best_loss, best = min(cand)
        dB, dH = out[best]["d_vs_B_1200"], out[best]["d_vs_H3k8c4_1200"]
        hfr = out[best].get("head_flops_vs_B")
        if dB <= 0.010 and hfr is not None and hfr <= 0.30:
            v = "gap closed"
        elif dH < -0.010:
            v = "partial"
        else:
            v = "no help"
        out.update({"best": best, "verdict": v, "seed1_needed": seed_flag(dB, (0.010,), 0.010) or seed_flag(dH, (-0.010,), 0.010),
                    "rule_M64": rule})
        L += ["", f"**Verdict (best M = {best}): Δ vs B @1200M {dB:+.4f}, Δ vs H3k8c4 {dH:+.4f}, head FLOPs {f4(hfr, '.3f')}× B → {v}.**"
              + (" Within ±0.010 of a threshold: the seed-1 rule applies." if out["seed1_needed"] else "")
              + (f" M=64 rule: tH3rs16 − H3k8c4 = {rule['delta']:+.4f} → M=64 {'run' if rule['run_M64'] else 'not run'}." if rule else "")]
        # seed 1 (pinned rule: both sides; the seed spread is the floor). H3k8c4 seed 1 = round 10's run; B seed 1 = round 8's if present.
        s1 = {"tH3rs16": R(11, "wsd_H3rs16_s1"), "H3k8c4": R(10, "wsd_H3k8c4_s1"), "B": R(8, "wsd_B_r8_s1") or R(11, "wsd_B_s1")}
        if out["seed1_needed"] and s1["tH3rs16"] and s1["H3k8c4"]:
            l1 = {k: (br(v, 1200e6)["loss"] if v and br(v, 1200e6) else None) for k, v in s1.items()}
            dH1 = l1["tH3rs16"] - l1["H3k8c4"]
            dHm = (dH + dH1) / 2
            spread = abs(dH - dH1)
            dB1 = None if l1["B"] is None else l1["tH3rs16"] - l1["B"]
            v2 = "partial" if dHm < -0.010 and abs(dHm) > spread else "no help"
            out["seed1"] = {"loss_1200": l1, "d_vs_H3k8c4_s1": dH1, "d_vs_H3k8c4_mean": dHm, "seed_spread": spread,
                            "d_vs_B_s1": dB1, "verdict_2seed": v2}
            L += ["", "Seed 1 (pinned seed rule; H3k8c4 seed 1 is round 10's `wsd_H3k8c4_s1`):", "",
                  "| arm @1200M | seed 0 | seed 1 |", "|---|---|---|",
                  f"| tH3rs16 | {out['tH3rs16']['loss'][1200]:.4f} | {l1['tH3rs16']:.4f} |",
                  f"| H3k8c4 | {br(H, 1200e6)['loss']:.4f} | {l1['H3k8c4']:.4f} |",
                  f"| B (B_r8) | {br(B, 1200e6)['loss']:.4f} | {f4(l1['B'], '.4f')} |",
                  f"| Δ tH3rs16 − H3k8c4 | {dH:+.4f} | {dH1:+.4f} |", "",
                  f"**Two-seed verdict: mean Δ vs H3k8c4 {dHm:+.4f} (seed spread {spread:.4f}) → {v2}.**"]
    # d=384
    H384 = R(10, "wsd_H3s384_575M_s0")
    for M in (16, 64):
        r384 = R(11, f"wsd_H3rs{M}_384_575M_s0")
        if r384:
            row = {t: br(r384, t * 1e6)["loss"] if br(r384, t * 1e6) else None for t in (300, 575)}
            out[f"H3rs{M}_384"] = {"loss": row, "d_vs_H3s384_575": row[575] - br(H384, 575e6)["loss"] if row[575] else None,
                                  "d_vs_B384_300": row[300] - br(R(9, "wsd_B384_s0"), 300e6)["loss"] if row[300] else None}
            L += ["", f"d=384: H3rs{M}-384 300M {f4(row[300], '.4f')} (Δ vs B384 300M {f4(out[f'H3rs{M}_384']['d_vs_B384_300'])}), "
                      f"575M {f4(row[575], '.4f')} (Δ vs H3*-384 575M {f4(out[f'H3rs{M}_384']['d_vs_H3s384_575'])})."]
    res["partC"] = out
    L.append("")


# ------------------------------------------------------------------ Part D
def part_d(L, res):
    for nm in ("wsd_H3s512_359M_s0", "wsd_H3rs16_512_359M_s0", "wsd_H3rs64_512_359M_s0"):
        r = R(11, nm)
        if r:
            H = R(10, "wsd_H3s384_575M_s0")
            b = br(r, 359.4e6)
            d = b["loss"] - br(H, 575e6)["loss"]
            res["partD"] = {"run": nm, "loss": b["loss"], "flops": flops(r, 359.4e6), "flops_H3s384_575": flops(H, 575e6), "d_vs_H3s384_575": d,
                            "seed1_needed": abs(d) <= 0.010}
            L += ["## Part D — H3*-512 at FLOPs matched to H3*-384's 575M run", "",
                  f"{nm}: loss {b['loss']:.4f} at {b['tokens'] / 1e6:.0f}M tokens, {res['partD']['flops']:.3e} FLOPs (H3*-384 575M: "
                  f"{res['partD']['flops_H3s384_575']:.3e}); Δ vs H3*-384 575M {d:+.4f} nats; wall {b['wall'] / 60:.1f} min.", ""]
            return
    L += ["## Part D — not run", "", "Optional; dropped for budget (the brief's first cut).", ""]
    res["partD"] = None


# ------------------------------------------------------------------ Part E
E_D384 = ["B384", "H3s384_300M", "H3s384_575M", "F8k384_329M", "F8k384_631M", "H3rs16_384_575M", "H3rs64_384_575M"]
E_D128 = ["B_r8", "H3k8c4", "H3rs16", "H3rs64"]
PAIRS = [("equal tokens (300M)", "H3s384_300M", "B384"), ("equal FLOPs (6.72e16 vs 6.39e16)", "H3s384_575M", "B384"),
         ("d=128, equal tokens (1200M)", "H3k8c4", "B_r8"), ("d=128 re-scored, equal tokens", "H3rs16", "B_r8")]


def e1(L, res, EB):
    L += ["### E1 — BPB in and out of domain", "",
          "Rows of 1,025 GPT-2 tokens (first token = context), EOT between documents; BPE models byte-aligned on the same bytes. "
          "Corpora (`runs/round11/corpora.json`): FineWeb-Edu val ({fineweb[rows]:,} rows, {fineweb[scored_bytes]:,} bytes), WikiText-103 test (raw, "
          "articles split at ' = Title = ' headings; {wikitext103[rows]:,} rows, {wikitext103[scored_bytes]:,} bytes), PG-19 test (emozilla/pg19-test "
          "mirror, books in order until ≥ 1M GPT-2 tokens; {pg19[rows]:,} rows, {pg19[scored_bytes]:,} bytes).".format(**js(os.path.join(OUT, "corpora.json"))), "",
          "| model | FineWeb-Edu | WikiText-103 | PG-19 |", "|---|---|---|---|"]
    tab = {}
    for lab in E_D384 + E_D128:
        rs = {c: last(EB, label=lab, corpus=c) for c in ("fineweb", "wikitext103", "pg19")}
        if not any(rs.values()):
            continue
        tab[lab] = {c: (r["bpb"] if r else None) for c, r in rs.items()}
        L.append(f"| {lab} | " + " | ".join(f4(tab[lab][c], ".4f") for c in ("fineweb", "wikitext103", "pg19")) + " |")
    L += ["", "| comparison | FineWeb-Edu Δ | WikiText-103 Δ | PG-19 Δ |", "|---|---|---|---|"]
    cmp_ = {}
    for nm, a, b in PAIRS + [("F8k-384 vs H3*-384, equal FLOPs (575M-matched)", "F8k384_631M", "H3s384_575M"),
                             ("F8k-384 vs H3*-384, equal FLOPs (300M-matched)", "F8k384_329M", "H3s384_300M")]:
        if a in tab and b in tab:
            cmp_[nm] = {c: (None if tab[a][c] is None or tab[b][c] is None else tab[a][c] - tab[b][c]) for c in tab[a]}
            L.append(f"| {a} − {b}, {nm} | " + " | ".join(f4(cmp_[nm][c]) for c in ("fineweb", "wikitext103", "pg19")) + " |")
    v = None
    if "H3s384_575M" in tab and "B384" in tab and all(tab[x][c] is not None for x in ("H3s384_575M", "B384") for c in ("wikitext103", "pg19")):
        ds = [tab["H3s384_575M"][c] - tab["B384"][c] for c in ("wikitext103", "pg19")]
        v = "holds" if all(d <= -0.003 for d in ds) else "does not hold"
        L += ["", f"**OOD verdict (H3*-384 575M ≤ B384 − 0.003 BPB on both OOD sets): WikiText {ds[0]:+.4f}, PG-19 {ds[1]:+.4f} → {v}.**"
              + (" Within ±0.003 of the threshold on a set: seed-1 rule applies." if any(seed_flag(d, (-0.003,), 0.003) for d in ds) else "")]
        res["E1_seed1_needed"] = any(seed_flag(d, (-0.003,), 0.003) for d in ds)
    # per-bucket Δ (GPT-2 models)
    L += ["", "Per-bucket Δ NLL (nats per target, two-level − flat) by training rank of the target:", "",
          "| comparison | corpus | " + " | ".join(BUCKETS) + " |", "|---" * 7 + "|"]
    bk = {}
    for nm, a, b in PAIRS:
        for c in ("fineweb", "wikitext103", "pg19"):
            ra, rb = last(EB, label=a, corpus=c), last(EB, label=b, corpus=c)
            if ra and rb and "bucket_nll_sum" in ra and "bucket_nll_sum" in rb:
                d = [(sa / na - sb / nb) if na and nb else None for sa, na, sb, nb in
                     zip(ra["bucket_nll_sum"], ra["bucket_n"], rb["bucket_nll_sum"], rb["bucket_n"])]
                bk[f"{a}-{b}:{c}"] = d
                L.append(f"| {a} − {b} | {c} | " + " | ".join(f4(x) for x in d) + " |")
    res["E1"] = {"bpb": tab, "deltas": cmp_, "bucket_deltas": bk, "verdict": v}
    L.append("")


def e2(L, res):
    rows = jl(os.path.join(OUT, "eval_blimp.jsonl"))
    if not rows:
        L += ["### E2 — BLiMP: not run yet", ""]
        return
    grp = {}
    tab = {}
    for lab in E_D384 + E_D128:
        r = last(rows, label=lab)
        if not r:
            continue
        g = {}
        for s, v in r["subsets"].items():
            t = BLIMP_GROUP.get(v["term"], v["term"])
            g.setdefault(t, [0, 0])
            g[t][0] += v["correct"]
            g[t][1] += v["n"]
        tab[lab] = {"overall": 100 * r["acc"], **{t: 100 * c / n for t, (c, n) in g.items()}}
        grp = sorted(g)
    L += ["### E2 — BLiMP (67 subsets, 12 phenomena)", "",
          "A pair is correct if sum log p(good) > sum log p(bad), each sentence after one EOT. Groups = the dataset's `linguistics_term`, with "
          "its 2 `s-selection` subsets (animate_subject_passive / _trans) folded into argument_structure as in the BLiMP paper (the field has 13 values; "
          "decided before any E2 result).", "",
          "| model | overall | " + " | ".join(grp) + " |", "|---" * (len(grp) + 2) + "|"]
    for lab, t in tab.items():
        L.append(f"| {lab} | {t['overall']:.2f} | " + " | ".join(f"{t[g]:.1f}" for g in grp) + " |")
    L += ["", "| comparison | overall Δ (points) | real (> 1 point)? |", "|---|---|---|"]
    out = {}
    for nm, a, b in PAIRS + [("F8k vs H3*, equal FLOPs", "F8k384_631M", "H3s384_575M")]:
        if a in tab and b in tab:
            d = tab[a]["overall"] - tab[b]["overall"]
            out[nm] = {"delta": d, "real": abs(d) > 1.0, "seed1_needed": seed_flag(abs(d), (1.0,), 1.0)}
            L.append(f"| {a} − {b}, {nm} | {d:+.2f} | {'yes' if abs(d) > 1 else 'no'} |")
    # seed 1 (pinned rule: a Δ within ±1 point of the 1-point threshold runs seed 1 for both sides; the seed spread is the floor)
    acc1 = {r["label"]: 100 * r["acc"] for r in rows if r["label"].endswith("_s1")}
    s1rows = []
    for nm, a, b in PAIRS:
        if nm in out and out[nm]["seed1_needed"] and a + "_s1" in acc1 and b + "_s1" in acc1:
            d0, d1 = out[nm]["delta"], acc1[a + "_s1"] - acc1[b + "_s1"]
            spread = max(abs(tab[a]["overall"] - acc1[a + "_s1"]), abs(tab[b]["overall"] - acc1[b + "_s1"]))
            mean = (d0 + d1) / 2
            real = abs(mean) > 1.0 and abs(mean) > spread
            out[nm].update({"delta_s1": d1, "delta_mean": mean, "seed_spread_points": spread, "real_2seed": real})
            s1rows.append(f"| {a} − {b}, {nm} | {d0:+.2f} | {d1:+.2f} | {mean:+.2f} | {spread:.2f} | {'yes' if real else 'no'} |")
    if s1rows:
        L += ["", "Seed 1 (BLiMP overall, points; seed spread = the larger same-model change between seeds):", "",
              "| comparison | Δ seed 0 | Δ seed 1 | mean | seed spread | real? |", "|---|---|---|---|---|---|"] + s1rows
    res["E2"] = {"acc": tab, "acc_s1": acc1, "deltas": out}
    L.append("")


def e3(L, res):
    rows = jl(os.path.join(OUT, "eval_lambada.jsonl"))
    if not rows:
        L += ["### E3 — LAMBADA: not run yet", ""]
        return
    L += ["### E3 — LAMBADA (5,153 test passages)", "",
          "Accuracy = every target token is the argmax of the full next-token distribution; word NLL in bits per UTF-8 byte of \" \" + last word; "
          "split by the training rank of the target's first GPT-2 token.", "",
          "| model | accuracy % | word bits/byte | acc, rank < 4096 | acc, rank ≥ 4096 | bits/byte, < 4096 | bits/byte, ≥ 4096 |", "|---|---|---|---|---|---|---|"]
    tab = {}
    for lab in E_D384 + E_D128:
        r = last(rows, label=lab)
        if r:
            tab[lab] = r
            L.append(f"| {lab} | {100 * r['acc']:.2f} | {r['word_bits_per_byte']:.4f} | {100 * r['acc_top4096']:.2f} (n={r['n_top4096']}) | "
                     f"{100 * r['acc_rest']:.2f} (n={r['n'] - r['n_top4096']}) | {r['bpb_top4096']:.4f} | {r['bpb_rest']:.4f} |")
    out = {}
    L += ["", "| comparison | Δ accuracy (points) | Δ word bits/byte | Δ acc ≥ 4096 | Δ bits/byte ≥ 4096 |", "|---|---|---|---|---|"]
    for nm, a, b in PAIRS + [("F8k vs H3*, equal FLOPs", "F8k384_631M", "H3s384_575M")]:
        if a in tab and b in tab:
            A, Bb = tab[a], tab[b]
            out[nm] = {"d_acc": 100 * (A["acc"] - Bb["acc"]), "d_bpb": A["word_bits_per_byte"] - Bb["word_bits_per_byte"],
                       "d_acc_rare": 100 * (A["acc_rest"] - Bb["acc_rest"]), "d_bpb_rare": A["bpb_rest"] - Bb["bpb_rest"]}
            L.append(f"| {a} − {b}, {nm} | {out[nm]['d_acc']:+.2f} | {out[nm]['d_bpb']:+.4f} | {out[nm]['d_acc_rare']:+.2f} | {out[nm]['d_bpb_rare']:+.4f} |")
    res["E3"] = {"rows": tab, "deltas": out}
    L.append("")


def e4(L, res):
    rows = jl(os.path.join(OUT, "eval_bios.jsonl"))
    if not rows:
        L += ["### E4 — synthetic bios: not run yet", ""]
        return
    tab = {lab: last(rows, label=lab) for lab in ("bioB_20000", "bioH_20000", "bioB_100000", "bioH_100000", "bioH_100000_eqF")}
    tab = {k: v for k, v in tab.items() if v}
    meta = js(os.path.join(OUT, "bios", "bios_meta.json"), {})
    L += ["### E4 — synthetic bios knowledge (d=128, B vs H3k8c4)", "",
          f"Population: 5 attributes per person, {meta.get('bits_per_person', 46.52):.2f} bits/person (log2 of 67,200 dates, 200 cities, 300 universities, "
          "100 majors, 250 employers). Knowledge bits = Σ max(0, log2|A| − NLL_bits(value | canonical probe)).", "",
          "| run | N | exposures | bits stored | of capacity | bits/param | bits/non-emb param | exact match | rank ≥ 4096 first-token share | exact, rare / common first token |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in tab.items():
        L.append(f"| {lab} | {r['N']:,} | {r['exposures']:.1f} | {r['bits']:,.0f} | {100 * r['bits'] / r['bits_capacity']:.1f}% | {r['bits_per_param']:.4f} | "
                 f"{r['bits_per_nonemb_param']:.4f} | {100 * r['exact']:.2f}% | {100 * r['rare_first_token_share']:.1f}% | "
                 f"{100 * r['exact_rare_vs_common'][0]:.1f}% / {100 * r['exact_rare_vs_common'][1]:.1f}% |")
    L += ["", "Per attribute (bits stored / exact match):", "", "| run | " + " | ".join(next(iter(tab.values()))["per_attr"]) + " |",
          "|---" * 6 + "|"]
    for lab, r in tab.items():
        L.append(f"| {lab} | " + " | ".join(f"{v['bits']:,.0f} / {100 * v['exact']:.1f}%" for v in r["per_attr"].values()) + " |")
    out = {"rows": {k: {x: v[x] for x in ("N", "exposures", "bits", "bits_capacity", "bits_per_param", "bits_per_nonemb_param", "exact", "rare_first_token_share")}
                    for k, v in tab.items()}}
    ratios = {N: tab[f"bioH_{N}"]["bits"] / tab[f"bioB_{N}"]["bits"] for N in (20000, 100000) if f"bioH_{N}" in tab and f"bioB_{N}" in tab and tab[f"bioB_{N}"]["bits"] > 0}
    if len(ratios) == 2:
        v1 = "no knowledge penalty" if all(x >= 0.95 for x in ratios.values()) else "knowledge penalty"
        out.update({"ratio_H_over_B": ratios, "verdict_penalty": v1, "seed1_needed": any(abs(x - 0.95) <= 0.05 for x in ratios.values())})
        L += ["", f"**H3k8c4 / B bits at equal exposures: N=20k {ratios[20000]:.3f}, N=100k {ratios[100000]:.3f} → {v1}** (rule: ≥ 0.95 at both)."]
    if "bioH_100000_eqF" in tab and "bioB_100000" in tab:
        r = tab["bioH_100000_eqF"]["bits"] / tab["bioB_100000"]["bits"]
        v2 = "more tokens → more knowledge" if r > 1 else "more tokens do not buy more knowledge"
        out.update({"eqF_ratio": r, "verdict_eqF": v2})
        L += [f"**Equal FLOPs at N=100k: H3k8c4 ({tab['bioH_100000_eqF']['exposures']:.0f} exposures) / B (100) bits = {r:.3f} → {v2}.**"]
    res["E4"] = out
    knowledge_plot(tab)
    L += ["", "Figure: `runs/round11/knowledge_bits.png`.", ""]


def knowledge_plot(tab):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    groups = [("N = 20,000\n100 exposures", "bioB_20000", "bioH_20000", None), ("N = 100,000\n100 exposures", "bioB_100000", "bioH_100000", None),
              ("N = 100,000\nequal FLOPs", "bioB_100000", None, "bioH_100000_eqF")]
    w = 0.36
    for i, (g, b, h, e) in enumerate(groups):
        items = [(b, GREY, "B (flat head)")] + ([(h, BLUE, "H3k8c4, equal exposures")] if h else [(e, AQUA, "H3k8c4, equal FLOPs (~237 exposures)")])
        for j, (lab, col, nm) in enumerate(items):
            if lab not in tab:
                continue
            r = tab[lab]
            x = i + (j - 0.5) * w
            ax.bar(x, r["bits"] / 1e6, w * 0.92, color=col, label=nm if i == (0 if col != AQUA else 2) else None)
            ax.text(x, r["bits"] / 1e6, f"{r['bits'] / 1e6:.2f}M\n{100 * r['bits'] / r['bits_capacity']:.0f}%", ha="center", va="bottom", fontsize=8.5, color="#333")
        cap = tab.get(b, {}).get("bits_capacity")
        if cap:
            ax.plot([i - 0.45, i + 0.45], [cap / 1e6] * 2, color="#222", lw=1, ls=":")
    ax.plot([], [], color="#222", lw=1, ls=":", label="capacity (all attributes known)")
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylabel("knowledge stored, million bits")
    ax.set_title("Synthetic bios: bits of knowledge stored (d=128, same body)", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "knowledge_bits.png"), dpi=130)
    plt.close(fig)


def e5(L, res):
    rows = jl(os.path.join(OUT, "eval_copy.jsonl"))
    if not rows:
        L += ["### E5 — copying: not run yet", ""]
        return
    tab = {lab: last(rows, label=lab) for lab in E_D384 + E_D128}
    tab = {k: v for k, v in tab.items() if v}

    def rare(rec, which):
        n = rec["bucket_n"][3:]
        v = rec[f"bucket_{which}"][3:]
        tot = sum(x for x in n)
        return sum((vv or 0) * nn for vv, nn in zip(v, n)) / tot if tot else None
    L += ["### E5 — in-context copying", "",
          "E5a: 2,000 FineWeb-Edu val windows of 512 tokens, repeated once; E5b: 2,000 lists of 512 tokens drawn uniformly from training ranks ≥ 4,096, "
          "repeated. NLL per token (nats), copy 1 / copy 2.", "",
          "| model | E5a copy 1 | E5a copy 2 | E5a copy 2, rank ≥ 4096 | E5b copy 1 | E5b copy 2 | E5b copy 2, 1st half | E5b copy 2, 2nd half |", "|---|---|---|---|---|---|---|---|"]
    for lab, r in tab.items():
        a, b = r["E5a"], r["E5b"]
        L.append(f"| {lab} | {a['copy1']:.4f} | {a['copy2']:.4f} | {rare(a, 'copy2'):.4f} | {b['copy1']:.4f} | {b['copy2']:.4f} | "
                 f"{b['copy2_first_half']:.4f} | {b['copy2_second_half']:.4f} |")
    L += ["", "| comparison | E5a Δ copy 1, rank ≥ 4096 | E5a Δ copy 2, rank ≥ 4096 | prior-only? | E5b Δ copy 1 | E5b Δ copy 2 | E5b Δ copy 2 1st / 2nd half | can't copy? |",
          "|---|---|---|---|---|---|---|---|"]
    out = {}
    for nm, a, b in PAIRS:
        if a in tab and b in tab:
            A, Bb = tab[a], tab[b]
            d1, d2 = rare(A["E5a"], "copy1") - rare(Bb["E5a"], "copy1"), rare(A["E5a"], "copy2") - rare(Bb["E5a"], "copy2")
            prior = d2 <= 0.02 and d2 <= d1 / 3
            e1_, e2_ = A["E5b"]["copy1"] - Bb["E5b"]["copy1"], A["E5b"]["copy2"] - Bb["E5b"]["copy2"]
            h1, h2 = A["E5b"]["copy2_first_half"] - Bb["E5b"]["copy2_first_half"], A["E5b"]["copy2_second_half"] - Bb["E5b"]["copy2_second_half"]
            cant = e2_ > 0.05
            out[nm] = {"pair": [a, b], "E5a_d_copy1_rare": d1, "E5a_d_copy2_rare": d2, "prior_only": prior, "E5b_d_copy1": e1_, "E5b_d_copy2": e2_,
                       "E5b_d_copy2_halves": [h1, h2], "cant_copy": cant}
            L.append(f"| {a} − {b}, {nm} | {d1:+.4f} | {d2:+.4f} | {'yes' if prior else 'no'} | {e1_:+.4f} | {e2_:+.4f} | {h1:+.4f} / {h2:+.4f} | {'yes' if cant else 'no'} |")
    res["E5"] = {"rows": {k: v for k, v in tab.items()}, "deltas": out}
    L += ["", "Rule: prior-only if the rank ≥ 4,096 Δ on copy 2 ≤ 0.02 nats and ≤ 1/3 of the copy-1 Δ (E5a); can't copy if the E5b copy-2 Δ > 0.05 nats. "
              "There is no equal-FLOPs d=128 pair (H3k8c4 was not run at B's FLOPs).", ""]


# ------------------------------------------------------------------ headline
def end_labels(fig, ax, ends, gap=11.0, fs=8.5):
    """Round 12: place end-of-line labels so they never overlap (spread apart vertically by >= gap points)."""
    if not ends:
        return
    fig.canvas.draw()  # fix the axis limits before converting to display units
    pt = fig.dpi / 72.0
    ys = [ax.transData.transform((x, y))[1] / pt for x, y, _ in ends]
    order = sorted(range(len(ends)), key=lambda i: ys[i])
    pos = {}
    last = None
    for i in order:  # bottom-up: push each label up until it clears the one below
        p = ys[i] if last is None else max(ys[i], last + gap)
        pos[i] = p
        last = p
    shift = (sum(ys) - sum(pos.values())) / len(ends)  # re-centre the stack on the points
    for i, (x, y, t) in enumerate(ends):
        ax.annotate(t, (x, y), xytext=(6, pos[i] + shift - ys[i] - 3), textcoords="offset points", fontsize=fs, color="#333")


def headline(res, EB):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
    ax = axs[0]
    H = R(10, "wsd_H3s384_575M_s0")
    B384 = R(9, "wsd_B384_s0")
    ser = [("B384 (flat, GPT-2 vocab)", GREY, "o", [(flops(B384, t), br(B384, t)["loss"] * BPN) for t in (150e6, 300e6)])]
    ser.append(("H3*-384 (two-level)", BLUE, "s", [(flops(H, t), br(H, t)["loss"] * BPN) for t in (300e6, 575e6)]))
    for arm, nm, toks, lab_, mk in (("F8k384", "wsd_F8k384_631M_s0", (329383936, 631308288), "F8k-384 (flat, BPE 8k)", "D"),
                                    ("F4k384", "wsd_F4k384_700M_s0", (365035520, 699531264), "F4k-384 (flat, BPE 4k)", "v")):
        run = R(11, nm)
        if run:
            pts = []
            for t in toks:
                e = last(EB, label=f"{arm}_{round(t / 1e6)}M", corpus="fineweb")
                if e:
                    pts.append((flops(run, t), e["bpb"]))
            if pts:
                ser.append((lab_, YEL, mk, pts))
    for M in (16, 64):
        r = R(11, f"wsd_H3rs{M}_384_575M_s0")
        if r:
            ser.append((f"H3rs{M}-384 (two-level + re-scoring)", AQUA, "^", [(flops(r, t), br(r, t)["loss"] * BPN) for t in (300e6, 575e6) if br(r, t)]))
    for nm in ("wsd_H3s512_359M_s0",):
        r = R(11, nm)
        if r:
            ser.append(("H3*-512 (two-level, d=512)", RED, "P", [(flops(r, t), br(r, t)["loss"] * BPN) for t in sorted(r["branch"])]))
    ends = []
    for lab, col, mk, pts in ser:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=col, lw=1.8)
        ax.plot(xs, ys, mk, color=col, ms=8, mec="white", mew=1.2, label=lab)
        ends.append((xs[-1], ys[-1], lab.split(" (")[0]))
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs (stable + decay)")
    ax.set_ylabel("bits per byte, FineWeb-Edu val (23.16M bytes)")
    ax.set_title("(a) BPB vs training FLOPs, d=384", loc="left", fontsize=12.5, fontweight="bold")
    ax.text(0.01, 0.02, "Different vocabularies (and d=512 if shown) compared at equal FLOPs on the same bytes", transform=ax.transAxes, fontsize=8.5, color="#555")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, lw=0.6)
    end_labels(fig, ax, ends)
    ax = axs[1]
    ends = []
    for nm, rnd, col, lab in (("wsd_B_r8_s0", 8, GREY, "B (flat)"), ("wsd_H3k8c4_s0", 10, BLUE, "H3k8c4 (two-level)"),
                              ("wsd_H3rs16_s0", 11, AQUA, "tH3rs16 (+ exact top-16)"), ("wsd_H3rs64_s0", 11, RED, "tH3rs64 (+ exact top-64)")):
        r = R(rnd, nm)
        if not r:
            continue
        ks = sorted(r["branch"])
        xs, ys = [r["branch"][k]["tokens"] for k in ks], [r["branch"][k]["loss"] for k in ks]
        ax.plot(xs, ys, "-o", color=col, lw=1.8, ms=6, mec="white", label=lab)
        ends.append((xs[-1], ys[-1], f"{ys[-1]:.4f}"))
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (stable + decay)")
    ax.set_ylabel("val loss, nats per GPT-2 token")
    ax.set_title("(b) d=128: loss vs tokens", loc="left", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    end_labels(fig, ax, ends)
    fig.suptitle("Round 11: is the two-level head a cheap AND good head?", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"round11_headline.{ext}"), dpi=130)
    plt.close(fig)


COMMANDS = [
    "python round11_bpe.py 8192 && python round11_bpe.py 4096          # tokenizers + round-trip gate (WSL, CPU)",
    "python round11_bios.py make                                        # bios population (WSL, CPU)",
    "JAX_PLATFORMS=cpu python round11_gates.py 16 64                    # Part C gates",
    "bash r11chain.sh one+d384+B:0+--label+B384+--tokens+300e6+--branch_at+150e6,300e6 one+d384+H3s384:0+--label+H3s384+--tokens+300e6+--branch_at+150e6,300e6",
    "bash r11chain.sh smoke+tB+tH3k8c4+tH3rs16+tH3rs64+B+H3s384+F8k384@631308288+F4k384@699531264+H3rs16_384+H3s512@359399424+bioB_20000@101056512+bioH_100000@1199505408 "
    "py+round11_smoke_check.py+--tag+all one+tiny+tH3rs16:0 pytry+round11_partc_next.py "
    "one+d384+F8k384:0+--label+F8k384+--tokens+631308288+--branch_at+329383936,631308288+--extra=--keep_branch=329383936 "
    "pytry+round11_eval.py+validate pytry+round11_eval.py+bpb pytry+round11_eval.py+lambada pytry+round11_eval.py+blimp pytry+round11_eval.py+copy "
    "pytry+round11_buckets.py+--tag+partC+--arms+H3k8c4:tH3k8c4:wsd_H3k8c4_s0,H3rs16:tH3rs16:wsd_H3rs16_s0 "
    "one+bios+bioB_20000:0+--tokens+101056512+--branch_at+101056512 one+bios+bioH_20000:0+--tokens+101056512+--branch_at+101056512 "
    "one+bios+bioB_100000:0+--tokens+505282560+--branch_at+505282560 one+bios+bioH_100000:0+--tokens+505282560+--branch_at+505282560 "
    "pytry+round11_e4.py one+d384+F4k384:0+--label+F4k384+--tokens+699531264+--branch_at+365035520,699531264+--extra=--keep_branch=365035520",
    "JAX_PLATFORMS=cpu python round11_eval.py corpora                   # E1 corpora cache",
]
EXTRA_CMDS = os.path.join(OUT, "commands_extra.txt")


def main():
    res, L = {}, ["# Round 11 — is the two-level head a cheap AND good head?", "",
                  "Generated by `PYTHONIOENCODING=utf8 python round11_analyze.py` (Windows Python) from `runs/round8`–`runs/round11`; every "
                  "decision rule is pinned in the script's docstring (18:20, before any run it depends on finished). Seed 0 unless stated.", ""]
    EB = jl(os.path.join(OUT, "eval_bpb.jsonl"))
    SM = js(os.path.join(OUT, "smoke_check_all.json"), {}).get("arms", {})
    runs = run_points()
    VERD = []
    part_a(runs, L, res)
    part_c(L, res, SM)
    part_b(L, res, EB)
    L += ["## Part E — what else does the two-level head change?", "",
          "Models: d=384 B384 (300M re-run), H3*-384 300M (re-run) and 575M (round 10), F8k-384 (both branches), the d=384 Part-C arm if run; "
          "d=128 B_r8, H3k8c4, tH3rs (1200M). Comparisons at equal tokens and at equal FLOPs (H3*-384 575M 6.72e16 vs B384 300M 6.39e16: 5% more).", ""]
    e1(L, res, EB)
    e2(L, res)
    e3(L, res)
    e4(L, res)
    e5(L, res)
    part_d(L, res)
    headline(res, EB)
    soc = sorted(glob.glob(os.path.join(OUT, "social", "*.png")))
    L += ["## Figures", "",
          "- Headline: `runs/round11/round11_headline.png` (+ `.svg`): (a) BPB vs FLOPs at d=384, (b) d=128 loss vs tokens for B, H3k8c4, tH3rs.",
          "- `runs/round11/frontier_loss_vs_flops.png`, `runs/round11/frontier_loss_vs_wall.png`, `runs/round11/knowledge_bits.png`.",
          "- Social (positive verdicts only; made by `python round11_social.py`; drafts in `runs/round11/social/tweets.md`): "
          + (", ".join(f"`runs/round11/social/{os.path.basename(p)}`" for p in soc) if soc else "none") + ".", ""]
    notes = os.path.join(OUT, "round11_notes.md")
    if os.path.exists(notes):
        L += [open(notes, encoding="utf8").read(), ""]
    L += ["## Exact commands", "", "```"] + COMMANDS + ([l.rstrip("\n") for l in open(EXTRA_CMDS, encoding="utf8") if l.strip()] if os.path.exists(EXTRA_CMDS) else []) + \
         ["PYTHONIOENCODING=utf8 python round11_analyze.py", "```", ""]
    for r in runs:
        r.pop("run", None)
    open(os.path.join(OUT, "round11_summary.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "round11_results.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
