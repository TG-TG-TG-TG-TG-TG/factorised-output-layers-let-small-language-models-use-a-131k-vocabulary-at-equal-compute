"""Round 13 analysis (2026-11-11): answers to the two reviews. Rules for experiments 1 and 2 pinned here at 21:40 CET on
2026-11-11, before any round-13 run started (copy: runs/round13/round13_analyze_docstring_2140.py.txt). Experiment 3
(adaptive softmax) gets its own dated section, pinned before any of its runs. Anything decided after seeing a result is
labelled post-hoc in the summary. Rounds 8-12 results are read, never rewritten. BPB = bits per UTF-8 byte of the shared
FineWeb-Edu val text, from the logged full-val loss of the branch (GPT-2-vocabulary arms, round 11's gate: equal to the
byte-aligned evaluator within 2e-5 BPB), exactly as round12_analyze.gpt2_bpb.

Budget points. 1x = 6.716845e16 training FLOPs (H3*-384 575M, stable + decay, logged flops_per_token.total x tokens);
0.5x = 3.504925e16 (the 300M point); 2x = 1.343369e17. A branch counts for a point if its FLOPs are within 0.5% of it.
Proxy cost P = FLOPs of wsd_tBproxy_50M_s0's final branch (stable + decay, logged), expected 2.7729e15 = 4.1% of 1x.
"Charged" = the two-level branch at (point - P); "uncharged" = the branch at the point itself.

Experiment 1 -- charge the 50k proxy. The 50k clusters of every earlier H3*-384 run came from wsd_B_r8_s0 (d=128 flat,
  1,200M stable tokens + decay = 6.66e16 FLOPs, 99% of 1x), never charged. Rebuilt from a cheap proxy:
  proxy  wsd_tBproxy_50M_s0 = arm tB (B_r8's recipe) at 50,003,968 stable tokens + 10% decay, seed 0 -- the 50k twin of
         round 12's 131k proxy. Classes: round13_classes.py (frequent list unchanged, tail k-means on the proxy's
         embedding).
  arm    H3p384 = H3*-384 with those classes (head_classes "adapt13"); everything else identical to wsd_H3s384_575M.
  runs   wsd_H3p384_575M_s0 and _s1: stable to 575e6 (8,774 steps), branches at 276,299,776 (0.5x charged),
         300e6 (0.5x uncharged), 551,288,832 (1x charged), 575e6 (1x uncharged).
  flat   B384 matched, round 12, unchanged: wsd_B384m_315M_s0/_s1, branches 164,429,824 (0.5x) and 315,162,624 (1x).
  Numbers: Delta_ch(1x) = mean_s BPB(H3p384 @551.3M) - mean_s BPB(B384m @315.2M); Delta_unch(1x) the same at 575M;
    per seed, the means, and the spread |s0 - s1| per side; the same at 0.5x; and (reported only) the cluster-source
    effect mean_s BPB(H3p384 @575M) - mean_s BPB(H3s384 @575M, rounds 11/12, B_r8 clusters).
  Verdict: the 50k headline survives charging if Delta_ch(1x) <= -0.003 AND each seed's charged delta (seed s vs seed s)
    is < 0. Otherwise it does not survive, and the paper is rescoped around the 131k result (already charged, -0.0736;
    the user's decision in the review brief). No seed 2 either way.

Experiment 2 -- budget scaling, small version (2x only; 4x/8x not run, the user's decision). One stable run per arm,
  seed 0, same recipe and data order (WSD: the stable phase up to a point is the same run as at 1x; branches decay 10%):
  flat       wsd_B384x2_630M_s0 = B:0 stable to 630,259,712 (9,617 steps; 2x within 0.01%), branches 315,162,624 (1x,
             a reproducibility check against wsd_B384m_315M_s0, |diff| reported) and 630,259,712.
  two-level  wsd_H3p384x2_1150M_s0 = H3p384:0 stable to 1,149,960,192 (17,547 steps), branches 1,126,236,160 (2x
             charged) and 1,149,960,192 (2x uncharged).
  Numbers: gap(F) = BPB(two-level) - BPB(flat), seed 0 vs seed 0, at F = 0.5x, 1x, 2x, charged (main series) and
    uncharged; the 2-seed gaps at 0.5x and 1x next to them. Crossover: least-squares line of the three seed-0 charged
    gaps against log2(F/1x); F* = where it reaches 0. If gap(2x) >= 0 the crossover is observed between 1x and 2x (linear
    interpolation in log2 F); if F* > 2x it is reported as an extrapolation from three points, not a result; the
    shrinkage is called monotone only if |gap| falls at both steps. |gap(2x)| <= 0.003 is reported as a tie at 2x; no
    seed-1 2x runs in any case.

Experiment 3 -- adaptive softmax baseline. Pinned at 13:27 CET on 2026-11-12, before any experiment-3 run or smoke
  (copy: runs/round13/round13_analyze_docstring_exp3.py.txt). Head: round13_heads.py (cfg.head "adasm"; Grave et al.
  2017): the two-level head's K = 8,192 frequent singletons, tied to the embedding (asserted the same list), plus 2 band
  logits; tail = 2 frequency-ordered bands, ranks [8,192, 24,576) and [24,576, V), each its own untied low-rank layer of
  width d/4 = 96 and d/16 = 24. One setting, chosen before any run from the literature defaults (division factor 4),
  never swept: no cutoff, width, init or LR changes after seeing a number (LR 2e-3, the recipe of every d=384 arm).
  Gates (round13_gates_as.py, CPU, all passed before the pin: runs/round13/gates_as.json): distribution sums to 1,
  fast NLL = reference NLL, gradients = reference gradients, backbone bit-identical to B at init, FLOP count = the hand
  count, evaluator path = training loss.
  FLOPs: charged as Grave et al.'s training algorithm runs -- the head for every token, band b only for targets in it,
    weighted by the band's training-target share (9.71% / 3.06% at 50k; 7.85% / 6.49% at 131k). This favours the
    baseline: the dense count (every band for every token, what the implementation computes) is 12.6% / 23.6% more
    total FLOPs per token, and it is reported next to it. Training fpt: AS384 97,788,832.3; AS128k384 98,492,631.0.
    No proxy, nothing to charge (the bands come from target counts, as the two-level head's frequent list).
  runs   wsd_AS384_624M_s0 and _s1: AS384 stable to 624,427,008 (1x within 0.002%), branch 325,844,992 (0.5x).
         wsd_AS128k384_620M_s0: AS128k384 stable to 619,970,560 (1x within 0.001%), branch 323,485,696 (0.5x).
  BPB: 50k from the logged val_loss_full x BPN as every GPT-2 arm (the evaluator path equals the training loss, gate
    A6); 131k from round11_eval.task_bpb (byte-aligned) on the saved checkpoints, as F128k / H128k in round 12.
  Numbers at 1x (0.5x the same, reported only):
    Delta_TL-AS(50k) = mean_s BPB(H3p384 @551.3M, charged) - mean_s BPB(AS384 @624.4M), seed s vs seed s, per seed and
      mean -- the main comparison (both sides proxy-free or charged); reported next to it: vs H3p384 uncharged @575M and
      vs H3s384 (B_r8 clusters, uncharged); Delta_AS-flat(50k) = mean_s BPB(AS384) - mean_s BPB(B384m @315.2M).
    Delta_TL-AS(131k) = BPB(H128k384 @521.8M, charged) - BPB(AS128k384 @620.0M), seed 0 only, labelled one seed;
      Delta_AS-flat(131k) = BPB(AS128k384) - BPB(F128k384 @160.8M).
    Also: head parameters, training tok/s and ms/token from the logs (dense implementation; not a speed claim for the
    method).
  Verdict (50k; the 131k one is the same rule on one seed): two-level beats adaptive softmax if Delta_TL-AS <= -0.003
    AND each seed's delta < 0; adaptive softmax beats two-level if Delta_TL-AS >= +0.003 AND each seed's delta > 0;
    otherwise a tie at this budget. What each outcome does to the paper (pinned now, so the framing is not chosen after
    the number):
      two-level wins  -> the contribution may say the two-level head beats a frequency-band adaptive softmax at matched
                         FLOPs (one untuned setting; FLOPs charged in the baseline's favour).
      tie             -> no claim over adaptive softmax; the paper says the gain over the flat head is shared with an
                         existing factorised softmax, and the head's own contribution is reduced to what else it
                         supports (decoding speed, the tied cluster logits) -- stated as such in the abstract.
      adaptive wins   -> said in the abstract; the two-level head is presented as a weaker member of the factorised-
                         softmax family, and the design-choice contribution is dropped.
  Failures: a crash or NaN is reported as is; one rerun only for an infrastructure failure (same command), never for a
    number. No seed 2, no 131k seed 1.
  Addendum 2026-11-12 ~18:50, implementation only (no rule, model or FLOP-count change), before any experiment-3 result:
    the 200-step smokes put the dense implementation at 169k tok/s (AS384, 75 min) and 84k tok/s (AS128k384, 150 min,
    over the user's 2-hour rule), so the 131k run was held (marker file) and is run as arm AS128k384s = AS128k384 +
    head_sparse: each tail band is computed only for the tokens whose target is in it (gather, score, scatter; a chunk
    with more band tokens than the capacity takes the dense path), i.e. Grave et al.'s own algorithm. Same parameters,
    init, FLOP count and run name (label AS128k384). Gated equal to the dense path (round13_gates_as_sparse.py, CPU:
    NLL within 1e-5, gradients within 1e-5 relative, both branches; runs/round13/gates_as_sparse.json); the dense path's
    own gates re-run bit-identical after the refactor. The 50k runs (AS384, seeds 0 and 1) stay on the dense path.

Experiment 4 -- does the two-level head beat the best adaptive softmax at 131k on two seeds? Pinned 2026-11-13, before
  any experiment-4 run or smoke (copy: runs/round13/round13_analyze_docstring_exp4.py.txt), after the experiment-3
  results were known (50k tie on 2 seeds; 131k two-level ahead by 0.0223 on seed 0). The user's instruction of
  2026-11-13 replaces experiment 3's "no 131k seed 1": the paper is to be about the two-level head, claiming it makes the
  131k vocabulary pay off at equal compute, and that framing is written only if this gate passes.
  Runs (131k = bpe131072, d=384, 1x = 6.7168e16 training FLOPs, LR 2e-3, the round-12 / experiment-3 recipe):
    wsd_H128k384_575M_s1     H128k384 seed 1, the round-12 seed-0 command (stable to 574,750,720, branches 299,892,736
                             / 521,797,632 kept), the same cluster files (from the d=128 proxy wsd_tF128k_proxy_50M_s0,
                             whose FLOPs are charged: the 521.8M branch is the charged 1x point, as for seed 0) -- the
                             50k precedent (experiment 1 seed 1 used the seed-0 proxy's clusters).
    wsd_AS128k384_620M_s1    AS128k384s seed 1 (label AS128k384), the experiment-3 seed-0 command.
    wsd_AS128k384c49_612M_s0 AS128k384c49 seed 0: the second adaptive-softmax configuration, one setting, not swept.
                             Same 8,192 shortlist (the two-level head's frequent list), same divisor 4 (d/4 = 96, d/16
                             = 24), second cutoff 24,576 -> 49,152. Reason, fixed before any run: the 131k weak spot of
                             configuration 1 is its narrow last band (106,496 tokens at width 24, 6.49% of targets);
                             49,152 puts 3.05% of targets in it, the 50k setting's share (3.06%) where adaptive softmax
                             tied the two-level head; band shares 11.28% / 3.05%. Charged head forward 7,309,422
                             FLOPs/token (configuration 1: 6,878,621); training fpt 99,785,034.3, so stable to
                             611,909,632 (1x within 0.002%), branch 319,356,928 (0.5x, reported only). Gated
                             (round13_gates_as.py / _sparse.py with arm AS128k384c49 -> gates_as_c49.json,
                             gates_as_sparse_c49.json).
  BPB: round11_eval.task_bpb (byte-aligned, FineWeb-Edu val) on the saved checkpoints, as experiment 3's 131k numbers.
  Gate (the user's rule, "not below -0.003 on both seeds" -> stop): with TL(s) = BPB(H128k384 seed s @521.8M, charged),
    A1(s) = BPB(AS128k384 seed s @620.0M), A2(0) = BPB(AS128k384c49 seed 0 @611.9M):
      best adaptive softmax on seed 0 = min(A1(0), A2(0)); Delta(0) = TL(0) - min(A1(0), A2(0)).
      if A1(0) <= A2(0): configuration 1 is the best one, Delta(1) = TL(1) - A1(1).
      if A2(0) <  A1(0): the best configuration has no seed-1 run, so Delta(1) is NOT DEFINED by these runs: stop,
                         report, and ask the user whether to run AS128k384c49 seed 1 (no adjustment of A1(1) is used).
    PASS iff Delta(0) < -0.003 AND Delta(1) < -0.003 (strict). PASS -> the head-centred framing may be written, with the
    seed count stated. Anything else (FAIL, undefined, a crashed or NaN run) -> stop and report; the framing is the
    user's decision, not chosen here.
  Reported next to it, not part of the gate: the 0.5x branches; mean over seeds; Delta vs flat (F128k384 seed 0 only);
    both adaptive-softmax configurations' numbers on seed 0; WikiText-103 and PG-19 BPB.
  Failures: a crash or NaN is reported as is; one rerun only for an infrastructure failure (same command, the run
    restarts from step 0), never for a number. No third seed, no third configuration, no LR or init change.
  Follow-up, pinned 2026-11-13 18:15, after the experiment-4 runs above gave UNDEFINED (configuration 2 best on seed
    0: A2(0) = 1.1629 < A1(0) = 1.1735; Delta(0) = -0.0117) and before the run below. User: "Yes, run seed 1".
    Run: wsd_AS128k384c49_612M_s1 -- AS128k384c49 seed 1, the wsd_AS128k384c49_612M_s0 command with seed 1 (stable to
    611,909,632, 0.5x branch 319,356,928 kept). No other run.
    Delta(1) = TL(1) - min(A1(1), A2(1)): the best adaptive softmax ON SEED 1 (at least as strict as using the seed-0
    winner; both differences are reported). The seed-0 side is unchanged: Delta(0) = TL(0) - min(A1(0), A2(0)).
    PASS iff Delta(0) < -0.003 AND Delta(1) < -0.003 (strict); anything else -> stop and report, as above. The
    seed-1 numbers already known when this was pinned: TL(1) = 1.1539, A1(1) = 1.1738.

Experiment 5 -- is the 131k win over adaptive softmax an artefact of its narrow tail bands? Pinned 2026-11-13 21:40 CET,
  before any experiment-5 run or smoke (copy: runs/round13/round13_analyze_docstring_exp5.py.txt). Asked by an outside
  review after the paper was rewritten around experiment 4's PASS: setup 2 scores its rarest band (81,920 of the 131k
  tokens) through a 24-dimensional projection (reduction factor 4 per band, fixed before any run); at d=384 that
  band may simply be too narrow. Everything in experiments 1-4 is known when this is pinned (TL(0) = 1.1512, TL(1) = 1.1539, A2(0) =
  1.1629, A2(1) = 1.1628).
  Run: wsd_AS128k384c49w_540M_s0 -- arm AS128k384c49w = setup 2 (AS128k384c49: cutoffs 8,192 / 49,152, the better
    adaptive softmax on both seeds, sparse path) with ONE change, cfg.head_full: both tail bands at full width d = 384
    with no projection (logits h . W_b[j]), W_b still untied (n_b x 384). Same shortlist (tied), init rule (W_b ~
    N(0, 1/384)), weight decay, LR 2e-3, data, seed, schedule. Tying the tail to the input embedding would be a second
    change and stays untested. Gated on CPU before any run: round13_gates_as.py (A1-A7, gates_as_c49w.json) and
    round13_gates_as_sparse.py vs the dense twin AS128k384c49wd (gates_as_sparse_c49w.json), both PASS; the old arms'
    gates re-run bit-identical after the change (AS384, AS128k384, AS128k384c49, sparse c49).
  Budget (equal training FLOPs, the rule of every round-13 run: (S + round(0.1 S)) x 65,536 tokens x fpt closest to
    1x = 6.716845e16): charged fpt 113,150,621.0 (head 35,293,853.0; setup 2: 99,785,034.3), so stable to 539,623,424
    (8,234 steps, 1x within 0.010%) and a 0.5x branch at 281,608,192 (kept, reported only). This is a fair
    equal-compute test: the wider head costs more, so the baseline sees 11.8% fewer tokens than setup 2, and still more
    than the charged two-level head (521,797,632). Head parameters 47.2M (setup 2: 5.9M); the model has about 108M.
  BPB: round11_eval.task_bpb (byte-aligned) on the saved checkpoint, label AS128k384c49w_540M, as experiment 4.
  Rule, with AW(s) = BPB(AS128k384c49w seed s @ 1x) and Delta_w(s) = TL(s) - AW(s) (TL = experiment 4's charged
    two-level point):
    seed 0 first.  Delta_w(0) < -0.009              -> PASS (1 seed).
                   Delta_w(0) > +0.003              -> ADAPTIVE SOFTMAX WINS (1 seed).
                   -0.009 <= Delta_w(0) <= +0.003   -> SEED 1 NEEDED (within 0.006 of the -0.003 threshold): run
                                                       wsd_AS128k384c49w_540M_s1 (the same command, seed 1) before any
                                                       conclusion; no conclusion from seed 0 alone.
    with seed 1:   PASS iff Delta_w(0) < -0.003 AND Delta_w(1) < -0.003; ADAPTIVE SOFTMAX WINS iff both > +0.003;
                   anything else TIE.
    (PASS at seed 0 needs AW(0) > 1.1602; the -0.003 threshold itself corresponds to AW(0) > 1.1542.)
  What each outcome does to the paper, decided now:
    PASS -> the title and the head-centred framing stay. The full-width run is added as a third adaptive-softmax setup
      (abstract, 5.2, Limitations), with its seed count; the 24-dim-band caveat is replaced by this result, and the
      "best adaptive softmax" of the 2-seed gate stays labelled as the best of setups 1-2.
    TIE -> the title claim goes. The paper becomes "factorised output layers let small models use a 131k vocabulary",
      with the two-level head as one of two heads that work; the abstract and 5.2 say the 131k advantage over setups
      1-2 came from their narrow tail bands.
    ADAPTIVE SOFTMAX WINS -> the title claim goes, as for TIE, and the paper says plainly that a full-width adaptive
      softmax beats the two-level head at 131k in this setting.
  Reported next to it, not part of the rule: the 0.5x branch, full width minus setup 2 (the width effect), and
    WikiText-103 / PG-19 BPB.
  Failures: a crash or NaN is reported as is; one rerun only for an infrastructure failure (same command, from step 0),
    never for a number. No other width, cutoff, tying, LR or init variant; no third seed.
  Speed: the chain stops after the smoke if its full-run estimate is not under 115 min (round13_require_fast.py). Such
    a stop changes no number; any speed change afterwards (for example the sparse caps, which leave results exact
    because a chunk over its cap takes the dense path) is pinned here before the run.
"""
import json
import os

import numpy as np

from round9_analyze import BPN, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
RD = {r: os.path.join(HERE, "runs", f"round{r}") for r in (8, 10, 11, 12, 13)}
OUT = RD[13]
EXP5_LABEL = "AS128k384c49w_540M"  # experiment 5 eval label (seed 1: + "_s1")
X1 = 6.716844637929216e16  # round 12's T575
PTS = {"0.5x": 3.504924836453376e16, "1x": X1, "2x": 2 * X1}
TOK = {  # stable tokens of each branch used (pinned above)
    "H": {"0.5x": (300023808, 276299776), "1x": (575012864, 551288832)},  # (uncharged, charged)
    "H2": {"2x": (1149960192, 1126236160)},
    "B": {"0.5x": 164429824, "1x": 315162624},
    "B2": {"1x": 315162624, "2x": 630259712},
}


def R(rnd, name):
    run = load_run(os.path.join(RD[rnd], name + ".jsonl"))
    if run:
        run["name"] = name
    return run


def br(run, tok, tol=0.002):
    if not run:
        return None
    ks = [k for k in run["branch"] if abs(k - tok) <= tol * tok]
    return run["branch"][ks[0]] if ks else None


def point(run, tok):
    """BPB, FLOPs and wall of the branch of `run` at `tok` stable tokens (None if missing)"""
    b = br(run, tok)
    if b is None:
        return None
    f = run["final"]["flops_per_token"]["total"] * b["tokens"]
    return {"run": run["name"], "stable_tokens": tok, "tokens": b["tokens"], "bpb": b["loss"] * BPN, "flops": f,
            "wall_min": b["wall"] / 60}


def f4(x, s="+.4f"):
    return "-" if x is None else format(x, s)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def ev(rows, label, corpus="fineweb"):
    m = [r for r in rows if r.get("label") == label and r.get("corpus") == corpus]
    return m[-1]["bpb"] if m else None


def verdict3(d, per_seed):
    """pinned three-way rule (experiment 3)"""
    if d is None or any(x is None for x in per_seed):
        return None
    if d <= -0.003 and all(x < 0 for x in per_seed):
        return "two-level wins"
    if d >= 0.003 and all(x > 0 for x in per_seed):
        return "adaptive softmax wins"
    return "tie"


def exp3(L, H, Hold, B):
    """Experiment 3 (rules pinned 2026-11-12 13:27): adaptive softmax vs the two-level head and the flat head at 1x / 0.5x."""
    AS = {s: R(13, f"wsd_AS384_624M_s{s}") for s in (0, 1)}
    A128 = R(13, "wsd_AS128k384_620M_s0")
    EB12, EB13 = jl(os.path.join(RD[12], "eval_bpb.jsonl")), jl(os.path.join(OUT, "eval_bpb.jsonl"))
    TA = {"0.5x": 325844992, "1x": 624427008}
    L += ["## Experiment 3 — adaptive softmax baseline (rules pinned 2026-11-12 13:27)", "",
          "50k, d=384. Δ TL−AS = two-level (charged proxy clusters) − adaptive softmax, seed s vs seed s; Δ AS−flat likewise.", "",
          "| point | seed | flat | adaptive softmax | two-level charged | two-level uncharged | old clusters (uncharged) | Δ TL−AS | Δ AS−flat |",
          "|---|---|---|---|---|---|---|---|---|"]
    e3 = {"50k": {}}
    for pt in ("0.5x", "1x"):
        tu, tc = TOK["H"][pt]
        rows = {}
        for s in (0, 1):
            a, b, hc, hu, ho = (point(AS[s], TA[pt]), point(B[s], TOK["B"][pt]), point(H[s], tc), point(H[s], tu),
                                point(Hold[s], tu))
            d_ta = None if not (a and hc) else hc["bpb"] - a["bpb"]
            d_af = None if not (a and b) else a["bpb"] - b["bpb"]
            rows[s] = {"as": a, "d_tl_as": d_ta, "d_as_flat": d_af,
                       "d_tlunch_as": None if not (a and hu) else hu["bpb"] - a["bpb"],
                       "d_tlold_as": None if not (a and ho) else ho["bpb"] - a["bpb"]}
            g = lambda x: f4(x and x["bpb"], ".4f")
            L.append(f"| {pt} | {s} | {g(b)} | {g(a)} | {g(hc)} | {g(hu)} | {g(ho)} | {f4(d_ta)} | {f4(d_af)} |")
        ds = [rows[s]["d_tl_as"] for s in (0, 1)]
        dm = mean(ds) if all(x is not None for x in ds) else None
        af = [rows[s]["d_as_flat"] for s in (0, 1)]
        e3["50k"][pt] = {"seeds": rows, "d_tl_as_mean": dm, "d_as_flat_mean": mean(af) if all(x is not None for x in af) else None,
                         "d_tlunch_as_mean": mean([rows[s]["d_tlunch_as"] for s in (0, 1)]),
                         "d_tlold_as_mean": mean([rows[s]["d_tlold_as"] for s in (0, 1)]),
                         "verdict": verdict3(dm, ds) if pt == "1x" else None}
        L.append(f"| {pt} | mean | | | | | | {f4(dm)} | {f4(e3['50k'][pt]['d_as_flat_mean'])} |")
    v = e3["50k"]["1x"]["verdict"]
    L += ["", f"**Verdict 50k (pinned):** {v or 'pending (needs both seeds)'}"
          + ("" if v is None else f" (Δ TL−AS = {e3['50k']['1x']['d_tl_as_mean']:+.4f})."), ""]
    if AS[0]:
        fl = AS[0]["final"]["flops_per_token"]
        e3["fpt"] = fl
        e3["params"] = AS[0]["final"].get("params")
    # 131k, seed 0, byte-aligned BPB
    a1, h1, f1 = ev(EB13, "AS128k384_620M"), ev(EB12, "H128k384_522M"), ev(EB12, "F128k384_161M")
    a5, h5, f5 = ev(EB13, "AS128k384_323M"), ev(EB12, "H128k384_300M"), ev(EB12, "F128k384_84M")
    d1 = None if a1 is None or h1 is None else h1 - a1
    e3["131k"] = {"1x": {"as": a1, "tl_charged": h1, "flat": f1, "d_tl_as": d1,
                         "d_as_flat": None if a1 is None or f1 is None else a1 - f1, "verdict": verdict3(d1, [d1])},
                  "0.5x": {"as": a5, "tl": h5, "flat": f5, "d_tl_as": None if a5 is None or h5 is None else h5 - a5}}
    L += ["131k (bpe131072), seed 0 only, byte-aligned BPB:", "",
          "| point | flat | adaptive softmax | two-level (1x charged, 0.5x uncharged) | Δ TL−AS | Δ AS−flat |", "|---|---|---|---|---|---|",
          f"| 0.5x | {f4(f5, '.4f')} | {f4(a5, '.4f')} | {f4(h5, '.4f')} | {f4(e3['131k']['0.5x']['d_tl_as'])} | "
          f"{f4(None if a5 is None or f5 is None else a5 - f5)} |",
          f"| 1x | {f4(f1, '.4f')} | {f4(a1, '.4f')} | {f4(h1, '.4f')} | {f4(d1)} | {f4(e3['131k']['1x']['d_as_flat'])} |", "",
          f"**Verdict 131k (pinned rule, one seed):** {e3['131k']['1x']['verdict'] or 'pending'}.", ""]
    return e3


def gate4(tl, a1, a2):
    """pinned experiment-4 gate (2026-11-13). tl / a1: {seed: BPB}, a2: {0: BPB}. Returns (verdict, d0, d1, best)."""
    if any(x is None for x in (tl.get(0), tl.get(1), a1.get(0), a1.get(1), a2.get(0))):
        return "pending (missing runs)", None, None, None
    best = "configuration 1" if a1[0] <= a2[0] else "configuration 2"
    d0 = tl[0] - min(a1[0], a2[0])
    if best == "configuration 2" and a2.get(1) is None:
        return "UNDEFINED: configuration 2 is best on seed 0 and has no seed-1 run -> stop, ask", d0, None, best
    d1 = tl[1] - (a1[1] if a2.get(1) is None else min(a1[1], a2[1]))   # follow-up 18:15: best AS on seed 1
    return ("PASS" if d0 < -0.003 and d1 < -0.003 else "FAIL"), d0, d1, best


def exp4(L):
    """Experiment 4 (rules pinned 2026-11-13): 131k, two seeds, the best of two adaptive-softmax configurations."""
    EB12, EB13 = jl(os.path.join(RD[12], "eval_bpb.jsonl")), jl(os.path.join(OUT, "eval_bpb.jsonl"))
    lab = {"tl": {0: (EB12, "H128k384_522M"), 1: (EB13, "H128k384_522M_s1")},
           "a1": {0: (EB13, "AS128k384_620M"), 1: (EB13, "AS128k384_620M_s1")},
           "a2": {0: (EB13, "AS128k384c49_612M"), 1: (EB13, "AS128k384c49_612M_s1")},
           "tl05": {0: (EB12, "H128k384_300M"), 1: (EB13, "H128k384_300M_s1")},
           "a105": {0: (EB13, "AS128k384_323M"), 1: (EB13, "AS128k384_323M_s1")},
           "a205": {0: (EB13, "AS128k384c49_319M"), 1: (EB13, "AS128k384c49_319M_s1")}}
    corp = ("fineweb", "wikitext103", "pg19")
    v = {k: {s: {c: ev(rows, l, c) for c in corp} for s, (rows, l) in d.items()} for k, d in lab.items()}
    fw = lambda k: {s: x["fineweb"] for s, x in v[k].items()}
    verdict, d0, d1, best = gate4(fw("tl"), fw("a1"), fw("a2"))
    flat = {c: ev(EB12, "F128k384_161M", c) for c in corp}
    L += ["## Experiment 4 — 131k, two seeds, best of two adaptive-softmax configurations (pinned 2026-11-13)", "",
          "FineWeb-Edu byte-aligned BPB at 1x (equal training FLOPs; two-level charged). Configuration 1 = cutoffs "
          "(8,192, 24,576), configuration 2 = (8,192, 49,152); both divisor 4.", "",
          "| seed | flat | adaptive softmax cfg 1 | adaptive softmax cfg 2 | two-level charged | Δ TL − best AS |",
          "|---|---|---|---|---|---|"]
    for s, d in ((0, d0), (1, d1)):
        L.append(f"| {s} | {f4(flat['fineweb'] if s == 0 else None, '.4f')} | {f4(v['a1'][s]['fineweb'], '.4f')} | "
                 f"{f4(v['a2'].get(s, {}).get('fineweb'), '.4f')} | {f4(v['tl'][s]['fineweb'], '.4f')} | {f4(d)} |")
    a2v, tlv = fw("a2"), fw("tl")
    d1w = None if a2v.get(1) is None or tlv.get(1) is None else tlv[1] - a2v[1]
    L += ["", f"Best adaptive softmax on seed 0: {best or '-'}. Seed 1 minus the seed-0 winner (configuration 2), "
          f"reported only: {f4(d1w)}.", "",
          f"**Gate (pinned; PASS iff Δ < −0.003 on both seeds):** {verdict}.", "",
          "Reported only — 0.5x (two-level uncharged) and the other corpora at 1x:", "",
          "| seed | point / corpus | adaptive softmax cfg 1 | adaptive softmax cfg 2 | two-level |", "|---|---|---|---|---|"]
    for s in (0, 1):
        L.append(f"| {s} | 0.5x fineweb | {f4(v['a105'][s]['fineweb'], '.4f')} | "
                 f"{f4(v['a205'].get(s, {}).get('fineweb'), '.4f')} | {f4(v['tl05'][s]['fineweb'], '.4f')} |")
        for c in corp[1:]:
            L.append(f"| {s} | 1x {c} | {f4(v['a1'][s][c], '.4f')} | {f4(v['a2'].get(s, {}).get(c), '.4f')} | "
                     f"{f4(v['tl'][s][c], '.4f')} |")
    L.append("")
    return {"bpb": v, "flat_s0": flat, "verdict": verdict, "delta_s0": d0, "delta_s1": d1, "best_seed0": best}


def gate5(tl, aw):
    """pinned experiment-5 rule (2026-11-13). tl / aw: {seed: BPB}. Returns (verdict, d0, d1)."""
    if tl.get(0) is None or aw.get(0) is None:
        return "pending (missing run)", None, None
    d0 = tl[0] - aw[0]
    if aw.get(1) is None:
        if d0 < -0.009:
            return "PASS (1 seed)", d0, None
        if d0 > 0.003:
            return "ADAPTIVE SOFTMAX WINS (1 seed)", d0, None
        return "SEED 1 NEEDED (-0.009 <= delta(0) <= +0.003) -> run AS128k384c49w seed 1, then re-run", d0, None
    d1 = tl[1] - aw[1]
    if d0 < -0.003 and d1 < -0.003:
        return "PASS (2 seeds)", d0, d1
    if d0 > 0.003 and d1 > 0.003:
        return "ADAPTIVE SOFTMAX WINS (2 seeds)", d0, d1
    return "TIE (2 seeds)", d0, d1


def exp5(L, e4):
    """Experiment 5 (rule pinned 2026-11-13): full-width adaptive softmax (setup 2 cutoffs, no projection, untied)."""
    EB13 = jl(os.path.join(OUT, "eval_bpb.jsonl"))
    corp = ("fineweb", "wikitext103", "pg19")
    aw = {s: {c: ev(EB13, f"{EXP5_LABEL}{'' if s == 0 else '_s1'}", c) for c in corp} for s in (0, 1)}
    tl = {s: e4["bpb"]["tl"][s]["fineweb"] for s in (0, 1)}
    a2 = {s: e4["bpb"]["a2"][s]["fineweb"] for s in (0, 1)}
    awf = {s: aw[s]["fineweb"] for s in (0, 1)}
    # 0.5x branch (reported only, as pinned; the 0.5x two-level points are uncharged, as in experiment 4)
    aw05 = {s: ev(EB13, f"AS128k384c49w_282M{'' if s == 0 else '_s1'}", "fineweb") for s in (0, 1)}
    verdict, d0, d1 = gate5(tl, awf)
    wid = {s: None if awf[s] is None or a2[s] is None else awf[s] - a2[s] for s in (0, 1)}
    L += ["## Experiment 5 — full-width adaptive softmax at 131k (rule pinned 2026-11-13)", "",
          "Setup 2 cutoffs (8,192, 49,152); tail bands at width 384 with no projection, untied; everything else as setup 2. "
          "FineWeb-Edu byte-aligned BPB at 1x (equal training FLOPs; two-level charged).", "",
          "| seed | adaptive softmax setup 2 (widths 96 / 24) | adaptive softmax full width | two-level charged | "
          "Δ TL − full width | full width − setup 2 |", "|---|---|---|---|---|---|"]
    for s, d in ((0, d0), (1, d1)):
        L.append(f"| {s} | {f4(a2[s], '.4f')} | {f4(awf[s], '.4f')} | {f4(tl[s], '.4f')} | {f4(d)} | {f4(wid[s])} |")
    L += ["", f"**Rule (pinned):** {verdict}.", "",
          "Reported only — other corpora at 1x: " + "; ".join(
              f"seed {s} {c} {f4(aw[s][c], '.4f')}" for s in (0, 1) for c in corp[1:] if aw[s][c] is not None), "",
          "Reported only — 0.5x FineWeb-Edu (two-level uncharged): " + "; ".join(
              f"seed {s} full width {f4(aw05[s], '.4f')}, two-level {f4(e4['bpb']['tl05'][s]['fineweb'], '.4f')}"
              for s in (0, 1) if aw05[s] is not None), ""]
    return {"bpb": aw, "bpb_05": aw05, "verdict": verdict, "delta_s0": d0, "delta_s1": d1, "width_effect": wid}


def main():
    L = ["# Round 13 — answers to the reviews (experiments 1 and 2)", "",
         "Rules pinned in round13_analyze.py's docstring at 21:40 CET on 2026-11-11, before any round-13 run. "
         "BPB = bits per byte of the FineWeb-Edu val text. Seed 0 unless stated.", ""]
    res = {}
    # ---------------- proxy
    px = R(13, "wsd_tBproxy_50M_s0")
    P = None
    if px:
        P = px["final"]["flops_per_token"]["total"] * px["final"]["tokens"]
        res["proxy"] = {"run": px["name"], "tokens": px["final"]["tokens"], "flops": P, "share_of_1x": P / X1,
                        "wall_min": px["final"]["wall_s"] / 60}
        L += [f"Proxy {px['name']} (d=128 flat, GPT-2 vocabulary): {px['final']['tokens'] / 1e6:.1f}M tokens, {P:.4e} FLOPs "
              f"({100 * P / X1:.1f}% of 1x), {px['final']['wall_s'] / 60:.1f} min.", ""]
    rep = os.path.join(OUT, "adapt_report_8192_4096.json")
    if os.path.exists(rep):
        res["classes"] = json.load(open(rep))
    B = {s: R(12, f"wsd_B384m_315M_s{s}") for s in (0, 1)}
    H = {s: R(13, f"wsd_H3p384_575M_s{s}") for s in (0, 1)}
    Hold = {0: R(10, "wsd_H3s384_575M_s0"), 1: R(11, "wsd_H3s384_575M_s1")}
    # ---------------- experiment 1
    L += ["## Experiment 1 — the 50k proxy charged", "",
          "| point | seed | flat BPB | two-level uncharged | Δ uncharged | two-level charged (FLOPs incl. proxy) | Δ charged |",
          "|---|---|---|---|---|---|---|"]
    e1 = {}
    for pt in ("0.5x", "1x"):
        tu, tc = TOK["H"][pt]
        rows = {}
        for s in (0, 1):
            b, hu, hc = point(B[s], TOK["B"][pt]), point(H[s], tu), point(H[s], tc)
            if hc and P is not None:
                hc["flops_incl_proxy"] = hc["flops"] + P
                hc["vs_point"] = hc["flops_incl_proxy"] / PTS[pt]
            du = None if not (b and hu) else hu["bpb"] - b["bpb"]
            dc = None if not (b and hc) else hc["bpb"] - b["bpb"]
            rows[s] = {"flat": b, "unch": hu, "ch": hc, "d_unch": du, "d_ch": dc}
            L.append(f"| {pt} | {s} | {f4(b and b['bpb'], '.4f')} | {f4(hu and hu['bpb'], '.4f')} | {f4(du)} | "
                     f"{f4(hc and hc['bpb'], '.4f')} ({'-' if not hc or 'vs_point' not in hc else format(hc['vs_point'], '.4f')}x) | {f4(dc)} |")
        mb = mean([r["flat"] and r["flat"]["bpb"] for r in rows.values()])
        mu = mean([r["unch"] and r["unch"]["bpb"] for r in rows.values()])
        mc = mean([r["ch"] and r["ch"]["bpb"] for r in rows.values()])
        full = all(rows[s][k] for s in (0, 1) for k in ("flat", "unch", "ch"))
        e1[pt] = {"seeds": rows, "two_seed": full,
                  "d_unch_mean": None if not full else mu - mb, "d_ch_mean": None if not full else mc - mb,
                  "spread_flat": None if not full else abs(rows[0]["flat"]["bpb"] - rows[1]["flat"]["bpb"]),
                  "spread_ch": None if not full else abs(rows[0]["ch"]["bpb"] - rows[1]["ch"]["bpb"])}
        L.append(f"| {pt} | mean | {f4(mb if full else None, '.4f')} | {f4(mu if full else None, '.4f')} | "
                 f"{f4(e1[pt]['d_unch_mean'])} | {f4(mc if full else None, '.4f')} | {f4(e1[pt]['d_ch_mean'])} |")
    L.append("")
    c1 = e1["1x"]
    if c1["two_seed"]:
        ok = c1["d_ch_mean"] <= -0.003 and all(c1["seeds"][s]["d_ch"] < 0 for s in (0, 1))
        c1["verdict_survives"] = ok
        L += [f"**Verdict (pinned):** charged 2-seed Δ at 1x = {c1['d_ch_mean']:+.4f} (seeds {c1['seeds'][0]['d_ch']:+.4f} / "
              f"{c1['seeds'][1]['d_ch']:+.4f}) → the 50k headline **{'survives' if ok else 'does not survive'}** charging"
              + ("." if ok else "; the paper is rescoped around the 131k result (the user's rule)."), ""]
    else:
        L += ["**Verdict:** pending (needs both seeds).", ""]
    cs = [point(H[s], 575012864) for s in (0, 1)], [point(Hold[s], 575012864) for s in (0, 1)]
    if all(cs[0]) and all(cs[1]):
        d = mean([x["bpb"] for x in cs[0]]) - mean([x["bpb"] for x in cs[1]])
        e1["cluster_source_effect"] = d
        L += [f"Cluster source (reported only): proxy clusters − B_r8 clusters at 575M, 2-seed means: {d:+.4f} BPB.", ""]
    elif cs[0][0] and cs[1][0]:
        e1["cluster_source_effect_s0"] = cs[0][0]["bpb"] - cs[1][0]["bpb"]
        L += [f"Cluster source (reported only, seed 0): {e1['cluster_source_effect_s0']:+.4f} BPB.", ""]
    res["exp1"] = e1
    # ---------------- experiment 2
    B2, H2 = R(13, "wsd_B384x2_630M_s0"), R(13, "wsd_H3p384x2_1150M_s0")
    L += ["## Experiment 2 — gap vs training FLOPs (seed 0)", "",
          "| point | FLOPs | flat BPB | two-level charged | gap charged | two-level uncharged | gap uncharged | 2-seed gap charged |",
          "|---|---|---|---|---|---|---|---|"]
    e2 = {}
    for pt in ("0.5x", "1x", "2x"):
        if pt == "2x":
            b, hu, hc = point(B2, TOK["B2"]["2x"]), point(H2, TOK["H2"]["2x"][0]), point(H2, TOK["H2"]["2x"][1])
            two = None
        else:
            r0 = e1[pt]["seeds"][0]
            b, hu, hc = r0["flat"], r0["unch"], r0["ch"]
            two = e1[pt]["d_ch_mean"]
        gc = None if not (b and hc) else hc["bpb"] - b["bpb"]
        gu = None if not (b and hu) else hu["bpb"] - b["bpb"]
        e2[pt] = {"flops": PTS[pt], "flat": b, "unch": hu, "ch": hc, "gap_ch": gc, "gap_unch": gu, "gap_ch_2seed": two}
        L.append(f"| {pt} | {PTS[pt]:.3e} | {f4(b and b['bpb'], '.4f')} | {f4(hc and hc['bpb'], '.4f')} | {f4(gc)} | "
                 f"{f4(hu and hu['bpb'], '.4f')} | {f4(gu)} | {f4(two)} |")
    L.append("")
    rp = point(B2, TOK["B2"]["1x"]), point(B[0], TOK["B"]["1x"])
    if all(rp):
        e2["repro_1x_flat"] = rp[0]["bpb"] - rp[1]["bpb"]
        L += [f"Reproducibility: the 2x flat run's 1x branch − wsd_B384m_315M_s0 = {e2['repro_1x_flat']:+.5f} BPB.", ""]
    g = [e2[p]["gap_ch"] for p in ("0.5x", "1x", "2x")]
    if all(x is not None for x in g):
        x = np.log2([0.5, 1.0, 2.0])
        a, c = np.polyfit(x, g, 1)  # gap = a * log2(F/1x) + c
        Fstar = 2 ** (-c / a) if a > 0 else None
        mono = abs(g[1]) < abs(g[0]) and abs(g[2]) < abs(g[1])
        e2["fit"] = {"slope_per_doubling": float(a), "intercept_1x": float(c), "F_star_x": Fstar, "monotone": mono}
        if g[2] >= 0:
            fr = np.interp(0, [g[1], g[2]], [0.0, 1.0])
            e2["crossover_observed_x"] = float(2 ** fr)
            txt = f"the gap reaches 0 between 1x and 2x (≈{2 ** fr:.2f}x by linear interpolation in log2 F)"
        elif Fstar:
            txt = (f"line through the three points: {a:+.4f} BPB per doubling, F* ≈ {Fstar:.1f}x "
                   f"({Fstar * X1:.2e} FLOPs) — an extrapolation from three points, not a result")
        else:
            txt = f"the line does not rise toward 0 (slope {a:+.4f} per doubling): no crossover predicted"
        L += [f"**Crossover:** {txt}. Shrinkage {'is' if mono else 'is not'} monotone (|gap| falls at both steps)."
              + (" Gap at 2x within ±0.003: a tie at 2x." if abs(g[2]) <= 0.003 else ""), ""]
    res["exp2"] = e2
    res["exp3"] = exp3(L, H, Hold, B)
    res["exp4"] = exp4(L)
    res["exp5"] = exp5(L, res["exp4"])
    open(os.path.join(OUT, "round13_summary.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "round13_results.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
