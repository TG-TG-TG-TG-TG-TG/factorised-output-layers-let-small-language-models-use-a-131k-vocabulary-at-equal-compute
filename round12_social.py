"""Round 12 Part F: social images (1600x900 and 1080x1080 PNG, light theme) and social/thread.md.

Every number is read from runs/round12/round12_results.json (and runs/round11/round11_results.json for the round-11
bios rows). An image whose verdict is not positive is skipped, and the reason is written to social/README.md.

python round12_social.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round12", "social")
os.makedirs(OUT, exist_ok=True)
R12 = json.load(open(os.path.join(HERE, "runs", "round12", "round12_results.json"), encoding="utf8"))
R11 = json.load(open(os.path.join(HERE, "runs", "round11", "round11_results.json"), encoding="utf8"))
sys.path.insert(0, os.path.join(HERE, "paper"))

# round-10 palette (light): ink / muted / grid, flat = neutral grey, two-level = blue, accent = amber
INK, MUTED, GRID, BG = "#1f1e1c", "#6f6e69", "#d9d7d0", "#ffffff"
FLAT, TWO, ACC = "#6f6e69", "#2a78d6", "#eda100"
PX = 72 / 100  # points per pixel at dpi 100
FS_T, FS_L, FS_S = 44 * PX, 32 * PX, 28 * PX  # title / labels / smallest text (>= 28 px)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_L, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "axes.linewidth": 1.4, "figure.facecolor": BG})
SIZES = {"1600x900": (16.0, 9.0), "1080x1080": (10.8, 10.8)}
AR = [x for x in R12["all_runs"] if not x.get("ngram")]  # n-gram / lookup arms are out of scope
PM50 = max(x["params"] for x in AR if x["vocab"] in (50257, 50304) and x["d"] <= 384)
PBIG = max((x["params"] for x in AR if x["vocab"] == 131072 and x["d"] == 384), default=None)
SCALE = f"tiny LMs, d=128–384, ≤ {int(np.ceil(PM50 / 1e6))}M params, FineWeb-Edu"  # brief's "≤ 31M": the d=384 50k runs are 31.5M
EQF = "equal training FLOPs"
made, skipped = [], []


def get(d, path):
    for k in path.split("/"):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def frame(name, title, footer, draw):
    for tag, (w, h) in SIZES.items():
        fig = plt.figure(figsize=(w, h), dpi=100)
        sq = tag == "1080x1080"
        top, bot = (0.80, 0.25) if sq else (0.84, 0.23)
        fig.text(0.04, 0.955, title, ha="left", va="top", fontsize=FS_T * (0.85 if sq else 1), weight="bold", color=INK,
                 wrap=True)
        ax = fig.add_axes([0.14 if sq else 0.10, bot, 0.82 if sq else 0.87, top - bot])
        draw(fig, ax, sq)
        fig.text(0.04, 0.025, footer, ha="left", va="bottom", fontsize=FS_S, color=MUTED, wrap=True)
        p = os.path.join(OUT, f"{name}_{tag}.png")
        fig.savefig(p, dpi=100, facecolor=BG)
        plt.close(fig)
        made.append(os.path.relpath(p, HERE).replace("\\", "/"))


def runs_pts(names):
    """{flops: [bpb per seed]} for these runs (all_runs rows, logged BPB)."""
    pts = {}
    for x in R12["all_runs"]:
        if x["run"] in names and x["bpb"] is not None:
            k = min(pts, key=lambda f: abs(f / x["flops"] - 1), default=None)
            k = k if k is not None and abs(k / x["flops"] - 1) < 0.01 else x["flops"]
            pts.setdefault(k, []).append(x["bpb"])
    return dict(sorted(pts.items()))


# ------------------------------------------------------------------ 1 hero
D1 = get(R12, "partD/D1") or {}
hd = D1.get("headline_delta")
seed_d = [get(D1, f"{s}/delta") for s in ("0", "1")]
hero_ok = hd is not None and hd <= -0.003  # the paper's rule (make_numbers: headlinewin): beyond the pinned +-0.003 band
if hero_ok:
    flat = runs_pts({"wsd_B384_300M_s0", "wsd_B384_300M_s1", "wsd_B384m_315M_s0", "wsd_B384m_315M_s1"})
    two = runs_pts({"wsd_H3s384_575M_s0", "wsd_H3s384_575M_s1", "wsd_H3s384_300M_s0"})
    mH, mB = D1["mean_H"], D1["mean_B"]
    xf = max(two)
    # "flat needs N x the compute" only if a measured flat point reaches the two-level BPB (no extrapolation)
    fx = sorted(flat)
    fy = [float(np.mean(flat[f])) for f in fx]
    nx = None
    for i in range(1, len(fx)):
        if fy[i] <= mH < fy[i - 1]:  # log-linear interpolation between two measured flat points
            t = (fy[i - 1] - mH) / (fy[i - 1] - fy[i])
            nx = float(np.exp(np.log(fx[i - 1]) + t * (np.log(fx[i]) - np.log(fx[i - 1])))) / xf

    def draw_hero(fig, ax, sq):
        for pts, col, lab, mk in ((flat, FLAT, "flat softmax head", "o"), (two, TWO, "two-level head", "s")):
            xs = list(pts)
            ax.plot(xs, [np.mean(pts[x]) for x in xs], "-", color=col, lw=3.5, label=lab, marker=mk, ms=14, mec=BG, mew=2)
            for x in xs:
                if len(pts[x]) > 1:
                    ax.plot([x] * len(pts[x]), pts[x], "_", color=col, ms=18, mew=2.5)
        ax.annotate("", xy=(xf * 1.04, mH), xytext=(xf * 1.04, mB), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5,
                                                                                     mutation_scale=28))
        txt = f"{hd:+.3f} bits/byte\n(mean of 2 seeds)"
        if nx is not None:
            txt += f"\nflat needs {nx:.1f}× the compute"
        ax.text(xf * 1.08, (mH + mB) / 2, txt, va="center", ha="left", fontsize=FS_L, color=INK, weight="bold")
        ax.set_xscale("log")
        ax.set_xlim(min(min(flat), min(two)) / 1.15, xf * (2.7 if sq else 1.7))
        ax.set_xlabel("training FLOPs")
        ax.set_ylabel("bits per byte (lower is better)")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([2e16, 4e16, 6e16])
        ax.set_xticklabels(["2e16", "4e16", "6e16"])
        ax.grid(alpha=0.6, color=GRID, lw=1)
        ax.legend(frameon=False, fontsize=FS_L, loc="upper right")

    frame("hero", "Same compute, better LM: a two-level softmax head", f"{SCALE} · d=384 · {EQF} · seeds 0 and 1", draw_hero)
else:
    skipped.append(("hero", f"D1 2-seed Δ = {hd} (per seed {seed_d}): needs ≤ −0.003 (the paper's headlinewin rule)"))

# ------------------------------------------------------------------ 2 vocabulary
V = get(R12, "partB/verdicts") or {}
win = get(V, "two_level_wins_128k/holds")
rows = get(R12, "partB/rows") or {}
# seed count: the verdict uses seed 0 unless the +-0.003 rule fired (then seed-1 runs exist and the analysis averages them)
vseeds = "seed 0; inside the ±0.003 band, seed 1 needed" if get(V, "two_level_wins_128k/seed1") else "seed 0"
if win:
    flat_v = [(4096, "F4k-384 (700M)"), (8192, "F8k-384 (631M)"), (50257, "B384 matched (315M)"), (131072, "F128k-384 (161M)")]
    two_v = [(50257, "H3*-384 (575M)"), (131072, "H128k-384 (575M)")]
    pmax = max((rows[k].get("params") or 0) for _, k in flat_v + two_v if k in rows)

    def draw_vocab(fig, ax, sq):
        for ser, col, lab, mk in ((flat_v, FLAT, "flat softmax head", "o"), (two_v, TWO, "two-level head", "s")):
            xy = [(v, rows[k]["bpb"]) for v, k in ser if k in rows and rows[k].get("bpb") is not None]
            ax.plot(*zip(*xy), "-", color=col, lw=3.5, marker=mk, ms=14, mec=BG, mew=2, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([4096, 8192, 50257, 131072])
        ax.set_xticklabels(["4k", "8k", "50k", "128k"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("vocabulary size")
        ax.set_ylabel("bits per byte (lower is better)")
        ax.grid(alpha=0.6, color=GRID, lw=1)
        ax.legend(frameon=False, fontsize=FS_L)

    frame("vocab", "At a 128k vocabulary, two-level gets better, flat gets worse",
          f"tiny LMs, d=384, ≤ {pmax / 1e6:.0f}M params, FineWeb-Edu · {EQF} (6.7e16) · {vseeds}", draw_vocab)
else:
    skipped.append(("vocab", f"'two-level wins at 128k' = {win} (Δ = {get(V, 'two_level_wins_128k/delta')})"))

# ------------------------------------------------------------------ 3 head cost
C = get(R12, "theory/cost_d384")
r50, r131 = get(R12, "theory/head_ratio_50k"), get(R12, "theory/head_ratio_131k")
if C and r50:
    def draw_cost(fig, ax, sq):
        bars = [("flat\n50k", C["flat_50k"], FLAT), ("two-level\n50k", C["two_level_50k"], TWO),
                ("flat\n131k", C["flat_131k"], FLAT), ("two-level\n131k", C["two_level_131k"], TWO)]
        xs = [0, 1, 2.4, 3.4]
        for x, (lab, v, col) in zip(xs, bars):
            ax.bar(x, 100 * v, width=0.78, color=col)
            ax.text(x, 100 * v + 2, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=FS_L, color=INK, weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([b[0] for b in bars], fontsize=FS_S)
        ax.set_ylim(0, 100)
        ax.set_ylabel("head's share of training FLOPs (%)")
        for x1, v1, r in ((1, C["two_level_50k"], r50), (3.4, C["two_level_131k"], r131)):
            ax.text(x1, 100 * v1 + 11, f"head FLOPs\n× {r:.3f}", ha="center", va="bottom", fontsize=FS_S, color=TWO,
                    weight="bold")
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    frame("headcost", f"Two-level head: {r50:.3f}× the head FLOPs",
          f"{SCALE} · cost model at d=384", draw_cost)
else:
    skipped.append(("headcost", "theory/cost_d384 missing from the results JSON"))

# ------------------------------------------------------------------ 4 knowledge
D2 = get(R12, "partD/D2") or {}
if D2.get("holds"):
    bB = [D2[s]["bits_B"] for s in ("0", "1")]
    bH = [D2[s]["bits_H_eqF"] for s in ("0", "1")]
    ex = D2["0"]["exposures_H"]

    def draw_know(fig, ax, sq):
        for x, vals, col, lab in ((0, bB, FLAT, "flat head\n100 exposures"), (1, bH, TWO, f"two-level head\n{ex:.0f} exposures")):
            m = np.mean(vals) / 1e3
            ax.bar(x, m, width=0.6, color=col, yerr=[[m - min(vals) / 1e3], [max(vals) / 1e3 - m]], capsize=14,
                   error_kw=dict(lw=2.5, ecolor=INK, capthick=2.5))
            ax.text(x, max(vals) / 1e3 + 0.03 * max(bH) / 1e3, f"{m:,.0f}k bits", ha="center", va="bottom", fontsize=FS_L, color=INK, weight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["flat head\n100 exposures", f"two-level head\n{ex:.0f} exposures"], fontsize=FS_S)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(bH) / 1e3 * 1.25)
        ax.set_ylabel("knowledge stored (thousand bits)")
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    frame("knowledge", f"Same compute, {D2['mean_ratio']:.2f}× the knowledge bits",
          f"tiny LMs, d=128, synthetic biographies (100k people) · {EQF} · bars: mean of 2 seeds; title: mean of per-seed ratios ({D2['0']['ratio']:.2f}, {D2['1']['ratio']:.2f})", draw_know)
else:
    skipped.append(("knowledge", f"D2 verdict = {D2.get('verdict')} (ratios {[get(D2, s + '/ratio') for s in ('0', '1')]})"))


# ------------------------------------------------------------------ 5 schematic
def draw_schem(fig, ax, sq):
    """Boxes are text bboxes, so they always fit their text; layout is left-to-right (wide) or top-to-bottom (square)."""
    pos = ax.get_position()
    ax.set_position([0.04, pos.y0, 0.92, pos.height])  # full width: the schematic has no y-axis labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def node(xy, text, ec, fc="white", tc=INK, bold=False):
        return ax.text(*xy, text, ha="center", va="center", fontsize=FS_L, color=tc, weight="bold" if bold else "normal",
                       bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.5", fc=fc, ec=ec, lw=2.5), zorder=3)

    def link(a, b, col=MUTED, lab=None, lxy=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=30,
                                                         shrinkA=0, shrinkB=0), zorder=2)
        if lab:
            ax.text(*lxy, lab, ha="center", va="center", fontsize=FS_S, color=col, style="italic")

    if sq:
        P = {"h": (0.5, 0.86), "top": (0.5, 0.57), "f": (0.24, 0.20), "c": (0.74, 0.20)}
        A = [((0.5, 0.81), (0.5, 0.68)), ((0.40, 0.46), (0.25, 0.30)), ((0.60, 0.46), (0.74, 0.32))]
        L = [None, (0.20, 0.39), (0.82, 0.40)]
    else:
        P = {"h": (0.08, 0.55), "top": (0.40, 0.55), "f": (0.82, 0.82), "c": (0.82, 0.30)}
        A = [((0.15, 0.55), (0.24, 0.55)), ((0.56, 0.62), (0.68, 0.79)), ((0.56, 0.48), (0.66, 0.34))]
        L = [None, (0.58, 0.78), (0.555, 0.30)]
    node(P["h"], "hidden state h", MUTED, "#f4f3ef")
    node(P["top"], "top-level softmax\n\n8,192 frequent tokens\n+ 4,096 clusters of\nrare tokens", TWO, "#eaf2fc")
    node(P["f"], "frequent token:\nits probability\nis read directly", MUTED)
    node(P["c"], "cluster picked:\na small softmax over\nits members only", ACC, "#fdf4e1")
    for (a, b), lab, lxy, col in zip(A, (None, "p(w | h)", "p(cluster | h)"), L, (MUTED, MUTED, ACC)):
        link(a, b, col, lab, lxy)
    ax.text(0.5, -0.05 if sq else 0.0, "p(rare w | h) = p(cluster | h) × p(w | cluster, h)", ha="center", va="bottom",
            fontsize=FS_L, color=INK)


frame("schematic", "The two-level softmax head", "frequent tokens get their own logit; rare tokens share a cluster logit,"
      " then a small softmax picks the member", draw_schem)

# ------------------------------------------------------------------ 6 speed, 7 memory (follow-up benchmark, round12_perf.py)
PF = get(R12, "perf") or {}
PAIRS = (("50k", "B384", "H3s384"), ("131k", "F128k384", "H128k384"))
PFOOT = (f"tiny LMs, d=384, ≤ {int(np.ceil(PBIG / 1e6)) if PBIG else '?'}M params · RTX 5070 Ti, random weights, "
         f"real val text")


def pv(kind, arm, n, key):
    r = get(PF, f"{kind}/{arm}/{n}")
    return r.get(key) if r and r.get("ok") else None


WORK = (("training\n16 rows", "train", "16"), ("scoring\n64 rows", "score", "64"),
        ("decoding\nbatch 1", "decode", "1"), ("decoding\nbatch 128", "decode", "128"))
SPD = {V: [None if pv(k, fa, n, "tok_s") is None or pv(k, ha, n, "tok_s") is None else pv(k, ha, n, "tok_s") / pv(k, fa, n, "tok_s")
           for _, k, n in WORK] for V, fa, ha in PAIRS}
if PF and all(v is not None for vs in SPD.values() for v in vs) and SPD["50k"][0] > 1:
    def draw_speed(fig, ax, sq):
        w = 0.36
        for j, (V, col) in enumerate((("50k", TWO), ("131k", ACC))):
            for i, v in enumerate(SPD[V]):
                x = i + (j - 0.5) * w * 1.08
                ax.bar(x, v, width=w, color=col, label=f"{V} vocabulary" if i == 0 else None)
                ax.text(x, v + 0.08, f"{v:.1f}×", ha="center", va="bottom", fontsize=FS_S, color=INK, weight="bold")
        ax.axhline(1, color=INK, lw=1.6, ls="--", label="flat head = 1×")
        ax.set_xticks(range(len(WORK)))
        ax.set_xticklabels([w_[0] for w_ in WORK], fontsize=FS_S)
        ax.set_ylim(0, max(max(v) for v in SPD.values()) * 1.22)
        ax.set_ylabel("speed vs flat head (×)")
        ax.legend(frameon=False, loc="upper right", fontsize=FS_S)
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    frame("speed", f"Two-level head trains {SPD['50k'][0]:.1f}× faster ({SPD['131k'][0]:.1f}× at a 131k vocab)",
          PFOOT + " · tokens/s, two-level ÷ flat at the same vocabulary", draw_speed)
else:
    skipped.append(("speed", "perf results missing or no training speed-up"))

# training only: its measured peak matches XLA's estimate; the scoring peak is set before the call (see round12_notes.md)
MEMW = tuple((f"{n} rows/step", "train", n, "peak_gb") for n in ("16", "32", "64"))
MEM = {(V, h): [pv(k, arm, n, key) for _, k, n, key in MEMW] for V, fa, ha in PAIRS for h, arm in (("flat", fa), ("two", ha))}
if PF and all(v is not None for vs in MEM.values() for v in vs) and all(
        MEM[(V, "two")][i] < MEM[(V, "flat")][i] for V in ("50k", "131k") for i in range(len(MEMW))):
    def draw_mem(fig, ax, sq):
        w = 0.19
        order = (("50k", "flat", FLAT, None), ("50k", "two", TWO, None), ("131k", "flat", FLAT, "//"), ("131k", "two", TWO, "//"))
        for j, (V, h, col, hat) in enumerate(order):
            for i, v in enumerate(MEM[(V, h)]):
                x = i + (j - 1.5) * w * 1.1
                ax.bar(x, v, width=w, color=col, hatch=hat, edgecolor=BG, lw=0,
                       label=f"{'flat' if h == 'flat' else 'two-level'} {V}" if i == 0 else None)
                ax.text(x, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=FS_S, color=INK)
        ax.set_xticks(range(len(MEMW)))
        ax.set_xticklabels([m[0] for m in MEMW], fontsize=FS_S)
        ax.set_ylim(0, max(max(v) for v in MEM.values()) * 1.45)
        ax.set_ylabel("peak GPU memory (GB)")
        ax.set_xlabel("training step, micro-batch of 1,024-token rows", fontsize=FS_S, color=MUTED)
        ax.legend(frameon=False, loc="upper right", fontsize=FS_S, ncol=2)
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    saved = MEM[("50k", "flat")][0] - MEM[("50k", "two")][0]
    frame("memory", f"…and less memory: −{saved:.1f} GB per training step at 50k",
          PFOOT + " · measured peak, same batch for both heads", draw_mem)
else:
    skipped.append(("memory", "perf results missing or two-level does not use less memory everywhere"))

# ------------------------------------------------------------------ thread
pending = "[pending]"


def n(x, f):
    return pending if x is None else format(x, f)


E1 = get(R11, "E1/deltas/equal FLOPs (6.72e16 vs 6.39e16)") or {}
E1t = get(R11, "E1/deltas/equal tokens (300M)") or {}
BL = get(R11, "E2/acc") or {}
if hd is None:
    hook = "two-level head vs flat softmax head at equal training FLOPs: [result pending]"
elif hd < 0:
    hook = (f"swapping the flat softmax head for a two-level one (8,192 frequent tokens + 4,096 clusters of rare ones) "
            f"lowers bits/byte by {-hd:.3f} at equal training FLOPs (mean of 2 seeds)")
else:
    hook = (f"a two-level softmax head (8,192 frequent tokens + 4,096 clusters) does not beat the flat head at equal "
            f"training FLOPs ({hd:+.3f} bits/byte, 2 seeds)")
tw = [f"Tiny-LM result: {hook}. Paper: [LINK] 🧵",
      "Method: 6-block GPT, d=128–384, FineWeb-Edu. Every arm gets the same training FLOPs and is scored in bits per byte "
      "on the same 23.2M validation bytes, so vocabularies compare fairly. Rules for the main comparisons were fixed before the runs.",
      f"Why: at d=384 the flat head is {n(C and 100 * C['flat_50k'], '.0f')}% of training FLOPs. The two-level head costs "
      f"{n(r50, '.3f')}× as much ({n(r131, '.3f')}× at a 131k vocab), so the same budget buys more tokens"
      + (f": measured, it trains {SPD['50k'][0]:.1f}× faster ({SPD['131k'][0]:.1f}× at 131k)" if SPD["50k"][0] else "")
      + f". Per token it is still worse ({n(get(R11, 'E1/deltas/equal tokens (300M)/fineweb'), '+.3f')} BPB at 300M tokens)."]
if win is not None:
    tw.append(f"Vocabulary: at 128k tokens the two-level head {'beats' if win else 'does not beat'} the flat head "
              f"({n(get(V, 'two_level_wins_128k/delta'), '+.3f')} BPB at equal FLOPs, {vseeds}). Flat 4k/8k vocabularies are "
              f"worse than the two-level head at 50k.")
ev = get(D1, "evals") or {}
hm, bm = ev.get("H3s384_575M") or {}, ev.get("B384m_s0") or {}
if all(x.get(k) is not None for x in (hm, bm) for k in ("wikitext103", "pg19", "blimp")):  # FLOP-matched flat (D1), preferred
    ow, op, bh, bb, how = (hm["wikitext103"] - bm["wikitext103"], hm["pg19"] - bm["pg19"], hm["blimp"], bm["blimp"],
                           "FLOP-matched flat, seed 0")
elif E1:  # round 11: flat at 6.39e16 vs two-level at 6.72e16
    ow, op, bh, bb, how = (E1.get("wikitext103"), E1.get("pg19"), get(BL, "H3s384_575M/overall"), get(BL, "B384/overall"),
                           "flat at 5% fewer FLOPs, seed 0")
else:
    ow = None
bdm = get(R11, "E2/deltas/equal FLOPs (6.72e16 vs 6.39e16)/delta_mean")
if ow is not None:
    tw.append(f"Out of domain, two-level minus flat: {ow:+.3f} BPB on WikiText-103, {op:+.3f} on PG-19 ({how}). "
              + (f"Grammar (BLiMP): no difference over 2 seeds (mean {n(bdm, '+.1f')} points, inside the pre-set ±1 band)."
                 if bdm is not None else f"Grammar (BLiMP): {n(bh, '.1f')}% vs {n(bb, '.1f')}% flat (seed 0)."))
if D2.get("ratio") is not None or D2.get("mean_ratio") is not None or get(D2, "0/ratio") is not None:
    tw.append(f"Knowledge: on synthetic biographies (d=128), the equal-FLOPs two-level model stores "
              f"{n(D2.get('mean_ratio'), '.2f')}× the bits of facts of the flat model "
              f"({' / '.join(n(get(D2, s + '/ratio'), '.2f') for s in ('0', '1'))} on seeds 0 / 1). Not at capacity.")
tw.append(f"Caveats: tiny models ({PM50 / 1e6:.1f}M params at a 50k vocab), 1–2 seeds per claim, one dataset. The idea is "
          "old: class-based softmax (Goodman 2001), hierarchical softmax (Morin & Bengio 2005), adaptive softmax (Grave et al. "
          "2017). What's new is the equal-FLOPs test.")
assert 5 <= len(tw) <= 7, len(tw)
for i, t in enumerate(tw):
    assert len(t) <= 280, (i, len(t), t)
with open(os.path.join(OUT, "thread.md"), "w", encoding="utf8", newline="\n") as f:
    f.write("# Thread draft (round 12)\n\nGenerated by `round12_social.py` from the results JSONs. "
            "Character counts in brackets; the limit is 280.\n\n")
    IMG = {"Tiny-LM": ("hero",), "Method:": ("schematic",), "Why:": ("headcost", "speed", "memory"),
           "Vocabulary:": ("vocab",), "Knowledge:": ("knowledge",)}  # Twitter allows up to 4 images per tweet
    for i, t in enumerate(tw, 1):
        im = [m for k in IMG.get(t.split()[0], ()) for m in made if m.endswith(f"/{k}_1600x900.png")]
        f.write(f"{i}/{len(tw)} [{len(t)} chars]" + (f" — images: {', '.join(im)}" if im else "") + f"\n\n{t}\n\n")
    f.write("Images: " + ", ".join(m for m in made if m.endswith("1600x900.png")) + "\n")
with open(os.path.join(OUT, "README.md"), "w", encoding="utf8", newline="\n") as f:
    f.write("# Social images\n\n" + "\n".join(f"- `{m}`" for m in made) + "\n\nSkipped (verdict not positive or data missing):\n\n"
            + ("\n".join(f"- {a}: {b}" for a, b in skipped) or "- none") + "\n")
print("made:", *made, sep="\n  ")
print("skipped:", *skipped, sep="\n  ")
