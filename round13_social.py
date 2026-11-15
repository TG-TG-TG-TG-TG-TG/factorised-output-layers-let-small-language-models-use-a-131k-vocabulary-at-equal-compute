"""Round 13: social images (1600x900 and 1080x1080 PNG, light theme) and runs/round13/social/thread.md, for the
rewritten paper (head-centred framing, allowed because experiment 4's pinned gate passed).
2026-11-14: experiment 5 (full-width adaptive softmax) gave TIE under its pinned rule, so the head-centred claim is gone:
the hero and thread now say that factorised heads (two-level, or adaptive softmax with full-width tails) make the 131k
vocabulary usable, and that the two-level head ties the full-width adaptive softmax with fewer parameters.

Every number is read from runs/round13/round13_results.json, runs/round13/gates_as*.json and (for the cost, speed,
memory and knowledge images, which do not involve the round-13 runs) runs/round12/round12_results.json. Round 12's
images in runs/round12/social are left as they are. An image whose verdict is not positive is skipped, and the reason is
written to social/README.md.

python round13_social.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round13", "social")
os.makedirs(OUT, exist_ok=True)
R13 = json.load(open(os.path.join(HERE, "runs", "round13", "round13_results.json"), encoding="utf8"))
R12 = json.load(open(os.path.join(HERE, "runs", "round12", "round12_results.json"), encoding="utf8"))
GA = {f: json.load(open(os.path.join(HERE, "runs", "round13", f), encoding="utf8"))
      for f in ("gates_as.json", "gates_as_c49.json", "gates_as_c49w.json")}

# the round-12 palette, plus green for adaptive softmax (the paper's colour)
INK, MUTED, GRID, BG = "#1f1e1c", "#6f6e69", "#d9d7d0", "#ffffff"
FLAT, TWO, ACC, ADA = "#6f6e69", "#2a78d6", "#eda100", "#1baf7a"
ADAW = "#c8551e"  # full-width adaptive softmax (the paper's colour)
PX = 72 / 100
FS_T, FS_L, FS_S = 44 * PX, 32 * PX, 28 * PX
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_L, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": FS_S, "ytick.labelsize": FS_S, "axes.linewidth": 1.4, "figure.facecolor": BG})
SIZES = {"1600x900": (16.0, 9.0), "1080x1080": (10.8, 10.8)}


def params13(run):
    for line in open(os.path.join(HERE, "runs", "round13", run + ".jsonl"), encoding="utf8"):
        d = json.loads(line)
        if "params" in d:
            return d["params"]["total"]


PMAX = params13("wsd_AS128k384c49w_540M_s0")  # the largest compared model (131k adaptive softmax, full width)
PTL = params13("wsd_H128k384_575M_s1")  # the 131k two-level model
SCALE = f"tiny LMs, 6 blocks, d=384, ≤ {int(np.ceil(PMAX / 1e6))}M params, FineWeb-Edu"
EQF = "equal training FLOPs, cluster cost charged"
made, skipped = [], []
mean = lambda xs: float(np.mean(xs))  # noqa: E731
sg = lambda v: f"{v:+.3f}".replace("-", "−")  # noqa: E731  (typographic minus in images)


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


# ------------------------------------------------------------------ numbers (2-seed means where two seeds exist)
E1, E2, E3, E4 = R13["exp1"], R13["exp2"], R13["exp3"], R13["exp4"]
B = E4["bpb"]
S = ("0", "1")
tl50 = mean([E1["1x"]["seeds"][s]["ch"]["bpb"] for s in S])
fl50 = mean([E1["1x"]["seeds"][s]["flat"]["bpb"] for s in S])
as50 = mean([E3["50k"]["1x"]["seeds"][s]["as"]["bpb"] for s in S])
tl131 = mean([B["tl"][s]["fineweb"] for s in S])
asb131 = {s: min(B["a1"][s]["fineweb"], B["a2"][s]["fineweb"]) for s in S}  # the better setup on each seed (the gate's rule)
as131 = mean(list(asb131.values()))
fl131 = E3["131k"]["1x"]["flat"]  # one seed
d131 = [B["tl"][s]["fineweb"] - asb131[s] for s in S]
d131m = mean(d131)
d131flat = B["tl"]["0"]["fineweb"] - fl131
d50as = E3["50k"]["1x"]["d_tl_as_mean"]
d50flat = E1["1x"]["d_ch_mean"]
d50asflat = E3["50k"]["1x"]["d_as_flat_mean"]
voc_tl, voc_as, voc_fl = tl131 - tl50, as131 - as50, fl131 - E1["1x"]["seeds"]["0"]["flat"]["bpb"]
gate = E4.get("verdict")
E5 = R13["exp5"]
gate5 = str(E5.get("verdict"))
aw131 = mean([E5["bpb"][s]["fineweb"] for s in S])
as2_131 = mean([B["a2"][s]["fineweb"] for s in S])
dw = [E5["delta_s0"], E5["delta_s1"]]
wid = [E5["width_effect"][s] for s in S]
best50 = min(tl50, as50)
gaps = [E2[p]["gap_ch"] for p in ("0.5x", "1x", "2x")]

# ------------------------------------------------------------------ 1 hero: factorised heads make 131k usable
if gate == "PASS" and gate5.startswith("TIE"):
    def draw_hero(fig, ax, sq):
        x = [0, 1]
        # direct labels at the middle of each line (no legend box)
        for ys, col, lab, mk, dy, va, xl in (([fl50, fl131], FLAT, "flat softmax head", "o", 0.003, "bottom", 0.5),
                                             ([as50, as2_131], ADA, "adaptive softmax, narrow tail", "^", 0.0012, "bottom", 0.5),
                                             ([tl50, tl131], TWO, "two-level head", "s", -0.0012, "top", 0.62)):
            ax.plot(x, ys, "-", color=col, lw=3.5, marker=mk, ms=15, mec=BG, mew=2)
            ax.text(xl, ys[0] + xl * (ys[1] - ys[0]) + dy, lab, ha="right" if col == FLAT else "center", va=va,
                    fontsize=FS_S, color=INK, weight="bold")
        ax.plot([1], [aw131], "D", color=ADAW, ms=26, mfc="none", mew=3.5)  # hollow: it ties the two-level point
        ax.text(1.09, aw131, f"adaptive softmax,\nfull-width tail:\ntie with two-level\n({sg(dw[0])} / {sg(dw[1])})",
                va="center", ha="left", fontsize=FS_S, color=INK, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["50k vocabulary", "131k vocabulary"])
        ax.set_xlim(-0.3, 1.75 if sq else 1.6)
        ax.set_ylim(min(tl131, tl50) - 0.012, fl131 + 0.006)
        ax.set_ylabel("bits per byte (lower is better)")
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)

    frame("hero", "Factorised softmax heads let tiny LMs use a 131k vocabulary",
          f"{SCALE} · {EQF} (6.7e16) · means of 2 seeds (flat 131k: 1 seed) · tie = within ±0.003", draw_hero)
else:
    skipped.append(("hero", f"experiment 4 = {gate}, experiment 5 = {gate5}: this hero is written for 4 PASS and 5 TIE"))

# ------------------------------------------------------------------ 2 compute: the gain over flat shrinks
def draw_compute(fig, ax, sq):
    xs = [0, 1, 2]
    ax.axhspan(-0.003, 0.003, color=GRID, alpha=0.6, lw=0, label="±0.003: counted as a tie")
    for x, g in zip(xs, gaps):
        ax.bar(x, g, width=0.6, color=TWO)
        ax.text(x, g - 0.0008, sg(g), ha="center", va="top", fontsize=FS_L, color=INK, weight="bold")
    ax.axhline(0, color=INK, lw=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.5× budget", "1× budget", "2× budget"])
    ax.set_ylim(min(gaps) * 1.35, 0.006)
    ax.set_ylabel("two-level − flat (bits/byte)")
    ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=FS_S, loc="lower right")


frame("compute", "The gain over a flat head shrinks with compute",
      f"{SCALE} · 50k vocabulary · {EQF} · seed 0 · 1× = 6.7e16 FLOPs; larger budgets untested", draw_compute)

# ------------------------------------------------------------------ 3 head cost (logged FLOPs per token)
rows = get(R12, "partB/rows") or {}
r50, r131 = get(R12, "theory/head_ratio_50k"), get(R12, "theory/head_ratio_131k")
a50 = GA["gates_as.json"]["AS384"]["A5"]
a131 = GA["gates_as_c49.json"]["AS128k384c49"]["A5"]
a131w = GA["gates_as_c49w.json"]["AS128k384c49w"]["A5"]
if rows and r50:
    sh = {"flat 50k": rows["B384 matched (315M)"]["head_share"], "two 50k": rows["H3*-384 (575M)"]["head_share"],
          "flat 131k": rows["F128k-384 (161M)"]["head_share"], "two 131k": rows["H128k-384 (575M)"]["head_share"],
          "as 50k": a50["head_train"] / a50["total"], "as 131k": a131["head_train"] / a131["total"],
          "asw 131k": a131w["head_train"] / a131w["total"]}

    def draw_cost(fig, ax, sq):
        bars = [("flat", sh["flat 50k"], FLAT), ("adaptive", sh["as 50k"], ADA), ("two-level", sh["two 50k"], TWO),
                ("flat", sh["flat 131k"], FLAT), ("adaptive", sh["as 131k"], ADA), ("adaptive\nfull width", sh["asw 131k"], ADAW),
                ("two-level", sh["two 131k"], TWO)]
        xs = [0, 1, 2, 3.6, 4.6, 5.6, 6.6]
        for x, (lab, v, col) in zip(xs, bars):
            ax.bar(x, 100 * v, width=0.8, color=col)
            ax.text(x, 100 * v + 2, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=FS_S, color=INK, weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([("adapt. full" if "\n" in b[0] else b[0]) if sq else b[0] for b in bars], fontsize=FS_S, rotation=30 if sq else 0,
                           ha="right" if sq else "center", rotation_mode="anchor")
        for xc, lab in ((1, "50k vocabulary"), (5.1, "131k vocabulary")):
            ax.text(xc, 96, lab, ha="center", va="top", fontsize=FS_S, color=MUTED)
        ax.set_ylim(0, 100)
        ax.set_ylabel("head's share of training FLOPs (%)")
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    frame("headcost", f"Head FLOPs: two-level = {r50:.3f}× flat ({r131:.3f}× at 131k)",
          f"{SCALE} · logged FLOPs per token · adaptive softmax charged by band share (in its favour; 131k narrow: setup 2)", draw_cost)
else:
    skipped.append(("headcost", "round-12 partB rows or theory ratios missing"))


# ------------------------------------------------------------------ 4 schematic (same as round 12)
def draw_schem(fig, ax, sq):
    pos = ax.get_position()
    ax.set_position([0.04, pos.y0, 0.92, pos.height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def node(xy, text, ec, fc="white"):
        return ax.text(*xy, text, ha="center", va="center", fontsize=FS_L, color=INK,
                       bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.5", fc=fc, ec=ec, lw=2.5), zorder=3)

    def link(a, b, col=MUTED, lab=None, lxy=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=30,
                                                         shrinkA=0, shrinkB=0), zorder=2)
        if lab:
            ax.text(*lxy, lab, ha="center", va="center", fontsize=FS_S, color=col, style="italic")

    if sq:
        # 2026-11-15 (outside review): boxes spread out; arrows are placed from the drawn box edges (below)
        P = {"h": (0.5, 1.02), "top": (0.5, 0.68), "f": (0.24, 0.20), "c": (0.74, 0.20)}
        A = None
        L = [None, (0.17, 0.42), (0.84, 0.42)]
    else:
        P = {"h": (0.08, 0.55), "top": (0.40, 0.55), "f": (0.82, 0.82), "c": (0.82, 0.30)}
        A = [((0.15, 0.55), (0.24, 0.55)), ((0.56, 0.62), (0.68, 0.79)), ((0.56, 0.48), (0.66, 0.34))]
        L = [None, (0.58, 0.78), (0.555, 0.30)]
    T = {"h": node(P["h"], "hidden state h", MUTED, "#f4f3ef"),
         "top": node(P["top"], "top-level softmax\n\n8,192 frequent tokens\n+ 4,096 clusters of\nrare tokens", TWO, "#eaf2fc"),
         "f": node(P["f"], "frequent token:\nits probability\nis read directly", MUTED),
         "c": node(P["c"], "cluster picked:\na small softmax over\nits members only", ACC, "#fdf4e1")}
    if A is None:  # square: arrows run from the bottom edge of one box to the top edge of the next, with a small gap
        fig.canvas.draw()  # the box patches are sized at draw time
        r = fig.canvas.get_renderer()
        E = {k: t.get_bbox_patch().get_window_extent(r).transformed(ax.transData.inverted()) for k, t in T.items()}
        g = 0.015
        A = [((0.5, E["h"].y0 - g), (0.5, E["top"].y1 + g)),
             ((E["top"].x0 + 0.08, E["top"].y0 - g), (E["f"].x0 + E["f"].width / 2, E["f"].y1 + g)),
             ((E["top"].x1 - 0.08, E["top"].y0 - g), (E["c"].x0 + E["c"].width / 2, E["c"].y1 + g))]
    for (a, b), lab, lxy, col in zip(A, (None, "p(w | h)", "p(cluster | h)"), L, (MUTED, MUTED, ACC)):
        link(a, b, col, lab, lxy)
    ax.text(0.5, -0.05 if sq else 0.0, "p(rare w | h) = p(cluster | h) × p(w | cluster, h)", ha="center", va="bottom",
            fontsize=FS_L, color=INK)


frame("schematic", "The two-level softmax head", "frequent tokens get their own logit; rare tokens share a cluster logit,"
      " then a small softmax picks the member", draw_schem)

# ------------------------------------------------------------------ 5 speed, 6 memory vs the flat head (round12_perf.py)
PF = get(R12, "perf") or {}
PAIRS = (("50k", "B384", "H3s384"), ("131k", "F128k384", "H128k384"))
PBIG = max((x["params"] for x in R12["all_runs"] if x["vocab"] == 131072 and x["d"] == 384 and not x.get("ngram")), default=None)
PFOOT = f"tiny LMs, d=384, ≤ {int(np.ceil(PBIG / 1e6)) if PBIG else '?'}M params · RTX 5070 Ti, random weights, real val text"


def pv(kind, arm, n, key):
    r = get(PF, f"{kind}/{arm}/{n}")
    return r.get(key) if r and r.get("ok") else None


WORK = (("training\n16 rows", "train", "16"), ("scoring\n64 rows", "score", "64"),
        ("decoding\nbatch 1", "decode", "1"), ("decoding\nbatch 128", "decode", "128"))
SPD = {V: [None if pv(k, fa, n, "tok_s") is None or pv(k, ha, n, "tok_s") is None else pv(k, ha, n, "tok_s") / pv(k, fa, n, "tok_s")
           for _, k, n in WORK] for V, fa, ha in PAIRS}
spd_ok = bool(PF) and all(v is not None for vs in SPD.values() for v in vs) and SPD["50k"][0] > 1
if spd_ok:
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

    frame("speed", f"vs a flat head: {SPD['50k'][0]:.1f}× training-step throughput ({SPD['131k'][0]:.1f}× at 131k)",
          PFOOT + " · tokens/s, two-level ÷ flat · no speed comparison with adaptive softmax", draw_speed)
else:
    skipped.append(("speed", "perf results missing or no training speed-up"))

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
    frame("memory", f"…and less memory than flat: −{saved:.1f} GB per training step at 50k",
          PFOOT + " · measured peak, same batch for both heads", draw_mem)
else:
    skipped.append(("memory", "perf results missing or two-level does not use less memory everywhere"))

# ------------------------------------------------------------------ 7 knowledge (round 12 D2, d=128 synthetic biographies)
D2 = get(R12, "partD/D2") or {}


def _final_rec(path):  # a run log's "final" record
    for line in open(os.path.join(HERE, *path.split("/")), encoding="utf8"):
        d = json.loads(line)
        if d.get("event") == "final":
            return d


_fr, _fb = _final_rec("runs/round8/wsd_B_r8_s0.jsonl"), _final_rec("runs/round11/wsd_bioB_100000_505M_s0.jsonl")
CCB = _fr["tokens"] * _fr["flops_per_token"]["total"] / (_fb["tokens"] * _fb["flops_per_token"]["total"])  # = paper ClusterCostBios
if D2.get("holds"):
    bB = [D2[s]["bits_B"] for s in S]
    bH = [D2[s]["bits_H_eqF"] for s in S]
    ex = D2["0"]["exposures_H"]

    def draw_know(fig, ax, sq):
        for x, vals, col in ((0, bB, FLAT), (1, bH, TWO)):
            m = np.mean(vals) / 1e3
            ax.bar(x, m, width=0.6, color=col, yerr=[[m - min(vals) / 1e3], [max(vals) / 1e3 - m]], capsize=14,
                   error_kw=dict(lw=2.5, ecolor=INK, capthick=2.5))
            ax.text(x, max(vals) / 1e3 + 0.03 * max(bH) / 1e3, f"{m:,.0f}k bits", ha="center", va="bottom", fontsize=FS_L,
                    color=INK, weight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["flat head\n100 exposures", f"two-level head\n{ex:.0f} exposures"], fontsize=FS_S)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(bH) / 1e3 * 1.25)
        ax.set_ylabel("knowledge stored (thousand bits)")
        ax.grid(axis="y", alpha=0.6, color=GRID, lw=1)
        ax.set_axisbelow(True)

    frame("knowledge", f"More exposures for the same training FLOPs: {D2['mean_ratio']:.2f}× the stored fact bits",
          f"tiny LMs, d=128, synthetic biographies (100k people) · equal training FLOPs, cluster cost NOT charged (built "
          f"from a FineWeb-Edu model costing {CCB:.1f}× this budget) · at equal exposures the heads are about level · bars: mean "
          f"of 2 seeds; title: mean of per-seed ratios ({D2['0']['ratio']:.2f}, {D2['1']['ratio']:.2f})", draw_know)
else:
    skipped.append(("knowledge", f"D2 verdict = {D2.get('verdict')}"))

# ------------------------------------------------------------------ thread
eqtok = get(json.load(open(os.path.join(HERE, "runs", "round11", "round11_results.json"), encoding="utf8")),
            "E1/deltas/equal tokens (300M)/fineweb")
C = get(R12, "theory/cost_d384") or {}
tw = []
if gate == "PASS" and gate5.startswith("TIE"):
    # 2026-11-14: the hook suggested in the outside review, chosen by the user
    tw.append(f"I thought my two-level head was what let a tiny LM use a 131k vocab. Then I gave adaptive softmax "
              f"a full-width tail, and it tied ({dw[0]:+.4f} / {dw[1]:+.4f} bits/byte, 2 seeds). What holds: at equal "
              f"training FLOPs, factorised heads beat a flat one. Paper: [LINK] 🧵")
else:
    tw.append(f"Tiny-LM result (experiment 5 verdict: {gate5}); see the paper: [LINK] 🧵")
tw.append("Method: 6-block GPT, d=384, FineWeb-Edu. Every arm gets the same training FLOPs, including the cost of building "
          "the clusters, and is scored in bits per byte on the same validation text. Decision rules were written down "
          "before the deciding runs.")
tw.append(f"Vocabulary, 50k → 131k at equal FLOPs: the flat head gets worse by {voc_fl:.3f} bits/byte (it spends "
          f"{100 * rows['F128k-384 (161M)']['head_share']:.0f}% of its FLOPs on the head). The 131k two-level and full-width "
          f"adaptive-softmax models are {best50 - tl131:.3f} and {best50 - aw131:.3f} better than the best d=384 50k model.")
tw.append(f"Adaptive softmax usually scores rare tokens through narrow projections (96 and 24 dims here). Making that tail "
          f"full width changed it by {wid[0]:+.3f} / {wid[1]:+.3f} bits/byte: from behind the two-level head to a tie. "
          f"Params: {PTL / 1e6:.1f}M two-level vs {PMAX / 1e6:.1f}M adaptive (its tail is untied).")
tw.append(f"At a 50k vocab the two-level head ties a (narrow-tail) adaptive softmax: {d50as:+.3f} bits/byte (2 seeds). Both "
          f"beat the flat head ({d50flat:+.3f} and {d50asflat:+.3f}), so the gain comes from factorising the softmax.")
tw.append(f"The gain over the flat head shrinks with compute: {gaps[0]:+.3f}, {gaps[1]:+.3f}, {gaps[2]:+.3f} bits/byte at "
          f"0.5×, 1×, 2× the budget (50k, seed 0). Larger budgets are untested. Per token the two-level head is worse "
          f"({eqtok:+.3f} at equal tokens).")
if spd_ok and C:
    tw.append(f"Why it works: at d=384 the flat head is {100 * C['flat_50k']:.0f}% of training FLOPs. The two-level head "
              f"costs {r50:.3f}× as much, so the same budget buys more tokens. On our GPU its training step runs at "
              f"{SPD['50k'][0]:.1f}× the tokens/s of flat ({SPD['131k'][0]:.1f}× at 131k).")
if False:  # dropped 2026-11-14: clusters uncharged in this experiment (their cost is 2.4x its whole flat budget)
    tw.append(f"Knowledge: on synthetic biographies (d=128) at equal FLOPs, the two-level model stores "
              f"{D2['mean_ratio']:.2f}× the fact bits of the flat model ({D2['0']['ratio']:.2f} / {D2['1']['ratio']:.2f} on "
              f"seeds 0 / 1). Neither model is at capacity, so this is bits stored for a fixed budget, not a capacity limit.")
tw.append(f"Caveats: tiny models (≤{int(np.ceil(PMAX / 1e6))}M params), 1–2 seeds, one English dataset, one LR, untuned "
          "adaptive softmax (untied tail). The 131k tokenizer is in-domain, the 50k one is GPT-2's. Old idea: Goodman 2001, "
          "Morin & Bengio 2005, Grave et al. 2017.")
assert 5 <= len(tw) <= 9, len(tw)
# X-style length: a link counts as 23 characters, an emoji as 2
xlen = lambda t: len(t) + t.count("[LINK]") * (23 - 6) + t.count("🧵")  # noqa: E731
for i, t in enumerate(tw):
    assert xlen(t) <= 280, (i, xlen(t), t)
# 2026-11-15 (user): a support reply posted under the last post, not inside the thread; one link (Ko-fi)
KOFI = "https://ko-fi.com/testtm"
REPLY = ("All of this ran on one RTX 5070 Ti at home, which is why the runs stop at 2× the budget. If you'd like to help fund "
         f"longer ones: {KOFI}")
rlen = len(REPLY) - len(KOFI) + 23
assert rlen <= 280, rlen
IMG = {"I": ("hero",), "Tiny-LM": ("hero",), "Method:": ("schematic",), "Vocabulary,": ("headcost",), "The": ("compute",),
       "Why": ("speed", "memory"), "Knowledge:": ("knowledge",)}
with open(os.path.join(OUT, "thread.md"), "w", encoding="utf8", newline="\n") as f:
    f.write("# Thread draft (round 13)\n\nGenerated by `round13_social.py` from the results JSONs. "
            "Character counts in brackets, as X counts them (a link = 23, an emoji = 2); the limit is 280. Replaces the round-12 draft, whose headline (−0.020) used "
            "uncharged clusters, and (2026-11-14) the head-centred draft that experiment 5 overturned.\n\n")
    for i, t in enumerate(tw, 1):
        im = [m for k in IMG.get(t.split()[0], ()) for m in made if m.endswith(f"/{k}_1600x900.png")]
        f.write(f"{i}/{len(tw)} [{xlen(t)} chars]" + (f" — images: {', '.join(im)}" if im else "") + f"\n\n{t}\n\n")
    f.write(f"Reply under post {len(tw)}/{len(tw)} (not part of the thread; no image) [{rlen} chars]\n\n{REPLY}\n\n")
    f.write("Images: " + ", ".join(m for m in made if m.endswith("1600x900.png")) + "\n")
with open(os.path.join(OUT, "README.md"), "w", encoding="utf8", newline="\n") as f:
    f.write("# Social images (round 13)\n\n" + "\n".join(f"- `{m}`" for m in made)
            + "\n\nSkipped (verdict not positive or data missing):\n\n"
            + ("\n".join(f"- {a}: {b}" for a, b in skipped) or "- none") + "\n")
print("made:", *made, sep="\n  ")
print("skipped:", *skipped, sep="\n  ")
