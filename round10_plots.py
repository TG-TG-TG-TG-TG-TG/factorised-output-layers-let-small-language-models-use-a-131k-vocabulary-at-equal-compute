"""Round-10 figures (called by round10_analyze.py; Windows Python, matplotlib).

Palette (dataviz validate_palette.js, light, --pairs all): blue #2a78d6 / yellow #eda100 / aqua #1baf7a / red #e34948
pass the normal-vision floor (worst 20.8); the worst CVD pair is red vs aqua at 6.9 (deutan), inside the 6-8 band that
is legal only with secondary encoding, so every arm also has its own marker and a direct label. Larger sets
(+ magenta, brown, violet, orange or green) all FAIL the normal-vision floor, so a figure never shows more than
four coloured arms plus grey. B (and B-family baselines) are grey.
In the headline the colour shows the Part (A yellow, C aqua, D red), the best arm is blue, B is grey; the arm is
named by a direct label and a marker.
"""
import math
import os

import numpy as np

GREY, BLUE, YEL, AQUA, RED = "#6f6e69", "#2a78d6", "#eda100", "#1baf7a", "#e34948"
FOUR = [BLUE, YEL, AQUA, RED]
MARK = {"B": "o", "H2atm": "s", "H3k8": "^", "H3c4": "v", "H3k8c4": "D", "Rloop": "P", "Rdeep": "X", "Rwide": "h", "Rwide1200": "h",
        "RB24": "o", "D1": "*", "D1_lazy": "*", "D2": "p", "B384": "o", "H2atm384": "s", "H3s384": "^", "tCgcyc": "<"}
PART = {"H2atm": "A", "H3k8": "A", "H3c4": "A", "H3k8c4": "A", "Rloop": "C", "Rdeep": "C", "Rwide": "C", "Rwide1200": "C",
        "D1": "D", "D1_lazy": "D", "D2": "D", "H3s384": "A", "H2atm384": "A"}
PARTCOL = {"A": YEL, "C": AQUA, "D": RED}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    return plt


def pts(r, key):
    ks = sorted(r["branch"])
    return [(r["branch"][k][key], r["branch"][k]["loss"]) for k in ks]


def _facet(ax, RS, labels, B, nice, delta, title):
    """one facet: B grey + up to 4 coloured arms (fixed order, colour per entity within the facet)."""
    cols = {}
    i = 0
    for l in labels:
        if l not in RS:
            continue
        if l in ("B", "B384", "RB24"):
            cols[l] = GREY
        elif l.endswith("_lazy") and l[:-5] in cols:   # lazy-update variant: same colour as its dense run, dashed
            cols[l] = cols[l[:-5]]
        else:
            cols[l] = FOUR[i % 4]
            i += 1
    for l in labels:
        if l not in RS:
            continue
        r = RS[l]
        ks = sorted(r["branch"])
        if delta:
            ks = [k for k in ks if any(abs(k - kb) <= 0.01 * k for kb in B["branch"])]
            kb = {k: min(B["branch"], key=lambda q: abs(q - k)) for k in ks}
            ys = [r["branch"][k]["loss"] - B["branch"][kb[k]]["loss"] for k in ks]
        else:
            ys = [r["branch"][k]["loss"] for k in ks]
        if not ks or (delta and l in ("B", "B384")):
            continue
        xs = [r["branch"][k]["tokens"] for k in ks]
        ls = "--" if (l == "RB24" or l.endswith("_lazy")) else "-"
        ax.plot(xs, ys, ls, marker=MARK.get(l, "o"), color=cols[l], lw=2.2 if l in ("B", "B384") else 1.6, ms=6,
                label=nice.get(l, l), mec="white", mew=0.8, zorder=3)
        ax.annotate(nice.get(l, l), (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                    color="#333", va="center")
    if delta:
        ax.axhline(0, color=GREY, lw=1.2)
        ax.text(0.01, 0.02, "0 = B at the same tokens", transform=ax.transAxes, fontsize=8.5, color="#555")
    ax.set_xscale("log")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("training tokens")
    ax.grid(alpha=0.22)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 * 1.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right" if not delta else "best")


def facets(RS, nice, h3, out, R10):
    plt = _plt()
    B, B384 = RS["B"], RS.get("B384")
    spec = [("Part A: tail fixes (d=128)", ["B", "H2atm", "H3k8", "H3c4", "H3k8c4"], B),
            ("Part C: backbone reinvestment (d=128, 600M" + ("; R-wide to 1200M)" if "Rwide1200" in RS else ")"),
             ["B", h3, "Rloop", "Rdeep", "Rwide1200" if "Rwide1200" in RS else "Rwide", "RB24"] if h3 else ["B"], B),
            ("Part D: hashed n-gram inputs (d=128)", ["B", h3, "D1", "D1_lazy", "D2", "D2_lazy"] if h3 else ["B"], B),
            ("Part B: d=384", ["B384", "H2atm384", "H3s384"], B384)]
    spec = [s for s in spec if s[2] is not None and sum(l in RS for l in s[1]) > 1]
    for delta, fname, yl in ((False, "loss_vs_tokens.png", "val loss (nats/token)"),
                             (True, "delta_vs_B.png", "Δ val loss vs B (d=384: vs B384), nats/token")):
        fig, axs = plt.subplots(1, len(spec), figsize=(5.2 * len(spec), 4.8), squeeze=False)
        for ax, (title, labs, base) in zip(axs[0], spec):
            labs = [l for i, l in enumerate(labs) if l not in labs[:i]]
            _facet(ax, RS, labs, base, nice, delta, title)
        axs[0][0].set_ylabel(yl)
        fig.tight_layout()
        fig.savefig(os.path.join(R10, fname), dpi=130)
        plt.close(fig)
        out.append(fname)


def buckets(BK, BD, h3, nice, out, R10):
    plt = _plt()
    if BK:
        labs = list(BK["arms"])
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        x = np.arange(5)
        w = 0.8 / len(labs)
        for ax, key, title in zip(axs, ("d_total_bucket", "d_top_bucket", "d_within_bucket"),
                                  ("Δ total vs B", "Δ top level (slot of the target)", "Δ within cluster")):
            for i, l in enumerate(labs):
                ax.bar(x - 0.4 + w * (i + 0.5), BK["arms"][l][key], w * 0.9, color=FOUR[i % 4],
                       label=nice.get(l, l) + (" = H3*" if l == h3 else ""), edgecolor="white", linewidth=1)
            ax.axhline(0, color=GREY, lw=1)
            ax.set_xticks(x, [f"{b}\n{s:.0%} of targets" for b, s in zip(BK["buckets"], BK["bucket_share"])], fontsize=8.5)
            ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
            ax.set_xlabel("target frequency-rank bucket")
            ax.grid(alpha=0.22, axis="y")
        axs[0].set_ylabel("arm − B_r8 (nats/token, bucket mean)")
        axs[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(R10, "buckets_partA.png"), dpi=130)
        plt.close(fig)
        out.append("buckets_partA.png")
    if BD and h3 and h3 in BD["arms"]:
        hb = BD["arms"][h3]
        labs = [l for l in BD["arms"] if l != h3]
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        x = np.arange(5)
        w = 0.8 / max(1, len(labs))
        for ax, key, title in zip(axs, ("d_total_bucket", "d_top_bucket", "d_within_bucket"),
                                  ("Δ total vs H3*", "Δ top level vs H3*", "Δ within cluster vs H3*")):
            for i, l in enumerate(labs):
                ax.bar(x - 0.4 + w * (i + 0.5), [a - b for a, b in zip(BD["arms"][l][key], hb[key])], w * 0.9,
                       color=[RED, AQUA, YEL, BLUE][i % 4], label=nice.get(l, l), edgecolor="white", linewidth=1)
            ax.axhline(0, color=GREY, lw=1)
            ax.set_xticks(x, [f"{b}\n{s:.0%} of targets" for b, s in zip(BD["buckets"], BD["bucket_share"])], fontsize=8.5)
            ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
            ax.set_xlabel("target frequency-rank bucket")
            ax.grid(alpha=0.22, axis="y")
        axs[0].set_ylabel(f"arm − {nice.get(h3, h3)} (nats/token, bucket mean)")
        axs[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(R10, "buckets_partD.png"), dpi=130)
        plt.close(fig)
        out.append("buckets_partD.png")


def _deoverlap(items, gap):
    items.sort(key=lambda t: t[1])
    for a, b in zip(items, items[1:]):
        if b[2] - a[2] < gap:
            b[2] = a[2] + gap
    return items


def headline(RS, AC, ARM_OF, nice, best, BPN, cap, R10, out):
    """(a) loss vs training FLOPs, (b) loss vs wall-clock, (c) loss vs total params; shared y; BPB on the right.
    Every arm's final branch is annotated with Δ vs B at the same tokens (d=384 arms: vs B384 at 300M)."""
    plt = _plt()
    ORDER = ["B", "H2atm", "H3k8", "H3c4", "H3k8c4", "Rloop", "Rdeep", "Rwide", "Rwide1200", "RB24", "D1", "D1_lazy", "D2", "D2_lazy",
             "B384", "H3s384"]
    arms = [l for l in ORDER if l in RS and ARM_OF[l] in AC and not (l == "Rwide" and "Rwide1200" in RS)]
    B, B384 = RS["B"], RS.get("B384")

    def base(l):
        return B384 if l in ("B384", "H3s384") else B

    def col(l):
        if l in ("B", "B384", "RB24"):
            return GREY
        if l == best:
            return BLUE
        return PARTCOL.get(PART.get(l), "#999999")

    def dB(l):
        r = RS[l]
        k = max(r["branch"])
        bb = base(l)
        if l == "H3s384":
            kb = min(bb["branch"], key=lambda q: abs(q - 300e6))
        else:
            kb = min(bb["branch"], key=lambda q: abs(q - k))
            if abs(kb - k) > 0.01 * k:
                return None, k
        return r["branch"][k]["loss"] - bb["branch"][kb]["loss"], k

    fig, axs = plt.subplots(1, 3, figsize=(19, 7.0), sharey=True)
    ally = [p[1] for l in arms for p in pts(RS[l], "tokens")]
    yr = max(ally) - min(ally)
    for pi, ax in enumerate(axs):
        lab = []
        for l in arms:
            r = RS[l]
            fpt = AC[ARM_OF[l]]["flops_per_token"]["total"]
            strong = l in ("B", "B384", best)
            if pi == 0:
                xy = [(t * fpt, y) for t, y in pts(r, "tokens")]
            elif pi == 1:
                xy = [(w / 60, y) for w, y in pts(r, "wall")]
            else:
                xy = [(AC[ARM_OF[l]]["total"], pts(r, "tokens")[-1][1])]
            ls = "--" if l in ("RB24", "B384") else "-"
            ax.plot([p[0] for p in xy], [p[1] for p in xy], ls, marker=MARK.get(l, "o"), color=col(l),
                    lw=2.6 if strong else 1.3, ms=8 if strong else 6, alpha=1 if strong else 0.85, mec="white", mew=0.9,
                    zorder=4 if strong else 3, label=nice.get(l, l))
            if pi == 2:
                ng = AC[ARM_OF[l]].get("ngram", 0)
                if ng:
                    xd = AC[ARM_OF[l]]["total"] - ng
                    ax.plot([xd, xy[0][0]], [xy[0][1]] * 2, ":", color=col(l), lw=1.2, zorder=2)
                    ax.plot([xd], [xy[0][1]], MARK.get(l, "o"), mfc="white", mec=col(l), mew=1.6, ms=8, zorder=3)
            d, k = dB(l)
            hz = f" {k / 1e6:.0f}M"
            txt = (f"{nice.get(l, l)}{hz}  " + ("ref." if l in ("B", "B384") else ("n/a" if d is None else f"{d:+.3f}")))
            lab.append([l, xy[-1][1], xy[-1][1], xy[-1][0], txt])
        # labels in one column at the panel's right (axes-fraction x), leader lines to each final point
        xs_all = [q for line in ax.get_lines() for q in line.get_xdata()]
        lo, hi = min(xs_all), max(xs_all)
        span = math.log(hi / lo)
        ax.set_xscale("log")
        ax.set_xlim(lo / math.exp(0.04 * span), math.exp(math.log(hi) + 0.75 * span))   # data ends at ~0.58 of the axis
        for l, y0, y, x, txt in _deoverlap(lab, 0.030 * yr + 0.004):
            ax.annotate(txt, (x, y0), xytext=(0.64, y), textcoords=("axes fraction", "data"), fontsize=8.6,
                        color="#222", va="center", fontweight="bold" if l == best else "normal",
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6, shrinkA=0, shrinkB=3),
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8), zorder=6)
        ax.grid(alpha=0.22)
    axs[0].set_title("(a) loss vs training FLOPs", loc="left", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("training FLOPs (6N rule + attention; head = 3 × its forward)")
    axs[1].set_title("(b) loss vs wall-clock, RTX 5070 Ti", loc="left", fontsize=13, fontweight="bold")
    axs[1].set_xlabel("training wall-clock (minutes, own schedule)")
    axs[2].set_title("(c) final loss vs total parameters", loc="left", fontsize=13, fontweight="bold")
    axs[2].set_xlabel("total parameters (filled); open marker = without n-gram lookup tables")
    axs[0].set_ylabel("val loss (nats/token)")
    sec = axs[2].secondary_yaxis("right", functions=(lambda v: v * BPN, lambda v: v / BPN))
    sec.set_ylabel("bits per byte (same scale)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=GREY, lw=2.6, marker="o", label="B (flat softmax) / B384 (dashed)"),
          Line2D([], [], color=BLUE, lw=2.6, marker=MARK.get(best, "o"), label=f"best arm: {nice.get(best, best)}"),
          Line2D([], [], color=YEL, lw=1.3, label="Part A / B: two-level heads"),
          Line2D([], [], color=AQUA, lw=1.3, label="Part C: deeper / wider / looped backbone"),
          Line2D([], [], color=RED, lw=1.3, label="Part D: hashed n-gram inputs")]
    axs[0].legend(handles=hs, frameon=False, fontsize=9, loc="lower left")
    fig.text(0.01, -0.02, cap, fontsize=11.5, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(R10, "round10_headline.png"), dpi=110, bbox_inches="tight")
    fig.savefig(os.path.join(R10, "round10_headline.svg"), bbox_inches="tight")
    plt.close(fig)
    out += ["round10_headline.png", "round10_headline.svg"]


def social(name, series, xlabel, title, footer, R10, xlog=True, ylabel="validation loss (nats/token)", xzero=False, note=None):
    """series: [(label, xs, ys, colour, strong)]. 1600x900 and 1080x1080 at dpi 100 (px = pt x 100/72):
    title 30 pt = 41.7 px, axis labels 21 pt = 29.2 px. Direct labels, no legend. y range = every plotted point plus
    20% padding (nothing cropped); xzero: x axis starts at 0 (wall-clock). note = (x, y_from, y_to, text): a
    measured Δ drawn as a vertical bracket. title / footer: explicit line breaks."""
    plt = _plt()
    from matplotlib.ticker import ScalarFormatter
    d = os.path.join(R10, "social")
    os.makedirs(d, exist_ok=True)
    files = []
    import textwrap
    for tag, size, rect, tfs, wrap in (("1600x900", (16, 9), [0.09, 0.25, 0.68, 0.52], 30, 138),
                                       ("1080x1080", (10.8, 10.8), [0.14, 0.27, 0.56, 0.47], 29, 98)):
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_axes(rect)
        for lab, xs, ys, c, strong in series:   # baseline labelled right of its end, the strong line below-left of its end
            ax.plot(xs, ys, "-o", color=c, lw=4 if strong else 3, ms=10, mec="white", mew=1.5, zorder=3)
            ax.annotate(lab, (xs[-1], ys[-1]), xytext=(-6, -34) if strong else (14, 0), textcoords="offset points", fontsize=20,
                        color=c, va="center", ha="right" if strong else "left", fontweight="bold", annotation_clip=False)
        if note:
            x, y0, y1, txt = note
            ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6))
            ax.annotate(txt, (x, (y0 + y1) / 2), xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=18, color="#222")
        if xlog:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
        if xzero:
            ax.set_xlim(0, max(x for s_ in series for x in s_[1]) * 1.05)
        ally = [y for s_ in series for y in s_[2]]
        pad = 0.20 * (max(ally) - min(ally))
        ax.set_ylim(min(ally) - pad, max(ally) + pad)
        ax.set_xlabel(xlabel, fontsize=21)
        ax.set_ylabel(ylabel, fontsize=21)
        ax.tick_params(labelsize=17)
        ax.grid(alpha=0.25)
        fig.text(0.04, 0.96, title, fontsize=tfs, fontweight="bold", va="top", ha="left", linespacing=1.15)
        foot = chr(10).join(textwrap.fill(line, wrap) for line in footer.split(chr(10)))
        fig.text(0.04, 0.025, foot, fontsize=14, va="bottom", ha="left", color="#444", linespacing=1.35)
        p = os.path.join(d, f"{name}_{tag}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        files.append(os.path.relpath(p, R10).replace(os.sep, "/"))
    return files
