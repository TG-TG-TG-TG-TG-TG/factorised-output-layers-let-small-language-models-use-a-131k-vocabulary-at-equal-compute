"""Paper figures (vector PDF + PNG in paper/figs/) from runs/round11/round11_results.json, runs/round12/round12_results.json
and runs/round13/round13_results.json (2026-11-13: vocab and compute figures use round 13; label fixes).
2026-11-14: the vocab and compute figures add the full-width adaptive softmax (round 13 experiment 5).
Panels whose data are not in the JSON yet are drawn with what exists and marked "pending".

python paper/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
R11 = json.load(open(os.path.join(ROOT, "runs", "round11", "round11_results.json"), encoding="utf8"))
P12 = os.path.join(ROOT, "runs", "round12", "round12_results.json")
R12 = json.load(open(P12, encoding="utf8")) if os.path.exists(P12) else {}
R13 = json.load(open(os.path.join(ROOT, "runs", "round13", "round13_results.json"), encoding="utf8"))
NATS_TO_BPB = 0.311414
LN2 = 0.6931471805599453
ADA = "#1baf7a"
ADAW = "#c8551e"  # full-width adaptive softmax (validated against FLAT/ADA/TWO, all pairs)
INK, MUTED, GRID = "#1f1e1c", "#6f6e69", "#d9d7d0"
FLAT, TWO, SMALLV = "#6f6e69", "#2a78d6", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def rows12():
    return (R12.get("partB") or {}).get("rows") or {}


# ---------------------------------------------------------------- schematic
def schematic(ax, big=False):
    fs = 15 if big else 8.5
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold" if bold else "normal")

    def arrow(a, b, col=MUTED):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14 if big else 9, lw=1.4, color=col))

    box(0.2, 2.5, 1.5, 1.0, "hidden\nstate h", "#f4f3ef", MUTED)
    box(2.4, 1.7, 3.4, 2.6, "", "#eaf2fc", TWO)
    ax.text(4.1, 4.05, "top-level softmax", ha="center", va="center", fontsize=fs, color=TWO, weight="bold")
    box(2.6, 2.85, 3.0, 0.8, "8,192 frequent tokens\n(one logit each)", "white", TWO)
    box(2.6, 1.9, 3.0, 0.8, "4,096 tail clusters\n(one logit each)", "white", TWO)
    box(6.8, 1.5, 3.1, 1.3, "within-cluster softmax\nover the cluster's\nmembers only", "#fdf4e1", SMALLV)
    box(6.8, 3.4, 3.1, 1.0, "frequent token w:\np(w|h) read directly", "white", MUTED)
    arrow((1.7, 3.0), (2.4, 3.0))
    arrow((5.6, 3.25), (6.8, 3.85))
    arrow((5.6, 2.3), (6.8, 2.15), SMALLV)
    ax.text(6.3, 2.0, "p(c|h)", ha="center", va="top", fontsize=fs * 0.9, color=INK, style="italic")  # below the arrow, clear of both boxes
    ax.text(8.35, 0.95, "p(w|h) = p(c(w)|h) · p(w|c(w),h)", ha="center", fontsize=fs, color=INK)


def fig_schematic():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    schematic(ax)
    save(fig, "schematic")


# ---------------------------------------------------------------- frontier
def fig_frontier():
    rows = R11["partA"]["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for fam, col, lab, mk in (("flat", FLAT, "flat tied head", "o"), ("r10", TWO, "two-level head", "s")):
        p = sorted((r["flops"], r["bpb"]) for r in rows if r["d"] == 384 and r["family"] == fam and not r["run"].startswith("wsd_H2atm"))
        p = [q for q in p if "H3rs" not in str(q)]
        if p:
            a.plot(*zip(*p), marker=mk, color=col, lw=1.6, ms=5, label=lab)
    R = rows12()
    for s, mk in (("B384 matched (164M)", "D"), ("B384 matched (315M)", "D")):
        if s in R and R[s]["bpb"] is not None:
            a.plot([R[s]["flops"]], [R[s]["bpb"]], marker=mk, color=FLAT, mfc="white", ms=6, lw=0)
    d1 = ((R12.get("partD") or {}).get("D1") or {})
    if d1.get("1", {}).get("B384m") is not None:
        a.plot([d1["1"]["flops_B"]], [d1["1"]["B384m"]], marker="D", color=FLAT, mfc="white", ms=6, lw=0, label="flat, FLOP-matched (D1)")
    a.set_xscale("log")
    a.set_xlabel("training FLOPs")
    a.set_ylabel("FineWeb-Edu val BPB")
    a.set_title("d = 384", fontsize=9)
    a.legend(frameon=False, fontsize=7)
    a.grid(alpha=0.3, color=GRID)
    for fam, col, lab in (("flat", FLAT, "B (flat)"), ("r10", TWO, "H3k8c4 (two-level)")):
        p = sorted((r["flops"], r["bpb"]) for r in rows if r["d"] == 128 and r["family"] == fam and
                   (r["run"].startswith("wsd_B_r8") or "H3k8c4" in r["run"]))
        if p:
            b.plot(*zip(*p), marker="o" if fam == "flat" else "s", color=col, lw=1.6, ms=4, label=lab)
    b.set_xscale("log")
    b.set_xlabel("training FLOPs")
    b.set_title("d = 128", fontsize=9)
    b.legend(frameon=False, fontsize=7)
    b.grid(alpha=0.3, color=GRID)
    fig.tight_layout()
    save(fig, "frontier")


# ---------------------------------------------------------------- vocabulary
def fig_vocab():
    R = rows12()
    e1, e3, e4 = R13["exp1"]["1x"]["seeds"], R13["exp3"]["50k"]["1x"]["seeds"], R13["exp4"]["bpb"]
    m = lambda xs: sum(xs) / len(xs)  # noqa: E731
    flat = [(4096, R["F4k-384 (700M)"]["bpb"]), (8192, R["F8k-384 (631M)"]["bpb"]),
            (50257, m([e1[s]["flat"]["bpb"] for s in "01"])), (131072, R13["exp3"]["131k"]["1x"]["flat"])]
    ada = [(50257, m([e3[s]["as"]["bpb"] for s in "01"])), (131072, m([e4["a2"][s]["fineweb"] for s in "01"]))]
    ada1 = m([e4["a1"][s]["fineweb"] for s in "01"])
    two = [(50257, m([e1[s]["ch"]["bpb"] for s in "01"])), (131072, m([e4["tl"][s]["fineweb"] for s in "01"]))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.0))
    a1.plot(*zip(*flat), marker="o", color=FLAT, lw=1.6, ms=5, label="flat")
    a1.plot(*zip(*ada), marker="^", color=ADA, lw=1.6, ms=6, label="adaptive softmax (setup 2 at 131k)")
    a1.plot([131072], [ada1], marker="^", mfc="white", color=ADA, ms=6, lw=0, label="adaptive softmax, setup 1")
    e5 = R13["exp5"]["bpb"]
    a1.plot([131072], [m([e5[s]["fineweb"] for s in "01"])], marker="D", color=ADAW, ms=9, mfc="none", mew=1.6, lw=0,
            label="adaptive softmax, full width (131k)")  # hollow: it ties the two-level point
    a1.plot(*zip(*two), marker="s", color=TWO, lw=1.6, ms=5, label="two-level (clusters charged)")
    a1.set_ylabel("FineWeb-Edu val BPB")
    a1.set_ylim(1.14, 1.235)
    # below the axes (2026-11-14 review): inside, it covered the flat curve at 4k
    a1.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=1)
    a1.set_title("BPB at 6.717e16 training FLOPs (d = 384)", fontsize=9)
    sh = {k: R[k]["head_share"] for k in R}
    a2.plot([4096, 8192, 50257, 131072], [sh["F4k-384 (700M)"], sh["F8k-384 (631M)"], sh["B384 matched (315M)"],
                                         sh["F128k-384 (161M)"]], marker="o", color=FLAT, lw=1.6, ms=5, label="flat")
    ga = [json.load(open(os.path.join(ROOT, "runs", "round13", f), encoding="utf8"))[a]["A5"]
          for f, a in (("gates_as.json", "AS384"), ("gates_as_c49.json", "AS128k384c49"))]  # charged head / total fpt
    a2.plot([50257, 131072], [x["head_train"] / x["total"] for x in ga], marker="^",
            color=ADA, lw=1.6, ms=6, label="adaptive softmax (charged by band share)")
    gw = json.load(open(os.path.join(ROOT, "runs", "round13", "gates_as_c49w.json"), encoding="utf8"))["AS128k384c49w"]["A5"]
    a2.plot([131072], [gw["head_train"] / gw["total"]], marker="D", color=ADAW, ms=5.5, lw=0,
            label="adaptive softmax, full width")
    a2.plot([50257, 131072], [sh["H3*-384 (575M)"], sh["H128k-384 (575M)"]], marker="s", color=TWO, lw=1.6, ms=5,
            label="two-level")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylabel("head share")
    a2.set_title("head share of training FLOPs (d = 384)", fontsize=9)
    a2.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=1)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([4096, 8192, 50257, 131072])
        a.set_xticklabels(["4k", "8k", "50k", "131k"])
        a.minorticks_off()
        a.set_xlabel("vocabulary size")
        a.grid(alpha=0.3, color=GRID)
    fig.tight_layout()
    save(fig, "vocab")


# ---------------------------------------------------------------- compute (round 13 experiments 2-4)
def fig_compute():
    X = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}
    e2, e1, e3, e4 = R13["exp2"], R13["exp1"], R13["exp3"]["50k"], R13["exp4"]["bpb"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    a.plot([X[p] for p in ("0.5x", "1x", "2x")], [e2[p]["flat"]["bpb"] for p in ("0.5x", "1x", "2x")], marker="o", color=FLAT,
           lw=1.6, ms=5, label="flat")
    a.plot([X[p] for p in ("0.5x", "1x", "2x")], [e2[p]["ch"]["bpb"] for p in ("0.5x", "1x", "2x")], marker="s", color=TWO,
           lw=1.6, ms=5, label="two-level (clusters charged)")
    a.plot([0.5, 1.0], [e3[p]["seeds"]["0"]["as"]["bpb"] for p in ("0.5x", "1x")], marker="^", color=ADA, lw=1.6, ms=6,
           label="adaptive softmax")
    a.set_ylabel("FineWeb-Edu val BPB")
    a.set_title("50k vocabulary, seed 0", fontsize=9)
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for s, mk in (("0", "s"), ("1", "D")):
        g = [e1[p]["seeds"][s]["d_ch"] for p in ("0.5x", "1x")] + ([e2["2x"]["gap_ch"]] if s == "0" else [])
        b.plot([0.5, 1.0, 2.0][:len(g)], g, marker=mk, color=FLAT, lw=1.4, ms=4,
               label="50k: two-level − flat" if s == "0" else None)
        g = [e3[p]["seeds"][s]["d_tl_as"] for p in ("0.5x", "1x")]
        b.plot([0.5, 1.0], g, marker=mk, color=ADA, lw=1.4, ms=4, label="50k: two-level − adaptive softmax (reduced width)" if s == "0" else None)
        g = [e4["tl05"][s]["fineweb"] - e4["a205"][s]["fineweb"], e4["tl"][s]["fineweb"] - e4["a2"][s]["fineweb"]]
        b.plot([0.5, 1.0], g, marker=mk, color=TWO, lw=1.4, ms=4,
               label="131k: two-level − adaptive softmax (setup 2)" if s == "0" else None)
        g = [e4["tl05"][s]["fineweb"] - R13["exp5"]["bpb_05"][s], e4["tl"][s]["fineweb"] - R13["exp5"]["bpb"][s]["fineweb"]]
        b.plot([0.5, 1.0], g, marker=mk, color=ADAW, lw=1.4, ms=4,
               label="131k: two-level − adaptive softmax (full width)" if s == "0" else None)
    b.axhline(0, color=MUTED, lw=0.8)
    b.axhspan(-0.003, 0.003, color=GRID, alpha=0.5, lw=0)
    b.set_ylabel("Δ BPB (negative: two-level better)")
    b.set_title("differences (squares seed 0, diamonds seed 1)", fontsize=9)
    b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2])
        ax.set_xticklabels(["0.5×", "1×", "2×"])
        ax.minorticks_off()
        ax.set_xlabel("training FLOPs (1× = 6.717e16)")
        ax.grid(alpha=0.3, color=GRID)
    fig.tight_layout()
    save(fig, "compute")


# ---------------------------------------------------------------- cost model
def head_share(d, V, n=6, T=1024, two=False):
    blocks, attn = 6 * n * 12 * d * d, 6 * n * T * d // 2 * 2
    L = -(-(V - 8192) // 4096)
    head = 6 * d * (8192 + 4096 + L) if two else 6 * d * V
    return head / (blocks + attn + head)


def fig_cost():
    import numpy as np
    ds = np.array([64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for V, ls in ((50257, "-"), (131072, "--")):
        ax.plot(ds, [head_share(int(x), V) for x in ds], ls, color=FLAT, lw=1.6, label=f"flat, V = {V // 1000}k")
        ax.plot(ds, [head_share(int(x), V, two=True) for x in ds], ls, color=TWO, lw=1.6,
                label=f"two-level, V = {V // 1000}k" + (" (on top of 50k)" if V > 60000 else ""))
    ax.set_xscale("log")
    ax.set_xlabel("model width d (6 blocks, f = 4d, T = 1024)")
    ax.set_ylabel("head share of training FLOPs")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3, color=GRID)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)  # below the axes: inside, it covered the curves
    save(fig, "cost")


# ---------------------------------------------------------------- buckets + Jensen
def fig_buckets():
    names = ["[0,256)", "[256,1k)", "[1k,4k)", "[4k,16k)", "[16k,V)"]
    bd = R11["E1"]["bucket_deltas"]
    J = ((R12.get("P5") or {}).get("jensen")) or {}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = range(5)
    for key, col, lab, off in (("H3k8c4-B_r8:fineweb", TWO, "d=128, equal tokens", -0.2),
                               ("H3s384_300M-B384:fineweb", "#1baf7a", "d=384, equal tokens", 0.0),
                               ("H3s384_575M-B384:fineweb", SMALLV, "d=384, equal FLOPs", 0.2)):
        a.bar([i + off for i in x], [v / LN2 for v in bd[key]], width=0.2, color=col, label=lab)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xticks(list(x))
    a.set_xticklabels(names, fontsize=7)
    a.set_xlabel("training-frequency rank of the target")
    a.set_ylabel("Δ loss, two-level − flat (bits/token)")
    a.legend(frameon=False, fontsize=6.5)
    if J:
        for lab, col, off, nm in (("H3k8c4", TWO, -0.15, "d=128"), ("H3s384_575M", SMALLV, 0.15, "d=384")):
            if lab in J:
                bk = J[lab]["val"]["buckets"]
                ys = [bk[k]["delta_mean"] / LN2 if bk[k]["delta_mean"] is not None else 0 for k in bk]
                b.bar([i + off for i in range(len(ys))], ys, width=0.3, color=col, label=nm)
        b.set_xticks(range(5))
        b.set_xticklabels(names, fontsize=7)
        b.legend(frameon=False, fontsize=6.5)
    else:
        b.text(0.5, 0.5, "pending (round12_jensen.py)", transform=b.transAxes, ha="center", color=MUTED)
    b.set_xlabel("training-frequency rank of the target")
    b.set_ylabel("mean Jensen gap Δ_c(h) (bits)")
    fig.tight_layout()
    save(fig, "buckets")


# ---------------------------------------------------------------- knowledge
def fig_knowledge():
    E4 = R11["E4"]["rows"]
    D2 = ((R12.get("partD") or {}).get("D2")) or {}
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    labs = [("bioB_100000", "flat\n100 exp.", FLAT), ("bioH_100000", "two-level\n100 exp.", TWO),
            ("bioH_100000_eqF", "two-level\nequal FLOPs\n(237 exp.)", SMALLV)]
    for i, (k, nm, col) in enumerate(labs):
        y0 = E4[k]["bits"]
        ax.bar(i, y0 / 1e3, color=col, width=0.6)
        ax.text(i, y0 / 1e3 + 15, f"{y0 / 1e3:.0f}k", ha="center", fontsize=7)
    s1 = D2.get("1") or {}
    if s1.get("bits_B") is not None:
        ax.plot([0], [s1["bits_B"] / 1e3], "k_", ms=14)
    if s1.get("bits_H_eqF") is not None:
        ax.plot([2], [s1["bits_H_eqF"] / 1e3], "k_", ms=14)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l[1] for l in labs], fontsize=7)
    ax.set_ylabel("knowledge bits stored (thousands)")
    ax.set_title("synthetic biographies, N = 100,000, d = 128", fontsize=8)
    save(fig, "knowledge")


if __name__ == "__main__":
    fig_schematic()
    fig_frontier()
    fig_vocab()
    fig_compute()
    fig_cost()
    fig_buckets()
    fig_knowledge()
