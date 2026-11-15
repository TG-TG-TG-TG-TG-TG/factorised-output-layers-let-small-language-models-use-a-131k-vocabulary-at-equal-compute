"""Round 11 social charts (only for positive verdicts): Part B / E1 (two-level beats flat and an 8k-vocab flat head at equal
FLOPs) and E4 (no knowledge penalty; more tokens -> more knowledge). Numbers from runs/round11/round11_results.json and the
run logs. python round11_social.py -> runs/round11/social/*.png"""
import json
import os

from round9_analyze import BPN, load_run
from round10_plots import _plt, social

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round11")
GREY, BLUE, RED, AQUA = "#6f6e69", "#2a78d6", "#e34948", "#1baf7a"
res = json.load(open(os.path.join(OUT, "round11_results.json")))
B = res["partB"]["rows"]

# --- chart 1: BPB vs training FLOPs, d=384 (B384 branches from round 9's 300M run, same shape as the round-11 re-run)
b9 = load_run(os.path.join(HERE, "runs", "round9", "wsd_B384_s0.jsonl"))
fpt = b9["final"]["flops_per_token"]["total"]
bx = [fpt * v["tokens"] / 1e16 for k, v in sorted(b9["branch"].items())]
by = [v["loss"] * BPN for k, v in sorted(b9["branch"].items())]
series = [
    ("flat head, 50k vocab", bx, by, GREY, False),
    ("flat head, 8k vocab", [B[k]["flops"] / 1e16 for k in ("F8k-384 (329M)", "F8k-384 (631M)")],
     [B[k]["bpb"] for k in ("F8k-384 (329M)", "F8k-384 (631M)")], RED, False),
    ("two-level head, 50k vocab", [B[k]["flops"] / 1e16 for k in ("H3*-384 (300M)", "H3*-384 (575M)")],
     [B[k]["bpb"] for k in ("H3*-384 (300M)", "H3*-384 (575M)")], BLUE, True),
]
d_flat = B["H3*-384 (575M)"]["bpb"] - B["B384 (300M)"]["bpb"]
d_8k = B["H3*-384 (575M)"]["bpb"] - B["F8k-384 (631M)"]["bpb"]
files = social("bpb_vs_flops", series, "training compute (10¹⁶ FLOPs)",
               "Same compute, fewer bits per byte:\ntwo-level head beats full softmax\nand an 8k vocabulary",
               f"Same d=384 GPT backbone on FineWeb-Edu, one RTX 5070 Ti; the vocabularies differ (GPT-2 50k vs our 8k BPE), so the y axis is "
               f"bits per byte of the same 23.2 MB of validation text. At {B['H3*-384 (575M)']['flops'] / 1e16:.1f}e16 FLOPs: two-level {B['H3*-384 (575M)']['bpb']:.3f}, "
               f"8k vocab {B['F8k-384 (631M)']['bpb']:.3f} ({d_8k:+.3f}); the flat 50k head's 300M-token run used 5% fewer FLOPs ({d_flat:+.3f}). "
               "One seed per arm.", OUT, xlog=False, ylabel="validation bits per byte")

# --- chart 2: E4 knowledge bits (bars)
E = res["E4"]["rows"]
plt = _plt()
groups = [("20k people\n100 exposures", [("flat", "bioB_20000", GREY), ("two-level", "bioH_20000", BLUE)]),
          ("100k people\n100 exposures", [("flat", "bioB_100000", GREY), ("two-level", "bioH_100000", BLUE)]),
          ("100k people\nsame FLOPs as flat", [("flat", "bioB_100000", GREY), ("two-level\n(237 exposures)", "bioH_100000_eqF", AQUA)])]
import textwrap
for tag, size, rect, tfs, wrap in (("1600x900", (16, 9), [0.09, 0.25, 0.86, 0.50], 30, 138),
                                   ("1080x1080", (10.8, 10.8), [0.13, 0.27, 0.83, 0.47], 29, 98)):
    fig = plt.figure(figsize=size, dpi=100)
    ax = fig.add_axes(rect)
    xt, xl = [], []
    for gi, (gname, bars) in enumerate(groups):
        for bi, (lab, key, c) in enumerate(bars):
            x = gi * 3 + bi
            v = E[key]["bits"] / 1e3
            ax.bar(x, v, 0.8, color=c, edgecolor="white", linewidth=2)
            ax.annotate(f"{v:,.0f}k", (x, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=17, color="#222")
            ax.annotate(lab, (x, 0), xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=14, color="#444")
        xt.append(gi * 3 + 0.5)
        xl.append(gname)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=16)
    ax.tick_params(axis="x", pad=48, length=0)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("knowledge stored (thousand bits)", fontsize=20)
    ax.set_ylim(0, max(E[k]["bits"] for k in E) / 1e3 * 1.15)
    ax.grid(axis="y", alpha=0.25)
    r20 = E["bioH_20000"]["bits"] / E["bioB_20000"]["bits"]
    r100 = E["bioH_100000"]["bits"] / E["bioB_100000"]["bits"]
    req = E["bioH_100000_eqF"]["bits"] / E["bioB_100000"]["bits"]
    fig.text(0.04, 0.96, "A cheaper output head doesn't cost\nmemorised facts; spending the savings\non more passes stores more of them",
             fontsize=tfs, fontweight="bold", va="top", ha="left", linespacing=1.15)
    foot = (f"Tiny GPT (d=128) trained only on synthetic biographies (5 facts each, 46.5 bits/person). Bits = sum of max(0, log2|A| − NLL) "
            f"over facts. Two-level/flat: {r20:.2f}× at 20k people, {r100:.2f}× at 100k; at equal training FLOPs the two-level head sees 2.4× the "
            f"passes and stores {req:.2f}× the bits. One seed per arm.")
    fig.text(0.04, 0.025, textwrap.fill(foot, wrap), fontsize=14, va="bottom", ha="left", color="#444", linespacing=1.35)
    p = os.path.join(OUT, "social", f"knowledge_bits_{tag}.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    files.append(os.path.relpath(p, OUT).replace(os.sep, "/"))
print(files)
