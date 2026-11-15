"""Round 11 Part D (optional, pinned rule): H3s512 at 5,484 stable steps (359,399,424 tokens), or H3rs<best M>_512 if Part C at
d=128 is 'partial' or better (best tH3rs arm beats H3k8c4 by more than 0.010 at 1200M). Smoke first if not smoked."""
import json
import os
import subprocess
import sys

from round9_analyze import load_run

OUT = os.path.join("runs", "round11")
h = load_run(os.path.join("runs", "round10", "wsd_H3k8c4_s0.jsonl"))
base = h["branch"][max(h["branch"])]["loss"]
got = {}
for M in (16, 64):
    r = load_run(os.path.join(OUT, f"wsd_H3rs{M}_s0.jsonl"))
    if r:
        got[M] = r["branch"][max(r["branch"])]["loss"]
arm = "H3s512"
if got:
    M = min(got, key=got.get)
    if got[M] < base - 0.010:
        arm = f"H3rs{M}_512"
json.dump({"tH3rs_1200M": got, "H3k8c4_1200M": base, "arm": arm}, open(os.path.join(OUT, "partD_choice.json"), "w"), indent=1)
print(f"[partD] arm {arm}", flush=True)
if not os.path.exists(os.path.join(OUT, "smoke", f"smoke_{arm}.jsonl")):
    subprocess.call(["bash", "r11smoke.sh", f"{arm}@359399424"])
sys.exit(subprocess.call(["bash", "r11one.sh", "d384", f"{arm}:0", "--label", arm, "--tokens", "359399424", "--branch_at", "359399424"]))
