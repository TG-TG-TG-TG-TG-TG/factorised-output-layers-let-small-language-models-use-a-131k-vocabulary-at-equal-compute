"""Round 11 Part C at d=384 (pinned rule): best M = the lower 1200M loss among the tH3rs arms run; run H3rs<M>_384 to 575M with
branches 300M / 575M (smoke first if that arm was not smoked). Writes runs/round11/partC_bestM.json."""
import json
import os
import subprocess
import sys

from round9_analyze import load_run

OUT = os.path.join("runs", "round11")
got = {}
for M in (16, 64):
    r = load_run(os.path.join(OUT, f"wsd_H3rs{M}_s0.jsonl"))
    if r:
        got[M] = r["branch"][max(r["branch"])]["loss"]
if not got:
    print("[partC384] no tH3rs arm DONE", flush=True)
    sys.exit(1)
M = min(got, key=got.get)
json.dump({"loss_1200M": got, "best_M": M}, open(os.path.join(OUT, "partC_bestM.json"), "w"), indent=1)
print(f"[partC384] 1200M losses {got}; best M = {M}", flush=True)
if not os.path.exists(os.path.join(OUT, "smoke", f"smoke_H3rs{M}_384.jsonl")):
    subprocess.call(["bash", "r11smoke.sh", f"H3rs{M}_384"])
sys.exit(subprocess.call(["bash", "r11one.sh", "d384", f"H3rs{M}_384:0", "--label", f"H3rs{M}_384", "--tokens", "575e6", "--branch_at", "300e6,575e6"]))
