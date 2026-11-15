"""Round 11 Part C, pinned rule (round11_analyze.py docstring): run tH3rs64 only if tH3rs16 beats H3k8c4 (wsd_H3k8c4_s0,
4.1552) by more than 0.010 at 1200M. Writes runs/round11/partC_rule.json; launches r11one.sh tiny tH3rs64:0 if it fires."""
import json
import os
import subprocess
import sys

from round9_analyze import load_run

OUT = os.path.join("runs", "round11")
r16 = load_run(os.path.join(OUT, "wsd_H3rs16_s0.jsonl"))
h = load_run(os.path.join("runs", "round10", "wsd_H3k8c4_s0.jsonl"))
k = max(h["branch"])
base = h["branch"][k]["loss"]
if r16 is None:
    print("[partC] tH3rs16 not DONE; nothing decided", flush=True)
    sys.exit(1)
l16 = r16["branch"][max(r16["branch"])]["loss"]
fire = l16 < base - 0.010
json.dump({"H3k8c4_1200M": base, "tH3rs16_1200M": l16, "delta": l16 - base, "run_M64": fire}, open(os.path.join(OUT, "partC_rule.json"), "w"), indent=1)
print(f"[partC] tH3rs16 {l16:.4f} vs H3k8c4 {base:.4f} (Δ {l16 - base:+.4f}); M=64 {'RUNS' if fire else 'not run'}", flush=True)
if fire:
    sys.exit(subprocess.call(["bash", "r11one.sh", "tiny", "tH3rs64:0"]))
