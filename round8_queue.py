"""Round-8 queue (2026-11-06): LM-head experiments on the round-5 tiny proxy (d=128).
Logs, DONE markers, the GPU-time ledger and every new output go to runs/round8/.

Recipe = round-5 part A (tB): WSD to 1200M stable tokens, branches 75M/150M/300M/600M/1200M,
LR 2e-3, 10% decay branches, n_evals 12. Final params kept (--keep_params) in ~/lm-data/ckpt
under the round-8 run name (wsd_<label>_s<seed>), so nothing older is overwritten.

python round8_queue.py --runs B_r8:0            # the round-8 rerun of tB (baseline; tB had no checkpoint)
python round8_queue.py --runs H2:0,H2r:0,CB:0
python round8_queue.py --runs H2:0 --tokens 600e6   # cut horizon (branches stop at 600M)
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round8")
LEDGER = os.path.join(OUT, "budget.json")
LR = 2e-3
BRANCHES = [75e6, 150e6, 300e6, 600e6, 1200e6]
LABEL_ARM = {"B_r8": "tB", "H2": "tH2", "H2r": "tH2r", "CB": "tCB", "H2w": "tH2w"}


def done(name):
    return os.path.exists(os.path.join(OUT, name + ".DONE"))


def run(label, seed, tokens, extra):
    name = f"wsd_{label}_s{seed}" + ("" if tokens >= 1200e6 else f"_{tokens / 1e6:.0f}M")
    if done(name):
        print(f"[r8] {name} is DONE; skipped", flush=True)
        return
    br = ",".join(f"{b:.0f}" for b in BRANCHES if b <= tokens)
    cmd = [sys.executable, "train.py", "--arm", LABEL_ARM[label], "--seed", str(seed), "--lr", str(LR),
           "--tokens", f"{tokens:.0f}", "--sched", "wsd", "--branch_at", br, "--n_evals", "12",
           "--name", name, "--runs_dir", OUT, "--keep_params"] + extra
    print(f"[r8] start {name}: {' '.join(cmd[1:])}", flush=True)
    t = time.time()
    with open(os.path.join(OUT, name + ".out"), "w") as out:
        rc = subprocess.call(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    dt = time.time() - t
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"used_s": 0.0, "runs": {}}
    L["used_s"] += dt
    L["runs"][name] = {"s": dt, "rc": rc}
    json.dump(L, open(LEDGER + ".tmp", "w"), indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)
    print(f"[r8] end {name}: rc {rc}, {dt / 60:.1f} min, round-8 total {L['used_s'] / 3600:.2f} h", flush=True)
    if rc != 0 or not done(name):
        print(f"[r8] {name} failed (rc {rc}); stopping so it can be looked at", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tokens", type=float, default=1200e6)
    ap.add_argument("--extra", default="", help="extra train.py args, space separated")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for x in a.runs.split(","):
        lab, s = x.split(":")
        run(lab, int(s), a.tokens, a.extra.split())
    print("[r8] finished", flush=True)
