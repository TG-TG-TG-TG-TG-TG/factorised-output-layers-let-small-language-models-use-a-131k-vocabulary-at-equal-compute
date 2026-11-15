"""Round-9 queue (2026-11-07): why the two-level head falls behind, and fixes. Logs, DONE markers,
the GPU-time ledger and every new output go to runs/round9/. Old runs are never touched.

Recipes (all LR 2e-3, WSD, warmup 3M tokens, n_evals 12, same data order as every earlier round):
  tiny (round-5 part A / round 8): stable to 1200M, branches 75M/150M/300M/600M/1200M, 10% decay
  C    (decay ablation):           stable to 1200M, branches 300M/1200M only, 20% decay (--decay_frac 0.2)
  D    (matched compute):          stable to 4800M, branches 1200M/2400M/4800M, 10% decay
  d384 (round-5 part B shape):     stable to 300M, branches 150M/300M, 10% decay

python round9_queue.py --recipe d384 --runs B384:0
python round9_queue.py --recipe tiny --runs H2tm:0,H2a:0,H2atm:0
python round9_queue.py --recipe C --runs H2a:0         # -> wsd_H2a_dec2_s0
python round9_queue.py --recipe D --runs H2a:0         # -> wsd_H2a_4800M_s0
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round9")
LEDGER = os.path.join(OUT, "budget.json")
LR = 2e-3
RECIPES = {  # tokens, branches, decay_frac, name suffix, keep params
    "tiny": ("1200e6", "75e6,150e6,300e6,600e6,1200e6", 0.1, "", True),
    "C": ("1200e6", "300e6,1200e6", 0.2, "_dec2", False),
    "D": ("4800e6", "1200e6,2400e6,4800e6", 0.1, "_4800M", True),
    "d384": ("300e6", "150e6,300e6", 0.1, "", False),
}
LABEL_ARM = {"B_r8": "tB", "H2": "tH2", "H2tm": "tH2tm", "H2a": "tH2a", "H2atm": "tH2atm", "H2u": "tH2u",
             "B384": "B", "H2_384": "H2_384", "H2tm384": "H2tm384", "H2a384": "H2a384", "H2atm384": "H2atm384"}


def done(name):
    return os.path.exists(os.path.join(OUT, name + ".DONE"))


def run(recipe, label, seed, extra):
    tokens, br, dfrac, suffix, keep = RECIPES[recipe]
    name = f"wsd_{label}{suffix}_s{seed}"
    if done(name):
        print(f"[r9] {name} is DONE; skipped", flush=True)
        return
    cmd = [sys.executable, "train.py", "--arm", LABEL_ARM[label], "--seed", str(seed), "--lr", str(LR),
           "--tokens", tokens, "--sched", "wsd", "--branch_at", br, "--decay_frac", str(dfrac), "--n_evals", "12",
           "--name", name, "--runs_dir", OUT] + (["--keep_params"] if keep else []) + extra
    print(f"[r9] start {name}: {' '.join(cmd[1:])}", flush=True)
    t = time.time()
    with open(os.path.join(OUT, name + ".out"), "w") as out:
        rc = subprocess.call(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    dt = time.time() - t
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"used_s": 0.0, "runs": {}}
    L["used_s"] += dt
    L["runs"][name] = {"s": dt, "rc": rc}
    json.dump(L, open(LEDGER + ".tmp", "w"), indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)
    print(f"[r9] end {name}: rc {rc}, {dt / 60:.1f} min, round-9 total {L['used_s'] / 3600:.2f} h", flush=True)
    if rc != 0 or not done(name):
        print(f"[r9] {name} FAILED (rc {rc}); stopping so it can be looked at", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--extra", default="", help="extra train.py args, space separated")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for x in a.runs.split(","):
        lab, s = x.split(":")
        run(a.recipe, lab, int(s), a.extra.split())
    print("[r9] finished", flush=True)
