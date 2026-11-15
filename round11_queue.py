"""Round-11 queue (2026-11-09): is the two-level head cheap AND good? Logs, DONE markers, the GPU-time ledger and
every new output go to runs/round11/. Old runs are never touched.

Recipes (LR 2e-3 unless --lr, WSD, warmup 3M tokens, 10% decay branches, n_evals 12, same data order as every
earlier round unless --extra selects another data source):
  tiny  (d=128, rounds 5/8-10): stable to 1200M, branches 75M/150M/300M/600M/1200M
  d384  (round-5 part-B shape):  stable to --tokens T, branches --branch_at (default "T")
  bios  (round-11 E4):           stable to --tokens T, one decay at T (branch_at T)

The run name is wsd_<label>[_lr..][_<T>M]_s<seed>; label = --label if given, else the arm without a leading "t".

python round11_queue.py --recipe d384 --runs B:0 --label B384 --tokens 300e6 --branch_at 150e6,300e6
python round11_queue.py --recipe tiny --runs tH3rs16:0
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round11")
LEDGER = os.path.join(OUT, "budget.json")
RECIPES = {  # tokens, branches, keep params
    "tiny": ("1200e6", "75e6,150e6,300e6,600e6,1200e6", True),
    "d384": (None, None, True),
    "bios": (None, None, True),
}


def done(name):
    return os.path.exists(os.path.join(OUT, name + ".DONE"))


def run(recipe, arm, seed, lr, tokens, branch_at, extra, label):
    tok, br, keep = RECIPES[recipe]
    tok = tokens or tok
    br = branch_at or br or tok
    label = label or (arm[1:] if arm.startswith("t") else arm)
    sfx = "" if lr == 2e-3 else f"_lr{lr:g}"
    if recipe in ("d384", "bios"):
        sfx += f"_{float(tok) / 1e6:.0f}M"
    name = f"wsd_{label}{sfx}_s{seed}"
    if done(name):
        print(f"[r11] {name} is DONE; skipped", flush=True)
        return
    cmd = [sys.executable, "train.py", "--arm", arm, "--seed", str(seed), "--lr", str(lr),
           "--tokens", tok, "--sched", "wsd", "--branch_at", br, "--decay_frac", "0.1", "--n_evals", "12",
           "--name", name, "--runs_dir", OUT] + (["--keep_params"] if keep else []) + extra
    print(f"[r11] start {name}: {' '.join(cmd[1:])}", flush=True)
    t = time.time()
    with open(os.path.join(OUT, name + ".out"), "w") as out:
        rc = subprocess.call(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    dt = time.time() - t
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"used_s": 0.0, "runs": {}}
    L["used_s"] += dt
    L["runs"][name] = {"s": dt, "rc": rc}
    json.dump(L, open(LEDGER + ".tmp", "w"), indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)
    print(f"[r11] end {name}: rc {rc}, {dt / 60:.1f} min, round-11 total {L['used_s'] / 3600:.2f} h", flush=True)
    if rc != 0 or not done(name):
        print(f"[r11] {name} FAILED (rc {rc}); stopping so it can be looked at", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--runs", required=True, help="arm:seed,...  (train.py arm names)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--branch_at", default=None)
    ap.add_argument("--extra", default="", help="extra train.py args, space separated")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for x in a.runs.split(","):
        arm, s = x.split(":")
        run(a.recipe, arm, int(s), a.lr, a.tokens, a.branch_at, a.extra.split(), a.label)
    print("[r11] finished", flush=True)
