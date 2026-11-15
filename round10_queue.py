"""Round-10 queue (2026-11-07): tail fixes, d=384 matched wall-clock, backbone reinvestment, hashed n-gram
input embeddings. Logs, DONE markers, the GPU-time ledger and every new output go to runs/round10/.
Old runs are never touched.

Recipes (LR 2e-3 unless --lr, WSD, warmup 3M tokens, 10% decay branches, n_evals 12, same data order as
every earlier round):
  tiny  (round-5 part A / rounds 8-9): stable to 1200M, branches 75M/150M/300M/600M/1200M
  c600  (part C):                      stable to 600M,  branches 75M/150M/300M/600M
  d384  (part B, round-5 part-B shape): stable to --tokens T, branches --branch_at (default "T")

python round10_queue.py --recipe tiny --runs tH3k8:0,tH3c4:0,tH3k8c4:0
python round10_queue.py --recipe d384 --runs H3s384:0 --tokens 650e6 --branch_at 650e6
python round10_queue.py --recipe c600 --runs Rloop:0 --lr 1e-3     # -> wsd_Rloop_lr0.001_s0
python round10_queue.py --recipe tiny --runs tD1:0 --extra=--ng_lazy  # -> wsd_D1_lazy_s0
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round10")
LEDGER = os.path.join(OUT, "budget.json")
RECIPES = {  # tokens, branches, keep params
    "tiny": ("1200e6", "75e6,150e6,300e6,600e6,1200e6", True),
    "c600": ("600e6", "75e6,150e6,300e6,600e6", True),
    "d384": (None, None, True),
}


def done(name):
    return os.path.exists(os.path.join(OUT, name + ".DONE"))


def run(recipe, arm, seed, lr, tokens, branch_at, extra):
    tok, br, keep = RECIPES[recipe]
    tok = tokens or tok
    br = branch_at or br or tok
    label = arm[1:] if arm.startswith("t") else arm  # tH3k8 -> H3k8 (as rounds 8-9 name tiny runs)
    sfx = "" if lr == 2e-3 else f"_lr{lr:g}"
    if recipe == "d384":
        sfx += f"_{float(tok) / 1e6:.0f}M"
    if "--ng_lazy" in extra:
        sfx += "_lazy"
    name = f"wsd_{label}{sfx}_s{seed}"
    if done(name):
        print(f"[r10] {name} is DONE; skipped", flush=True)
        return
    cmd = [sys.executable, "train.py", "--arm", arm, "--seed", str(seed), "--lr", str(lr),
           "--tokens", tok, "--sched", "wsd", "--branch_at", br, "--decay_frac", "0.1", "--n_evals", "12",
           "--name", name, "--runs_dir", OUT] + (["--keep_params"] if keep else []) + extra
    print(f"[r10] start {name}: {' '.join(cmd[1:])}", flush=True)
    t = time.time()
    with open(os.path.join(OUT, name + ".out"), "w") as out:
        rc = subprocess.call(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    dt = time.time() - t
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"used_s": 0.0, "runs": {}}
    L["used_s"] += dt
    L["runs"][name] = {"s": dt, "rc": rc}
    json.dump(L, open(LEDGER + ".tmp", "w"), indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)
    print(f"[r10] end {name}: rc {rc}, {dt / 60:.1f} min, round-10 total {L['used_s'] / 3600:.2f} h", flush=True)
    if rc != 0 or not done(name):
        print(f"[r10] {name} FAILED (rc {rc}); stopping so it can be looked at", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--runs", required=True, help="arm:seed,...  (train.py arm names)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--branch_at", default=None)
    ap.add_argument("--extra", default="", help="extra train.py args, space separated")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for x in a.runs.split(","):
        arm, s = x.split(":")
        run(a.recipe, arm, int(s), a.lr, a.tokens, a.branch_at, a.extra.split())
    print("[r10] finished", flush=True)
