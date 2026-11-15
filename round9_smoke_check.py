"""Round 9 smoke check: tok/s, step time, peak VRAM, head forward FLOPs/token and full-run estimates from the
200-step smoke runs in runs/round9/smoke/, plus the data-hash half of gate 4 (data_hash events vs
runs/round8/data_hash_ref.json, the first 3 batches every round-8 run saw).

Full-run estimate = 1.103 x total tokens / tok_s (1.103 = round-5 tB wall over its pure stable-phase time:
branches + evals overhead, as round 8). Total tokens: tiny 1,320,026,112; d384 330,039,296 (300M + 10% decay).
Ratios are against tB (tiny arms) or B (d=384 arms) from the same smoke batch.

python round9_smoke_check.py [--tag NAME]  -> runs/round9/smoke_check[_NAME].json
"""
import argparse
import glob
import json
import os

from arms import ARMS
from round8_heads import head_flops_fwd as hf8
import round9_heads as R9

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--arms", default="", help="comma list; default every smoke log")
a = ap.parse_args()
S = "runs/round9/smoke"
OVER = 4440.97 / (1_320_026_112 / 327814.77)
TOK = {128: 1_320_026_112, 384: 330_039_296}
ref = json.load(open("runs/round8/data_hash_ref.json"))["sha256_first3"]
want = set(a.arms.split(",")) if a.arms else None
res, fails = {}, []
for p in sorted(glob.glob(os.path.join(S, "smoke_*.jsonl"))):
    R = [json.loads(l) for l in open(p)]
    arm = R[0]["arm"]
    if want and arm not in want:
        continue
    cfg = ARMS[arm]
    cfgrec = next(r for r in R if r.get("event") == "config")
    tr = [r for r in R if r.get("event") == "train"]
    if not tr:
        print(f"{arm:8s} CRASHED: no train events (see {p[:-6]}.out)")
        res[arm] = {"crashed": True}
        fails.append(f"crash {arm}")
        continue
    last = tr[-1]
    hs = [r["sha256"] for r in sorted((r for r in R if r.get("event") == "data_hash"), key=lambda r: r["step"])]
    tok_s = last["tok_s"]
    est = OVER * TOK[cfg.d] / tok_s
    hf = R9.head_flops_fwd(cfg) if cfg.head == "hier9" else hf8(cfg)
    res[arm] = {"d": cfg.d, "steps": last["step"], "loss_at_200": last["loss"], "tok_s": tok_s,
                "step_ms": 1000 * cfgrec["tokens_per_step"] / tok_s, "peak_mem_gb": last["peak_mem_gb"],
                "data_wait_ms": last.get("data_wait_ms"), "est_full_min": est / 60,
                "head_fwd_flops_per_token": hf, "data_hash_ok": hs == ref,
                "finite": all(r["loss"] == r["loss"] for r in tr), "params": cfgrec["params"],
                "flops_per_token": cfgrec["flops_per_token"], "mtime": os.path.getmtime(p)}
    if hs != ref:
        fails.append(f"gate 4 data hash {arm}")
for arm, r in res.items():
    if r.get("crashed"):
        continue
    base = res.get("tB" if r["d"] == 128 else "B", {})
    r["speed_vs_B"] = r["tok_s"] / base["tok_s"] if base.get("tok_s") else None
    r["head_flops_vs_B"] = r["head_fwd_flops_per_token"] / base["head_fwd_flops_per_token"] if base.get("tok_s") else None
    print(f"{arm:8s} loss@200 {r['loss_at_200']:.4f}  {r['tok_s']:>10,.0f} tok/s ({r['speed_vs_B'] or 0:.2f}x B)  "
          f"step {r['step_ms']:.0f} ms  peak {r['peak_mem_gb']:.2f} GB  wait {r['data_wait_ms']:.1f} ms  "
          f"head {r['head_fwd_flops_per_token']:,.0f} FLOPs/tok ({r['head_flops_vs_B'] or 0:.3f}x B)  "
          f"est full {r['est_full_min']:.0f} min  data hash {'OK' if r['data_hash_ok'] else 'MISMATCH'}")
print("gate 4 (data) / smoke:", "OK" if not fails else "FAILED " + ", ".join(fails))
out = f"runs/round9/smoke_check{'_' + a.tag if a.tag else ''}.json"
json.dump({"arms": res, "overhead_factor": OVER, "fails": fails}, open(out, "w"), indent=1)
