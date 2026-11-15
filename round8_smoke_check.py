"""Round 8, step 5 check: tok/s, step time, peak VRAM and full-run estimates from the 200-step smoke
runs (runs/round8/smoke/), plus gate 5 (data_hash events vs runs/round8/data_hash_ref.json).

Full-run estimate = 1.103 x 1,320,026,112 tokens / tok_s, where 1.103 = round-5 tB wall time over its
pure stable-phase time (4441 s / (1.320e9 / 327,815 tok/s)): branches + evals overhead. The cut rule
(> 2 h -> 600M horizon) is applied to this estimate.

python round8_smoke_check.py  -> runs/round8/smoke_check.json
"""
import glob
import json
import os

S = "runs/round8/smoke"
TOK = 1_320_026_112
OVER = 4440.97 / (TOK / 327814.77)
ref = json.load(open("runs/round8/data_hash_ref.json"))["sha256_first3"]
res, fails = {}, []
for p in sorted(glob.glob(os.path.join(S, "smoke_*.jsonl"))):
    R = [json.loads(l) for l in open(p)]
    arm = R[0]["arm"]
    cfgrec = next(r for r in R if r.get("event") == "config")
    tr = [r for r in R if r.get("event") == "train"]
    if not tr:
        print(f"{arm:5s} CRASHED: no train events (see {p[:-6]}.out)")
        res[arm] = {"crashed": True}
        fails.append(f"crash {arm}")
        continue
    last = tr[-1]
    hs = [r["sha256"] for r in sorted((r for r in R if r.get("event") == "data_hash"), key=lambda r: r["step"])]
    tok_s = last["tok_s"]
    est = OVER * TOK / tok_s
    res[arm] = {"steps": last["step"], "loss_at_200": last["loss"], "tok_s": tok_s,
                "step_ms": 1000 * cfgrec["tokens_per_step"] / tok_s, "peak_mem_gb": last["peak_mem_gb"],
                "data_wait_ms": last.get("data_wait_ms"), "est_full_min": est / 60, "cut_to_600M": est > 7200,
                "data_hash_ok": hs == ref, "finite": all(r["loss"] == r["loss"] for r in tr),
                "params": cfgrec["params"], "flops_per_token": cfgrec["flops_per_token"]}
    if hs != ref:
        fails.append(f"gate 5 data hash {arm}")
b = res.get("tB", {}).get("tok_s")
for arm, r in res.items():
    if r.get("crashed"):
        continue
    r["speed_vs_tB"] = r["tok_s"] / b if b else None
    print(f"{arm:5s} loss@200 {r['loss_at_200']:.4f}  {r['tok_s']:>9,.0f} tok/s ({r['speed_vs_tB'] or 0:.2f}x tB)  "
          f"step {r['step_ms']:.0f} ms  peak {r['peak_mem_gb']:.2f} GB  wait {r['data_wait_ms']:.1f} ms  "
          f"est full {r['est_full_min']:.0f} min{'  -> CUT to 600M' if r['cut_to_600M'] else ''}  "
          f"data hash {'OK' if r['data_hash_ok'] else 'MISMATCH'}")
print("gate 5 / smoke:", "OK (all arms ran; all see the reference first 3 batches)" if not fails else "FAILED " + ", ".join(fails))
json.dump({"arms": res, "overhead_factor": OVER, "fails": fails}, open("runs/round8/smoke_check.json", "w"), indent=1)
