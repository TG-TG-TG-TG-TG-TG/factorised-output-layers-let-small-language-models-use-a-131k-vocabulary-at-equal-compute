"""Round 12 smoke check (sed copy of round11_smoke_check.py) (copy of round10_smoke_check.py: dirs -> runs/round12; head FLOPs from model.flops_per_token; the data hash is
checked only for FineWeb-Edu GPT-2 arms, cfg.data == ""; bpe / bios arms log their own first-batch hashes): tok/s, step time, peak VRAM, head forward FLOPs/token and full-run estimates from the
200-step smoke runs in runs/round10/smoke/, plus the data-hash half of gate 4 (data_hash events vs
runs/round8/data_hash_ref.json, the first 3 batches every round-8 run saw).

Full-run estimate = 1.103 x total tokens / tok_s (1.103 = round-5 tB wall over its pure stable-phase time:
branches + evals overhead, as round 8). Total tokens = the smoke's --tokens + its 10% final decay.
Ratios are against tB (d=128 / 256 arms) or B (d=384 arms) from the same smoke batch.

python round12_smoke_check.py [--tag NAME] [--arms a,b]  -> runs/round12/smoke_check[_NAME].json
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
S = "runs/round12/smoke"
OVER = 4440.97 / (1_320_026_112 / 327814.77)
ref = json.load(open("runs/round8/data_hash_ref.json"))["sha256_first3"]
want = set(a.arms.split(",")) if a.arms else None
res, fails = {}, []
for p in sorted(glob.glob(os.path.join(S, "smoke_*.jsonl"))):
    R = [json.loads(l) for l in open(p)]
    arm = os.path.basename(p)[6:-6]   # smoke_<tag>.jsonl: tag = arm, or arm_lazy for --ng_lazy smokes
    if want and arm not in want:
        continue
    cfg = ARMS[R[0]["arm"]]
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
    tot = round(float(cfgrec['args']['tokens']) / 65536) * 65536 * 1.1
    est = OVER * tot / tok_s
    hf = cfgrec["flops_per_token"]["head"] / 3 if cfg.head else 2 * cfg.vocab * cfg.d
    res[arm] = {"d": cfg.d, "steps": last["step"], "loss_at_200": last["loss"], "tok_s": tok_s,
                "step_ms": 1000 * cfgrec["tokens_per_step"] / tok_s, "peak_mem_gb": last["peak_mem_gb"],
                "data_wait_ms": last.get("data_wait_ms"), "est_full_min": est / 60,
                "head_fwd_flops_per_token": hf, "data_hash_ok": (hs == ref) if not cfg.data else None, "data": cfg.data, "first_hashes": hs,
                "finite": all(r["loss"] == r["loss"] for r in tr), "params": cfgrec["params"],
                "flops_per_token": cfgrec["flops_per_token"], "total_tokens": tot, "mtime": os.path.getmtime(p)}
    if hs != ref and not cfg.data:
        fails.append(f"gate 4 data hash {arm}")
for arm, r in res.items():
    if r.get("crashed"):
        continue
    base = res.get("B" if r["d"] >= 384 else "tB", {})
    r["speed_vs_B"] = r["tok_s"] / base["tok_s"] if base.get("tok_s") else None
    r["head_flops_vs_B"] = r["head_fwd_flops_per_token"] / base["head_fwd_flops_per_token"] if base.get("tok_s") else None
    r["train_flops_vs_B"] = r["flops_per_token"]["total"] / base["flops_per_token"]["total"] if base.get("tok_s") else None
    print(f"{arm:8s} loss@200 {r['loss_at_200']:.4f}  {r['tok_s']:>10,.0f} tok/s ({r['speed_vs_B'] or 0:.2f}x B)  "
          f"step {r['step_ms']:.0f} ms  peak {r['peak_mem_gb']:.2f} GB  wait {r['data_wait_ms']:.1f} ms  "
          f"head {r['head_fwd_flops_per_token']:,.0f} FLOPs/tok ({r['head_flops_vs_B'] or 0:.3f}x B)  "
          f"train FLOPs {r['train_flops_vs_B'] or 0:.3f}x B  est full {r['est_full_min']:.0f} min  data hash {'n/a (' + r['data'] + ')' if r['data_hash_ok'] is None else 'OK' if r['data_hash_ok'] else 'MISMATCH'}")
print("gate 4 (data) / smoke:", "OK" if not fails else "FAILED " + ", ".join(fails))
out = f"runs/round12/smoke_check{'_' + a.tag if a.tag else ''}.json"
json.dump({"arms": res, "overhead_factor": OVER, "fails": fails}, open(out, "w"), indent=1)
if fails:
    raise SystemExit(1)
