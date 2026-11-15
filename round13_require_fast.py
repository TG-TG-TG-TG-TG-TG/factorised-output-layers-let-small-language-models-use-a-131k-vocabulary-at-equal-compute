"""Stop the chain (exit 1) unless the named smoke's full-run estimate is under the user's 2-hour rule (default 115 min).
python round13_require_fast.py runs/round13/smoke_check_exp3s.json AS128k384s [max_min]"""
import json
import sys

d = json.load(open(sys.argv[1]))["arms"][sys.argv[2]]
lim = float(sys.argv[3]) if len(sys.argv) > 3 else 115.0
print(f"[require_fast] {sys.argv[2]}: est {d['est_full_min']:.0f} min, {d['tok_s']:,.0f} tok/s (limit {lim:.0f} min)", flush=True)
sys.exit(0 if not d.get("crashed") and d["est_full_min"] < lim and d["finite"] else 1)
