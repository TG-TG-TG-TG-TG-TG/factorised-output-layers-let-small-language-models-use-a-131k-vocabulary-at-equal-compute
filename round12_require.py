"""Round 12 chain guard: exit 1 (the chain stops) unless the named checks pass.

python round12_require.py gates   gates_flat.json and gates_hier.json exist with no fails
python round12_require.py smoke   smoke_check.json: no fails, F128k384 / H128k384 first-3 data hashes identical (gate 4),
                                  every full-run estimate <= 120 min (the 2-hour rule)
"""
import json
import os
import sys

D = "runs/round12"
bad = []
for what in sys.argv[1:]:
    if what == "gates":
        for g in ("flat", "hier"):
            p = os.path.join(D, f"gates_{g}.json")
            if not os.path.exists(p):
                bad.append(f"{p} missing")
            elif json.load(open(p))["fails"]:
                bad.append(f"{p}: {json.load(open(p))['fails']}")
    elif what == "smoke":
        s = json.load(open(os.path.join(D, "smoke_check.json")))
        bad += s["fails"]
        A = s["arms"]
        h1, h2 = A.get("F128k384", {}).get("first_hashes"), A.get("H128k384", {}).get("first_hashes")
        if not h1 or h1 != h2:
            bad.append(f"gate 4: F128k / H128k first-batch hashes differ or missing ({h1} vs {h2})")
        else:
            print("gate 4 OK: F128k384 and H128k384 saw identical first 3 batches")
        for k, r in A.items():
            if r.get("est_full_min", 0) > 120:
                bad.append(f"2-hour rule: {k} estimated {r['est_full_min']:.0f} min")
            else:
                print(f"{k}: est full {r.get('est_full_min', 0):.0f} min")
print("REQUIRE OK" if not bad else "REQUIRE FAILED: " + "; ".join(bad), flush=True)
sys.exit(1 if bad else 0)
