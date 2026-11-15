#!/bin/bash
# 2026-11-06: round-8 steps 1-5 after B_r8 is DONE. Stops at the first failure; starts no full run.
set -eo pipefail
source "/path/to/repo/env.sh"
O=runs/round8
test -f $O/wsd_B_r8_s0.DONE
echo "[post] classes $(date)"
JAX_PLATFORMS=cpu python round8_classes.py > $O/classes.out 2>&1
tail -3 $O/classes.out
python - <<'EOF'
import numpy as np, json
km = np.load("runs/round8/classes_kmeans1024.npy"); s = np.bincount(km, minlength=1024)
print("empty k-means classes:", int((s == 0).sum()))
assert (s > 0).all(), "empty k-means classes: hier normalisation would leak mass; stop and report"
EOF
echo "[post] diag B $(date)"
python round8_diag.py --arm tB --ckpt wsd_B_r8_s0 --tag B > $O/diag_B.out 2>&1
grep -A3 grad_retention_flat $O/diag_B.out | head -8
echo "[post] gates $(date)"
JAX_PLATFORMS=cpu python round8_gates.py > $O/gates.out 2>&1
grep -E "FAIL|ALL GATES" $O/gates.out
grep -q "ALL GATES OK" $O/gates.out
echo "[post] smoke $(date)"
bash r8smoke.sh
echo "[post] done $(date)"
