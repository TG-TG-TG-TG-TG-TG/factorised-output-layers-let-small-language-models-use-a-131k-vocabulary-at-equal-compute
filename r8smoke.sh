#!/bin/bash
# 2026-11-06: round-8 step 5, 200 steps of each arm (same WSD recipe/args as the full runs), then the check.
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C: (2026-11-06)
source "/path/to/repo/env.sh"
S=runs/round8/smoke
mkdir -p $S
for arm in ${@:-tB tH2 tH2r tCB tH2w}; do
  rm -f $S/smoke_$arm.jsonl
  python -u train.py --arm $arm --seed 0 --lr 0.002 --tokens 1200e6 --sched wsd \
    --branch_at 75e6,150e6,300e6,600e6,1200e6 --n_evals 12 --max_steps 200 \
    --name smoke_$arm --runs_dir $S > $S/smoke_$arm.out 2>&1
  tail -2 $S/smoke_$arm.out
done
python round8_smoke_check.py | tee runs/round8/smoke_check.txt
