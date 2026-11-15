#!/bin/bash
# 2026-11-07: round-9 200-step smoke runs, back to back (tiny arms: tiny recipe; d=384 arms: d384 recipe).
# bash r9smoke.sh tB tH2 tH2tm tH2a tH2atm      |   bash r9smoke.sh B H2a384
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
S=runs/round9/smoke
mkdir -p $S
for arm in "$@"; do
  case $arm in
    t*) tok=1200e6; br=75e6,150e6,300e6,600e6,1200e6 ;;
    *)  tok=300e6;  br=150e6,300e6 ;;
  esac
  rm -f $S/smoke_$arm.jsonl
  python -u train.py --arm $arm --seed 0 --lr 0.002 --tokens $tok --sched wsd \
    --branch_at $br --n_evals 12 --max_steps 200 \
    --name smoke_$arm --runs_dir $S > $S/smoke_$arm.out 2>&1
  tail -2 $S/smoke_$arm.out
done
