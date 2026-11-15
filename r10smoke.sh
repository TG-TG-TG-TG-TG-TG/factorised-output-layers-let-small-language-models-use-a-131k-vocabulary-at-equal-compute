#!/bin/bash
# 2026-11-07: round-10 200-step smoke runs, back to back. The horizon only sets the full-run estimate:
# t* (part A / D, d=128 1200M), R* (part C, 600M), *384 / B (d=384, 650M and 300M).
# bash r10smoke.sh tB tH3k8 tH3c4 tH3k8c4    |   bash r10smoke.sh B H3s384   |   bash r10smoke.sh tD1 tD1@lazy
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
S=runs/round10/smoke
mkdir -p $S
for spec in "$@"; do
  arm=${spec%@lazy}; extra=""; tag=$arm          # tD1@lazy -> arm tD1 with --ng_lazy, log smoke_tD1_lazy
  if [ "$spec" != "$arm" ]; then extra="--ng_lazy"; tag=${arm}_lazy; fi
  case $arm in
    t*) tok=1200e6; br=75e6,150e6,300e6,600e6,1200e6 ;;
    R*) tok=600e6;  br=75e6,150e6,300e6,600e6 ;;
    B)  tok=300e6;  br=150e6,300e6 ;;
    *)  tok=650e6;  br=300e6,650e6 ;;
  esac
  rm -f $S/smoke_$tag.jsonl
  python -u train.py --arm $arm --seed 0 --lr 0.002 --tokens $tok --sched wsd \
    --branch_at $br --n_evals 12 --max_steps 200 $extra \
    --name smoke_$tag --runs_dir $S > $S/smoke_$tag.out 2>&1
  tail -2 $S/smoke_$tag.out
done
