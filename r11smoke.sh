#!/bin/bash
# 2026-11-09: round-11 200-step smoke runs, back to back. spec = arm[@stable_tokens]; the horizon only sets the full-run estimate.
# bash r11smoke.sh tB tH3k8c4 tH3rs16     |   bash r11smoke.sh B H3s384 F8k384@631.3e6   |   bash r11smoke.sh bioB_100000@505.8e6
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
S=runs/round11/smoke
mkdir -p $S
for spec in "$@"; do
  arm=${spec%@*}; tok=""
  if [ "$spec" != "$arm" ]; then tok=${spec#*@}; fi
  case $arm in
    t*) tok=${tok:-1200e6} ;;
    B)  tok=${tok:-300e6} ;;
    *)  tok=${tok:-575e6} ;;
  esac
  rm -f $S/smoke_$arm.jsonl
  python -u train.py --arm $arm --seed 0 --lr 0.002 --tokens $tok --sched wsd \
    --branch_at $tok --n_evals 12 --max_steps 200 \
    --name smoke_$arm --runs_dir $S > $S/smoke_$arm.out 2>&1
  tail -2 $S/smoke_$arm.out
done
