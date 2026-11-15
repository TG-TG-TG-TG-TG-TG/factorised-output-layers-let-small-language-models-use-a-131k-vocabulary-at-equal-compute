#!/bin/bash
# 2026-11-06: round-8 run(s), nothing starts after the list.  bash r8one.sh B_r8:0   |  bash r8one.sh H2:0 600e6
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C: (2026-11-06)
source "/path/to/repo/env.sh"
mkdir -p runs/round8
log=runs/round8/queue.log
echo "[round8] run $1 ${2:-1200e6}, start $(date)" >> $log
python -u round8_queue.py --runs "$1" --tokens "${2:-1200e6}" 2>&1 | tee -a $log
echo "[round8] run $1, end $(date)" >> $log
