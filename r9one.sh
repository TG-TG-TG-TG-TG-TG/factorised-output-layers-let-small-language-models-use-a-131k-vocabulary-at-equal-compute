#!/bin/bash
# 2026-11-07: round-9 run(s), nothing starts after the list.  bash r9one.sh d384 B384:0  |  bash r9one.sh tiny H2tm:0
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
mkdir -p runs/round9
log=runs/round9/queue.log
echo "[round9] $1 $2, start $(date)" >> $log
python -u round9_queue.py --recipe "$1" --runs "$2" 2>&1 | tee -a $log
echo "[round9] $1 $2, end $(date)" >> $log
