#!/bin/bash
# 2026-11-07: round-10 run(s), nothing starts after the list.  bash r10one.sh tiny H3k8:0  |  bash r10one.sh c600 Rloop:0
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
mkdir -p runs/round10
log=runs/round10/queue.log
echo "[round10] $* start $(date)" >> $log
python -u round10_queue.py --recipe "$1" --runs "$2" "${@:3}" 2>&1 | tee -a $log
echo "[round10] $* end $(date)" >> $log
