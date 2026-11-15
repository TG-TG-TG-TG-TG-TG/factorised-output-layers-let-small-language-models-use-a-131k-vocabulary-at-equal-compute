#!/bin/bash
# 2026-11-10: round-12 run(s), nothing starts after the list.  bash r11one.sh tiny tH3rs16:0  |  bash r11one.sh d384 B:0 --label B384 --tokens 300e6
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
mkdir -p runs/round12
log=runs/round12/queue.log
echo "[round12] $* start $(date)" >> $log
python -u round12_queue.py --recipe "$1" --runs "$2" "${@:3}" 2>&1 | tee -a $log
echo "[round12] $* end $(date)" >> $log
