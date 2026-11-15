#!/bin/bash
# 2026-11-11: round-12 run(s), nothing starts after the list.  bash r11one.sh tiny tH3rs16:0  |  bash r11one.sh d384 B:0 --label B384 --tokens 300e6
set -o pipefail
ulimit -c 0  # no multi-GB WSL core dumps on C:
source "/path/to/repo/env.sh"
mkdir -p runs/round13
log=runs/round13/queue.log
echo "[round13] $* start $(date)" >> $log
python -u round13_queue.py --recipe "$1" --runs "$2" "${@:3}" 2>&1 | tee -a $log
echo "[round13] $* end $(date)" >> $log
