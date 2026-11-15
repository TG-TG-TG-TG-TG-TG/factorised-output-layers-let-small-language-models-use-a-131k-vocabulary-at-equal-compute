#!/bin/bash
# 2026-11-11: round-12 follow-up memory/speed sweep (round12_perf.py); one fresh process per configuration, because
# configurations sharing a process slowed each other (micro 32 after micro 16: 77k vs 157k tok/s).
ulimit -c 0
cd "/path/to/repo"
source env.sh
for arm in B384 H3s384 F8k384 F128k384 H128k384; do
  for m in 16 32 64; do python -u round12_perf.py train $arm $m 2>&1 | grep '^{'; done
  for n in 32 64; do python -u round12_perf.py score $arm $n 2>&1 | grep '^{'; done
done
python -u round12_perf.py decode 2>&1 | grep '^{'
python round12_perf.py md
echo "[perf] done $(date)"
