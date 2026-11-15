#!/bin/bash
# 2026-11-07: run a few round-9 steps one after another, stop at the first failure.
# wsl -d AEON-Training --cd <win path> bash r9chain.sh smoke+B one+d384+B384:0   (steps use + instead of spaces)
ulimit -c 0
cd "/path/to/repo"
for step in "$@"; do
  set -- ${step//+/ }
  kind=$1; shift
  echo "[r9chain] $kind $* start $(date)"
  case $kind in
    smoke) bash r9smoke.sh "$@" || exit 1 ;;
    one)   bash r9one.sh "$@" || exit 1 ;;
    py)    (source env.sh; python -u "$@") || exit 1 ;;
    pytry) (source env.sh; python -u "$@") || echo "[r9chain] $* failed (not fatal)" ;;
    waitdone) while [ ! -f "runs/round9/$1.DONE" ]; do sleep 20; done ;;
    waitidle) while pgrep -f "round8_gates|round9_gates|round9_freq|train.py" > /dev/null; do sleep 15; done ;;
  esac
  echo "[r9chain] $kind $* end $(date)"
done
