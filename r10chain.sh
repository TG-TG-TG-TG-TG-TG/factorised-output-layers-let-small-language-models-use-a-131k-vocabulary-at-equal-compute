#!/bin/bash
# 2026-11-07: run round-10 steps one after another, stop at the first failure.
# wsl -d AEON-Training --cd <win path> bash r10chain.sh smoke+tB+tH3k8 one+tiny+H3k8:0   (steps use + instead of spaces)
ulimit -c 0
cd "/path/to/repo"
mkdir -p runs/round10
for step in "$@"; do
  set -- ${step//+/ }
  kind=$1; shift
  echo "[r10chain] $kind $* start $(date)"
  case $kind in
    smoke) bash r10smoke.sh "$@" || exit 1 ;;
    one)   bash r10one.sh "$@" || exit 1 ;;
    py)    (source env.sh; python -u "$@") || exit 1 ;;
    cpu)   (source env.sh; JAX_PLATFORMS=cpu python -u "$@") || exit 1 ;;
    pytry) (source env.sh; python -u "$@") || echo "[r10chain] $* failed (not fatal)" ;;
    waitidle) while pgrep -f "round10_gates|round10_classes|train.py" > /dev/null; do sleep 15; done ;;
  esac
  echo "[r10chain] $kind $* end $(date)"
done
