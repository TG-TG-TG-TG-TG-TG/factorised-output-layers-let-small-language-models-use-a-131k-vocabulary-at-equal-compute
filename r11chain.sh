#!/bin/bash
# 2026-11-09: run round-11 steps one after another, stop at the first failure. Steps use + instead of spaces.
# wsl -d AEON-Training --cd <win path> bash r11chain.sh one+d384+B:0+--label+B384+--tokens+300e6+--branch_at+150e6,300e6
ulimit -c 0
cd "/path/to/repo"
mkdir -p runs/round11
for step in "$@"; do
  set -- ${step//+/ }
  kind=$1; shift
  echo "[r11chain] $kind $* start $(date)"
  case $kind in
    smoke) bash r11smoke.sh "$@" || exit 1 ;;
    one)   bash r11one.sh "$@" || exit 1 ;;
    py)    (source env.sh; python -u "$@") || exit 1 ;;
    cpu)   (source env.sh; JAX_PLATFORMS=cpu python -u "$@") || exit 1 ;;
    pytry) (source env.sh; python -u "$@") || echo "[r11chain] $* failed (not fatal)" ;;
    waitidle) while pgrep -f "train.py|round11_" > /dev/null; do sleep 15; done ;;
  esac
  echo "[r11chain] $kind $* end $(date)"
done
