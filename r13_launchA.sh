#!/bin/bash
# 2026-11-11 21:45: round-13 experiment 1, tonight's part: the charged 50k proxy -> its classes (CPU) -> 200-step smoke
# of H3p384 -> H3p384 seed 0 (branches 0.5x charged, 0.5x, 1x charged, 1x). Seed 1 and experiment 2 run tomorrow.
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh one+d384+tB:0+--label+tBproxy+--tokens+50003968+--branch_at+50003968 \
  cpu+round13_classes.py \
  smoke+H3p384@575e6 \
  one+d384+H3p384:0+--label+H3p384+--tokens+575e6+--branch_at+276299776,300e6,551288832,575e6
echo "[launchA] done $(date)"
