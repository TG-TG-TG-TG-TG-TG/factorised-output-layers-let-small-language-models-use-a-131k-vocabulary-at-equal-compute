#!/bin/bash
# 2026-11-12: round-13 experiment 3 (adaptive softmax, rules pinned 13:27 in round13_analyze.py): 200-step smokes of
# AS384 and AS128k384 -> smoke check -> AS384 seeds 0 and 1 (1x, branch 0.5x) -> AS128k384 seed 0 (1x, branch 0.5x kept)
# -> byte-aligned BPB evals (131k of record; 50k s0 cross-check). Not started until the user says so.
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchC.sh > runs/round13/chainC.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh smoke+AS384@624427008+AS128k384@619970560 \
  pytry+round13_smoke_check.py+--tag+exp3 \
  one+d384+AS384:0+--label+AS384+--tokens+624427008+--branch_at+325844992,624427008 \
  one+d384+AS384:1+--label+AS384+--tokens+624427008+--branch_at+325844992,624427008 \
  one+d384+AS128k384:0+--label+AS128k384+--tokens+619970560+--branch_at+323485696,619970560+--extra=--keep_branch=323485696 \
  py+round13_eval.py+bpb+AS128k384_620M,AS128k384_323M,AS384_624M_s0
echo "[launchC] done $(date)"
