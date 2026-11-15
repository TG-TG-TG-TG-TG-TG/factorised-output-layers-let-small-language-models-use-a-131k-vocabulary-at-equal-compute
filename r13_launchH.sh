#!/bin/bash
# 2026-11-13 21:45: round-13 experiment 5, seed 1 -- ONLY if the pinned rule says "SEED 1 NEEDED"
# (-0.009 <= Delta_w(0) <= +0.003, round13_analyze.py). The r13_launchG.sh command with seed 1; no smoke (same arm).
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchH.sh > runs/round13/chainH.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh one+d384+AS128k384c49w:1+--label+AS128k384c49w+--tokens+539623424+--branch_at+281608192,539623424+--extra=--keep_branch=281608192 \
  py+round13_eval.py+bpb+AS128k384c49w_540M_s1,AS128k384c49w_282M_s1+skipmissing \
  cpu+round13_analyze.py
echo "[launchH] done $(date)"
