#!/bin/bash
# 2026-11-13 18:15: round-13 experiment 4 follow-up (rule pinned in round13_analyze.py before this run): adaptive softmax
# configuration 2 (AS128k384c49, cutoffs 8,192 / 49,152) seed 1, the wsd_AS128k384c49_612M_s0 command with seed 1, then
# its byte-aligned BPB evals and the analysis. ~110 min (seed 0 took 109 min on a shared GPU; smoke estimate 83 min).
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchF.sh > runs/round13/chainF.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh one+d384+AS128k384c49:1+--label+AS128k384c49+--tokens+611909632+--branch_at+319356928,611909632+--extra=--keep_branch=319356928 \
  py+round13_eval.py+bpb+AS128k384c49_612M_s1,AS128k384c49_319M_s1+skipmissing \
  cpu+round13_analyze.py
echo "[launchF] done $(date)"
