#!/bin/bash
# 2026-11-13 21:45: round-13 experiment 5 (rule pinned in round13_analyze.py at 21:40, before this script and any run):
# adaptive softmax setup 2 with full-width tail bands (AS128k384c49w: no projection, untied) seed 0 at 1x, then its
# byte-aligned BPB evals and the analysis. The smoke runs first and the chain STOPS unless its full-run estimate is under
# 115 min (round13_require_fast.py), so a too-slow arm costs only the smoke. train.py cannot resume: start it only when
# the GPU is free for ~2 h. The queue skips a run whose .DONE exists.
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchG.sh > runs/round13/chainG.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh smoke+AS128k384c49w@539623424 \
  py+round13_smoke_check.py+--tag+exp5+--arms+AS128k384c49w \
  py+round13_require_fast.py+runs/round13/smoke_check_exp5.json+AS128k384c49w \
  one+d384+AS128k384c49w:0+--label+AS128k384c49w+--tokens+539623424+--branch_at+281608192,539623424+--extra=--keep_branch=281608192 \
  py+round13_eval.py+bpb+AS128k384c49w_540M,AS128k384c49w_282M+skipmissing \
  cpu+round13_analyze.py
echo "[launchG] done $(date)"
