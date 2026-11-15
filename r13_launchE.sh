#!/bin/bash
# 2026-11-13: round-13 experiment 4 (rules pinned in round13_analyze.py before any of these runs): 131k at 1x,
# two-level seed 1, adaptive softmax seed 1 (configuration 1), adaptive softmax configuration 2 (cutoffs 8,192 / 49,152)
# seed 0, then the byte-aligned BPB evals. ~4 h in all (~40 + ~90 + ~92 min + smoke + evals); every run is under the
# 2-hour rule and the queue skips a run whose .DONE exists, so the same command resumes after a shutdown (an interrupted
# run restarts from step 0). Only the new arm is smoked (smoke_AS128k384c49; yesterday's smokes are not overwritten);
# its estimate guards both adaptive-softmax runs (same sparse code, 1.3% more FLOPs/token).
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchE.sh > runs/round13/chainE.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh smoke+AS128k384c49@611909632 \
  py+round13_smoke_check.py+--tag+exp4+--arms+AS128k384c49 \
  py+round13_require_fast.py+runs/round13/smoke_check_exp4.json+AS128k384c49 \
  one+d384+H128k384:1+--label+H128k384+--tokens+574750720+--branch_at+299892736,521797632,574750720+--extra=--keep_branch=299892736,521797632 \
  one+d384+AS128k384s:1+--label+AS128k384+--tokens+619970560+--branch_at+323485696,619970560+--extra=--keep_branch=323485696 \
  one+d384+AS128k384c49:0+--label+AS128k384c49+--tokens+611909632+--branch_at+319356928,611909632+--extra=--keep_branch=319356928 \
  py+round13_eval.py+bpb+H128k384_522M_s1,H128k384_300M_s1,AS128k384_620M_s1,AS128k384_323M_s1,AS128k384c49_612M,AS128k384c49_319M+skipmissing \
  cpu+round13_analyze.py
echo "[launchE] done $(date)"
