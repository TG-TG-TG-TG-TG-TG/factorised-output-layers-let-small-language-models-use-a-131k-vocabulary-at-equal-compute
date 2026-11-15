#!/bin/bash
# 2026-11-12: round-13 experiment 3, the 131k adaptive-softmax run with the sparse band path (arm AS128k384s = AS128k384
# + head_sparse; same model and FLOP count, gated equal in round13_gates_as_sparse.py). Smoke -> smoke check -> the run
# (label AS128k384, so the name is wsd_AS128k384_620M_s0; the HOLD marker of that name must be deleted first) -> evals.
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchD.sh > runs/round13/chainD.out 2>&1
ulimit -c 0
cd "/path/to/repo"
if grep -q HOLD runs/round13/wsd_AS128k384_620M_s0.DONE 2>/dev/null; then echo "[launchD] HOLD marker still there; delete it first"; exit 1; fi
bash r13chain.sh smoke+AS128k384s@619970560 \
  py+round13_smoke_check.py+--tag+exp3s+--arms+AS128k384s \
  py+round13_require_fast.py+runs/round13/smoke_check_exp3s.json+AS128k384s \
  one+d384+AS128k384s:0+--label+AS128k384+--tokens+619970560+--branch_at+323485696,619970560+--extra=--keep_branch=323485696 \
  py+round13_eval.py+bpb+AS128k384_620M,AS128k384_323M,AS384_624M_s0
echo "[launchD] done $(date)"
