#!/bin/bash
# round-13 experiment 1 seed 1, then experiment 2 (2x budget, seed 0): flat B384 to 630M (1x branch = reproducibility
# check) and two-level H3p384 to 1150M (2x charged + 2x uncharged branches). Estimates: ~42 + ~77 + ~75 min.
# Start with: wsl -d AEON-Training --cd <project> bash r13_launchB.sh > runs/round13/chainB.out 2>&1
ulimit -c 0
cd "/path/to/repo"
bash r13chain.sh one+d384+H3p384:1+--label+H3p384+--tokens+575e6+--branch_at+276299776,300e6,551288832,575e6 \
  one+d384+B:0+--label+B384x2+--tokens+630259712+--branch_at+315162624,630259712 \
  one+d384+H3p384:0+--label+H3p384x2+--tokens+1149960192+--branch_at+1126236160,1149960192
echo "[launchB] done $(date)"
