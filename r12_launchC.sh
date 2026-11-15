#!/bin/bash
# 2026-11-11 08:55: resume of round 12 after the user's 01:03 stop. Queue A from the F128k384 step (killed partial moved to
# runs/round12/void_killed), then chainR7's tB s1 + BLiMP, then queue B v2's eval steps unchanged (r12_launchB2.sh).
ulimit -c 0
cd "/path/to/repo"
echo "[launchC] start $(date)"
bash r12chain.sh \
  one+d384+F128k384:0+--label+F128k384+--tokens+160759808+--branch_at+83886080,160759808+--extra=--keep_branch=83886080 \
  one+d384+H128k384:0+--label+H128k384+--tokens+574750720+--branch_at+299892736,521797632,574750720+--extra=--keep_branch=299892736,521797632 \
  one+bios+bioB_100000:1+--tokens+505282560+--branch_at+505282560 \
  one+bios+bioH_100000:1+--tokens+1199505408+--branch_at+1199505408 || exit 1
bash r11chain.sh one+tiny+tB:1 pytry+round11_eval.py+blimp+B_r8_s1 >> runs/round11/chainR7b.out 2>&1
sed -n '/^bash r12chainB.sh/,/^echo "\[launchB2\] done/p' r12_launchB2.sh | head -n -1 > /tmp/r12B_steps.sh
bash /tmp/r12B_steps.sh
echo "[launchC] done $(date)"
