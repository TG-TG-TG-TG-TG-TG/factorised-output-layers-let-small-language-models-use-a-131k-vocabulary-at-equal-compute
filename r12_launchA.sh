#!/bin/bash
# 2026-11-10 23:00: round-12 queue A. Waits for chainR7's B384 seed-1 run, stops chainR7 before its next run starts, then:
# the bpe131072 d=128 proxy -> the rest of chainR7 (H3*-384 s1 + BLiMP s1) -> gates guard -> smokes -> 2-hour guard ->
# D1 B384 matched s0/s1 -> F128k-384 -> H128k-384 -> D2 bios seed 1 -> chainR7's tiny tB s1 + its BLiMP (moved last).
ulimit -c 0
cd "/path/to/repo"
while [ ! -f runs/round11/wsd_B384_300M_s1.DONE ]; do sleep 10; done
sleep 3
pkill -f "[r]11chain.sh"; pkill -f "[r]11one.sh"; pkill -f "[r]ound11_queue.py"; sleep 2; pkill -f "[t]rain.py"; sleep 8
echo "[launchA] chainR7 stopped after B384 s1 $(date)"
bash r12chain.sh one+d384+tF128k:0+--label+tF128k_proxy+--tokens+50003968+--branch_at+50003968 || exit 1
bash r11chain.sh one+d384+H3s384:1+--label+H3s384+--tokens+575e6+--branch_at+300e6,575e6+--extra=--keep_branch=300e6 \
  pytry+round11_eval.py+blimp+B384_s1,H3s384_300M_s1,H3s384_575M_s1,H3k8c4_s1,H3rs16_s1 >> runs/round11/chainR7b.out 2>&1 || exit 1
bash r12chain.sh py+round12_require.py+gates \
  smoke+B@315162624+F128k384@160759808+H128k384@574750720 py+round12_smoke_check.py py+round12_require.py+smoke \
  one+d384+B:0+--label+B384m+--tokens+315162624+--branch_at+164429824,315162624 \
  one+d384+B:1+--label+B384m+--tokens+315162624+--branch_at+164429824,315162624 \
  one+d384+F128k384:0+--label+F128k384+--tokens+160759808+--branch_at+83886080,160759808+--extra=--keep_branch=83886080 \
  one+d384+H128k384:0+--label+H128k384+--tokens+574750720+--branch_at+299892736,521797632,574750720+--extra=--keep_branch=299892736,521797632 \
  one+bios+bioB_100000:1+--tokens+505282560+--branch_at+505282560 \
  one+bios+bioH_100000:1+--tokens+1199505408+--branch_at+1199505408 || exit 1
bash r11chain.sh one+tiny+tB:1 pytry+round11_eval.py+blimp+B_r8_s1 >> runs/round11/chainR7b.out 2>&1
echo "[launchA] done $(date)"
