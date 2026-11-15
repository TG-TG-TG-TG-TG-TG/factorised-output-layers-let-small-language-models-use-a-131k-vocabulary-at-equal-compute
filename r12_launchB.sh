#!/bin/bash
# 2026-11-10: round-12 queue B (evals), starts when queue A (r12_launchA.sh) has exited.
ulimit -c 0
cd "/path/to/repo"
while pgrep -f "[r]12_launchA.sh" > /dev/null; do sleep 30; done
echo "[launchB] start $(date)"
bash r12chain.sh cpu+round12_gates.py+trained+wsd_H128k384_575M_s0 \
  pytry+round12_eval.py+bpb+B384m_s0,B384m_s1,F128k384_84M,F128k384_161M,H128k384_300M,H128k384_522M,H128k384_575M \
  pytry+round12_eval.py+blimp+B384m_s0,B384m_s1,F128k384_161M,H128k384_575M \
  pytry+round12_eval.py+lambada+F128k384_161M,H128k384_575M \
  pytry+round12_eval.py+e4+bioB_100000_s1,bioH_100000_eqF_s1 \
  pytry+round12_jensen.py \
  pytry+round12_decode.py+run+0+1 pytry+round12_decode.py+run+1+1 pytry+round12_decode.py+run+2+1 \
  pytry+round12_decode.py+run+0+32 pytry+round12_decode.py+run+1+32 pytry+round12_decode.py+run+2+32 pytry+round12_decode.py+md
echo "[launchB] done $(date)"
