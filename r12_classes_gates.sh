#!/bin/bash
# 2026-11-10: after the bpe131072 proxy finishes (GPU), build H128k's classes and run the hier gates on CPU (queue A's
# "require gates" step waits on gates_hier.json).
ulimit -c 0
source "/path/to/repo/env.sh"
while [ ! -f runs/round12/wsd_tF128k_proxy_50M_s0.DONE ]; do sleep 15; done
JAX_PLATFORMS=cpu python -u round12_classes.py && JAX_PLATFORMS=cpu python -u round12_gates.py hier
