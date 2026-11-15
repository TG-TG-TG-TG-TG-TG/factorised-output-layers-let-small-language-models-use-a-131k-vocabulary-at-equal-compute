"""Round 12 Part B.1: byte-level BPE with V = 131,072 (EOT = id 131,071), round11_bpe.py's recipe unchanged (first 2.0 GB
of the training stream, shuffle seed 0, GPT-2 pre-tokenisation, min_frequency 2), written to runs/round12/; then the
round-trip gate on all val docs (runs/round12/bpe131072.json: sha256, val bytes/token).

python round12_bpe.py train 131072  |  python round12_bpe.py gate 131072
"""
import os
import sys

import round11_bpe as B

B.OUT = os.path.join(B.HERE, "runs", "round12")

if __name__ == "__main__":
    if sys.argv[1] == "train":
        B.train([int(v) for v in sys.argv[2:]])
    else:
        for v in sys.argv[2:]:
            B.gate(int(v))
