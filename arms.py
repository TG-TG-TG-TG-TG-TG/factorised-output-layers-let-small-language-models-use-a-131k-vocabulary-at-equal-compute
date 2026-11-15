"""Every experiment arm as a Config (d=384 "B" shape and the d=128 tiny shape, 6 unique blocks)."""
from dataclasses import replace

from model import Config

BASE = Config()

ARMS = {
    # flat tied head, 6 unique blocks, d=384
    "B": replace(BASE, arch="unique", n_unique=6),
}

# tiny shape (d=128, H=2, f=512, 6 unique blocks, same recipe)
TINY = replace(BASE, d=128, f=512, H=2)
ARMS["tB"] = replace(TINY, arch="unique", n_unique=6)
# looped tiny model: the 6 blocks applied twice in cycle order, per-copy norms/alpha, per-neuron FFN gates
# (the body of round 10's Rloop)
ARMS["tCgcyc"] = replace(TINY, arch="cycle", gate=True)
# round 8 (2026-11-06): LM-head arms on the tiny B backbone (round8_heads.py; C = 1024 classes from B_r8's U)
ARMS["tH2"] = replace(ARMS["tB"], head="hier", head_classes="kmeans")
ARMS["tH2r"] = replace(ARMS["tB"], head="hier", head_classes="random")
ARMS["tCB"] = replace(ARMS["tB"], head="cb", head_classes="kmeans")
ARMS["tH2w"] = replace(ARMS["tB"], head="hier", head_classes="kmeans", head_wide=512)

# round 9 (2026-11-07): two-level head fixes (round9_heads.py, head "hier9"); K = 4096 singletons + 1024 tail clusters
ARMS["tH2tm"] = replace(ARMS["tB"], head="hier9", head_classes="kmeans", head_tm=True)
ARMS["tH2a"] = replace(ARMS["tB"], head="hier9", head_classes="adapt", head_adapt=4096)
ARMS["tH2atm"] = replace(ARMS["tH2a"], head_tm=True)
ARMS["tH2u"] = replace(ARMS["tB"], head="hier9", head_classes="kmeans", head_untied=True)
# part E: the d=384 B shape with each candidate head; classes come from the d=128 B_r8
ARMS["H2_384"] = replace(ARMS["B"], head="hier", head_classes="kmeans")
for k in ("tH2tm", "tH2a", "tH2atm"):
    c = ARMS[k]
    ARMS[k[1:] + "384"] = replace(ARMS["B"], head="hier9", head_classes=c.head_classes, head_adapt=c.head_adapt, head_tm=c.head_tm)


def has_shared(cfg):
    return cfg.arch != "unique"

# round 10 (2026-11-07), part A: tail fixes = H2atm with K singletons and C' tied-mean tail clusters
# (classes from round10_classes.py; K = 4096 / C' = 1024 is H2atm itself)
for nm, K, C in (("tH3k8", 8192, 1024), ("tH3c4", 4096, 4096), ("tH3k8c4", 8192, 4096)):
    ARMS[nm] = replace(ARMS["tH2atm"], head_adapt=K, head_C=C)

# round 10 parts B/C/D (2026-11-08): head = H3* = tH3k8c4 (lower two-seed mean at 1200M, rule pinned in round10_analyze.py)
_h3 = ARMS["tH3k8c4"]
_hk = dict(head=_h3.head, head_classes=_h3.head_classes, head_adapt=_h3.head_adapt, head_C=_h3.head_C, head_tm=_h3.head_tm)
ARMS["Rloop"] = replace(ARMS["tCgcyc"], n_loops=4, **_hk)      # part C: 6 blocks x 4 passes, cycle order, tCgcyc gates
ARMS["Rdeep"] = replace(ARMS["tB"], n_unique=24, **_hk)        # part C: 24 unique blocks
ARMS["Rwide"] = replace(ARMS["tB"], d=256, H=4, f=1024, **_hk)  # part C: d 256, H 4, f 1024
ARMS["RB24"] = replace(ARMS["tB"], n_unique=24)                # part C optional: flat head, 24 blocks
ARMS["tD1"] = replace(_h3, ngram=2)                            # part D: + hashed bigram input table
ARMS["tD2"] = replace(_h3, ngram=3)                            # part D: + bigram and trigram tables
ARMS["H3s384"] = replace(ARMS["B"], **_hk)                     # part B: d=384 B shape with the H3* head
ARMS["Rwide1200"] = ARMS["Rwide"]  # part C rule: best arm > 0.02 below B at 600M -> rerun to 1200M (own name: wsd_Rwide_s0 is the 600M run)

# round 11 (2026-11-09). Part C: H3k8c4 with the top-M tail clusters re-scored exactly (round11_heads.py)
for _m in (0, 16, 64):
    ARMS[f"tH3rs{_m}"] = replace(ARMS["tH3k8c4"], head_rescore=_m)
    ARMS[f"H3rs{_m}_384"] = replace(ARMS["H3s384"], head_rescore=_m)
# Part B: small-vocabulary control, d=384 B shape, flat tied head, FineWeb-Edu re-tokenised with round11_bpe.py
ARMS["F8k384"] = replace(ARMS["B"], vocab=8192, data="bpe8192")
ARMS["F4k384"] = replace(ARMS["B"], vocab=4096, data="bpe4096")
# Part D (optional): d=512, H 8, f 2048, 6 unique blocks, two-level head (H3k8c4 classes; head_rescore set at launch)
ARMS["H3s512"] = replace(ARMS["H3s384"], d=512, H=8, f=2048)
for _m in (16, 64):
    ARMS[f"H3rs{_m}_512"] = replace(ARMS["H3s512"], head_rescore=_m)
# E4: synthetic biographies (round11_bios.py), d=128 shape, from scratch on bios only
for _n in (20000, 100000):
    ARMS[f"bioB_{_n}"] = replace(ARMS["tB"], data=f"bios{_n}")
    ARMS[f"bioH_{_n}"] = replace(ARMS["tH3k8c4"], data=f"bios{_n}")

# round 12 (2026-11-10): big vocabulary at equal FLOPs (bpe131072 = round12_bpe.py). d=384 B shape; the two-level arm uses
# the H3* head (K 8,192 singletons + C' 4,096 tied-mean clusters) with round-12 class files (runs/round12/v131072_*).
ARMS["F128k384"] = replace(ARMS["B"], vocab=131072, data="bpe131072")
ARMS["H128k384"] = replace(ARMS["H3s384"], vocab=131072, data="bpe131072")
ARMS["tF128k"] = replace(ARMS["tB"], vocab=131072, data="bpe131072")   # flat d=128 proxy: its embedding seeds H128k's clusters

# round 13 (2026-11-11): the H3*-384 head with its tail clusters rebuilt from a cheap d=128 flat proxy whose FLOPs are
# charged (wsd_tBproxy_50M_s0, 50M stable tokens + 10% decay; round13_classes.py -> runs/round13/adapt_*8192*.npy)
ARMS["H3p384"] = replace(ARMS["H3s384"], head_classes="adapt13")

# round 13 experiment 3 (2026-11-12): adaptive softmax baseline (Grave et al. 2017; round13_heads.py) on the d=384 B shape:
# the two-level head's K = 8,192 frequent singletons (tied), then two frequency-ordered tail bands with untied d/4, d/16 layers
ARMS["AS384"] = replace(ARMS["B"], head="adasm", head_adapt=8192)
ARMS["AS128k384"] = replace(ARMS["F128k384"], head="adasm", head_adapt=8192)
# same model, faster implementation: each tail band only for the tokens whose target is in it (gated equal; round13_heads)
ARMS["AS128k384s"] = replace(ARMS["AS128k384"], head_sparse=True)
ARMS["AS384s"] = replace(ARMS["AS384"], head_sparse=True)   # gates / timing only
# round 13 experiment 4 (2026-11-13): the second adaptive-softmax configuration at 131k -- the same shortlist and divisor,
# the second cutoff moved 24,576 -> 49,152 so the narrow d/16 band holds 3.05% of training targets, the 50k setting's
# share (3.06%), where adaptive softmax tied the two-level head (pinned in round13_analyze.py before any run)
ARMS["AS128k384c49"] = replace(ARMS["AS128k384s"], head_cuts=(8192, 49152))
ARMS["AS128k384c49d"] = replace(ARMS["AS128k384c49"], head_sparse=False)   # dense twin, gates only
# round 13 experiment 5 (2026-11-13): setup 2 with full-width tail bands, no projection, untied (only the width changes)
ARMS["AS128k384c49w"] = replace(ARMS["AS128k384c49"], head_full=True)
ARMS["AS128k384c49wd"] = replace(ARMS["AS128k384c49w"], head_sparse=False)   # dense twin, gates only
