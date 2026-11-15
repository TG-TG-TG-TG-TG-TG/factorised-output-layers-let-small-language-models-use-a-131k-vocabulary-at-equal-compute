# Round 13 — answers to the reviews (experiments 1 and 2)

Rules pinned in round13_analyze.py's docstring at 21:40 CET on 2026-11-11, before any round-13 run. BPB = bits per byte of the FineWeb-Edu val text. Seed 0 unless stated.

Proxy wsd_tBproxy_50M_s0 (d=128 flat, GPT-2 vocabulary): 55.0M tokens, 2.7729e+15 FLOPs (4.1% of 1x), 3.7 min.

## Experiment 1 — the 50k proxy charged

| point | seed | flat BPB | two-level uncharged | Δ uncharged | two-level charged (FLOPs incl. proxy) | Δ charged |
|---|---|---|---|---|---|---|
| 0.5x | 0 | 1.2229 | 1.1951 | -0.0278 | 1.2003 (1.0001x) | -0.0226 |
| 0.5x | 1 | 1.2240 | 1.1942 | -0.0298 | 1.1995 (1.0001x) | -0.0245 |
| 0.5x | mean | 1.2235 | 1.1947 | -0.0288 | 1.1999 | -0.0236 |
| 1x | 0 | 1.1753 | 1.1607 | -0.0146 | 1.1626 (1.0000x) | -0.0127 |
| 1x | 1 | 1.1763 | 1.1596 | -0.0167 | 1.1616 (1.0000x) | -0.0147 |
| 1x | mean | 1.1758 | 1.1602 | -0.0156 | 1.1621 | -0.0137 |

**Verdict (pinned):** charged 2-seed Δ at 1x = -0.0137 (seeds -0.0127 / -0.0147) → the 50k headline **survives** charging.

Cluster source (reported only): proxy clusters − B_r8 clusters at 575M, 2-seed means: +0.0048 BPB.

## Experiment 2 — gap vs training FLOPs (seed 0)

| point | FLOPs | flat BPB | two-level charged | gap charged | two-level uncharged | gap uncharged | 2-seed gap charged |
|---|---|---|---|---|---|---|---|
| 0.5x | 3.505e+16 | 1.2229 | 1.2003 | -0.0226 | 1.1951 | -0.0278 | -0.0236 |
| 1x | 6.717e+16 | 1.1753 | 1.1626 | -0.0127 | 1.1607 | -0.0146 | -0.0137 |
| 2x | 1.343e+17 | 1.1409 | 1.1352 | -0.0056 | 1.1342 | -0.0067 | - |

Reproducibility: the 2x flat run's 1x branch − wsd_B384m_315M_s0 = -0.00032 BPB.

**Crossover:** line through the three points: +0.0085 BPB per doubling, F* ≈ 3.1x (2.05e+17 FLOPs) — an extrapolation from three points, not a result. Shrinkage is monotone (|gap| falls at both steps).

## Experiment 3 — adaptive softmax baseline (rules pinned 2026-11-12 13:27)

50k, d=384. Δ TL−AS = two-level (charged proxy clusters) − adaptive softmax, seed s vs seed s; Δ AS−flat likewise.

| point | seed | flat | adaptive softmax | two-level charged | two-level uncharged | old clusters (uncharged) | Δ TL−AS | Δ AS−flat |
|---|---|---|---|---|---|---|---|---|
| 0.5x | 0 | 1.2229 | 1.1972 | 1.2003 | 1.1951 | 1.1902 | +0.0032 | -0.0258 |
| 0.5x | 1 | 1.2240 | 1.1980 | 1.1995 | 1.1942 | 1.1892 | +0.0015 | -0.0260 |
| 0.5x | mean | | | | | | +0.0023 | -0.0259 |
| 1x | 0 | 1.1753 | 1.1624 | 1.1626 | 1.1607 | 1.1559 | +0.0001 | -0.0129 |
| 1x | 1 | 1.1763 | 1.1632 | 1.1616 | 1.1596 | 1.1548 | -0.0016 | -0.0131 |
| 1x | mean | | | | | | -0.0007 | -0.0130 |

**Verdict 50k (pinned):** tie (Δ TL−AS = -0.0007).

131k (bpe131072), seed 0 only, byte-aligned BPB:

| point | flat | adaptive softmax | two-level (1x charged, 0.5x uncharged) | Δ TL−AS | Δ AS−flat |
|---|---|---|---|---|---|
| 0.5x | 1.3026 | 1.2052 | 1.1804 | -0.0248 | -0.0974 |
| 1x | 1.2248 | 1.1735 | 1.1512 | -0.0223 | -0.0514 |

**Verdict 131k (pinned rule, one seed):** two-level wins.

## Experiment 4 — 131k, two seeds, best of two adaptive-softmax configurations (pinned 2026-11-13)

FineWeb-Edu byte-aligned BPB at 1x (equal training FLOPs; two-level charged). Configuration 1 = cutoffs (8,192, 24,576), configuration 2 = (8,192, 49,152); both divisor 4.

| seed | flat | adaptive softmax cfg 1 | adaptive softmax cfg 2 | two-level charged | Δ TL − best AS |
|---|---|---|---|---|---|
| 0 | 1.2248 | 1.1735 | 1.1629 | 1.1512 | -0.0117 |
| 1 | - | 1.1738 | 1.1628 | 1.1539 | -0.0089 |

Best adaptive softmax on seed 0: configuration 2. Seed 1 minus the seed-0 winner (configuration 2), reported only: -0.0089.

**Gate (pinned; PASS iff Δ < −0.003 on both seeds):** PASS.

Reported only — 0.5x (two-level uncharged) and the other corpora at 1x:

| seed | point / corpus | adaptive softmax cfg 1 | adaptive softmax cfg 2 | two-level |
|---|---|---|---|---|
| 0 | 0.5x fineweb | 1.2052 | 1.1963 | 1.1804 |
| 0 | 1x wikitext103 | 1.4370 | 1.4292 | 1.4063 |
| 0 | 1x pg19 | 1.7384 | 1.7179 | 1.6834 |
| 1 | 0.5x fineweb | 1.2054 | 1.1964 | 1.1835 |
| 1 | 1x wikitext103 | 1.4395 | 1.4260 | 1.4130 |
| 1 | 1x pg19 | 1.7374 | 1.7219 | 1.6879 |

## Experiment 5 — full-width adaptive softmax at 131k (rule pinned 2026-11-13)

Setup 2 cutoffs (8,192, 49,152); tail bands at width 384 with no projection, untied; everything else as setup 2. FineWeb-Edu byte-aligned BPB at 1x (equal training FLOPs; two-level charged).

| seed | adaptive softmax setup 2 (widths 96 / 24) | adaptive softmax full width | two-level charged | Δ TL − full width | full width − setup 2 |
|---|---|---|---|---|---|
| 0 | 1.1629 | 1.1522 | 1.1512 | -0.0010 | -0.0107 |
| 1 | 1.1628 | 1.1536 | 1.1539 | +0.0004 | -0.0093 |

**Rule (pinned):** TIE (2 seeds).

Reported only — other corpora at 1x: seed 0 wikitext103 1.4166; seed 0 pg19 1.7228; seed 1 wikitext103 1.4204; seed 1 pg19 1.7210

Reported only — 0.5x FineWeb-Edu (two-level uncharged): seed 0 full width 1.1874, two-level 1.1804; seed 1 full width 1.1905, two-level 1.1835

