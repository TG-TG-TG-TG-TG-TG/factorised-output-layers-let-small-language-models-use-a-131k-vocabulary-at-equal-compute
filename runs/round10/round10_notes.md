## 9. Notes (hand-written)

### Decisions and deviations

- **H3\* tie (2026-11-07 22:06).** Seed 0 at 1200M: H3k8c4 4.1552, H3k8 4.1597. They are 0.0045 apart, which is inside the ±0.010 seed band, and the pick decides Part C:
  - With an H3k8c4 head, every Part-C arm is about 1.12× B's FLOPs, over the ≤ 1.1× limit.
  - With an H3k8 head, the arms are about 1.08×.

  So seed 1 runs for both arms. The rule was pinned before any seed 1 ran (`round10_analyze.py` docstring): H3\* = the lower mean of seeds 0 and 1, even when the means differ by less than the seed floor. The FLOPs rule plays no part in the pick.

  **Result (23:37):** H3k8 gave 4.1597 on seed 0 and 4.1618 on seed 1 (mean 4.1607). H3k8c4 gave 4.1552 and 4.1544 (mean 4.1548). **H3\* = H3k8c4**, 0.0059 ahead on the two-seed mean, which is larger than either arm's seed spread (0.0021 and 0.0008). So every Part-C arm carries the H3k8c4 head at about 1.12× B's FLOPs, over the ≤ 1.1× limit. That will be reported as it stands; the block counts are not retuned.
- **Tail fixed = no, without seeds.** The best Part-A arm (H3k8c4) is +0.0324 vs B_r8 at 1200M, 0.0124 above the +0.020 threshold. That is outside the ±0.010 band, so no seed 1 was needed for this verdict. Seed 1 was run for the H3\* pick only.
- **Palette.** `validate_palette.js --pairs all` passes only four hues plus grey: blue, yellow, aqua and red. Adding magenta, brown, violet, orange or green fails the normal-vision floor. So no figure shows more than four coloured arms. In the headline, colour marks the Part and the best arm is blue; markers and direct labels name the arms.

- **Input memory verdict: user decision (2026-11-08).** After seeing the results, the user ruled that lazy D1 counts: it is −0.1505 vs H3\* at 1200M and 1.01× H3\*'s full-run tok/s, so both D thresholds are met. The verdict is therefore yes (via lazy AdamW). The literal dense-only reading (no: 0.80× tok/s) stays in the summary. This decision came after the results, not before; it is a post-hoc call. `USER_COUNTS_LAZY` in `round10_analyze.py` records it, and the `input_memory` social charts follow from it.

### Surprises

- **K = 8192 helps more than C' = 4096.** At 1200M:
  - K 4096 → 8192 (H3k8): 0.015 better than H2atm.
  - C' 1024 → 4096 (H3c4): no better than H2atm (+0.0514 vs +0.0518).
  - Both (H3k8c4): only 0.0045 beyond H3k8.

  The tail gap is mostly the singleton/cluster split, not the cluster count.
- **The H3k8c4 advantage is front-loaded.** It is +0.026 at 75M and stays around +0.033 from 150M to 1200M. None of the arms closes the gap as training goes on.
- **Wall-clock.** H3c4 trains fastest (19.9 min own wall, the same as H2atm), even though its 4096 clusters make its top level the widest.

- **The remaining gap sits in the top level, not within clusters.** For every arm, Δwithin is no more than 0.004 in absolute value, and the bucket breakdown is in `buckets_partA.json`. Nearly all of Δtotal is Δtop, in target ranks ≥ 4096:
  - Ranks [4096, 16384): +0.17 to +0.29 per token, the fewer singletons the worse.
  - Ranks ≥ 16384: +0.20 to +0.27.

  The head prices the tail cluster as a whole badly, while choosing the word inside the cluster costs nothing. Two other points:
  - **The two tail buckets react in opposite directions.** Raising K from 4096 to 8192 halves the [4096, 16384) gap, because those tokens become singletons. But the ≥ 16384 gap grows slightly (+0.205 → +0.210 for H3k8, +0.250 for H3k8c4).
  - **More clusters pull the other way.** They make the ≥ 16384 top-level gap worse (+0.27 vs +0.20) and make within-cluster loss slightly better (−0.018).
- **The tiny backbone is the bottleneck, not the head.** At 600M:
  - R-wide (d=256): −0.322 vs B.
  - R-deep (24 blocks): −0.194 vs B.

  Both cost 1.123× B's training FLOPs. B is 7.6M params, of which only 1.18M are blocks; R-wide has 2.4× B's total params and R-deep 1.5×. Taking the head compute off the output and giving it to blocks pays off far more than any head change in Part A. The literal verdict is still "no", because 1.123× is over the brief's 1.1× cap. That overrun comes from H3\* = H3k8c4, whose head costs 0.245× of B's.
- **R-loop (6 blocks × 4 passes) gains nothing over two passes.** It is +0.0055 vs B and +0.036 vs round-5 tCgcyc (6 × 2 passes, flat head), at 600M. Measured against its own head, R-loop is −0.028 vs H3k8c4 at 600M; tCgcyc's two passes gave −0.031 vs B. So four passes buy about what two did. R-loop's grad-norm max is 1.73× B's, under the 3× rule, so it got no LR 1e-3 rerun.
- **Decode at batch 1 follows the number of block applications.** R-loop and R-deep (24 applications) are about 3.5× B's ms/token. R-wide (6 applications at d=256) is 1.05× and D1 is 1.00×. In this round's arms, depth costs latency and width doesn't.
- **Hashed bigram inputs are large and cheap in FLOPs, but dense AdamW is slow.**
  - D1 is −0.136 vs H3\* at 1200M (−0.103 vs B).
  - Dense AdamW over the 2^20 × 128 table runs at 0.80× H3\*'s full-run tok/s (0.84× in the smoke), which fails the 0.85× bar.
  - Lazy row-sparse AdamW runs at 1.01× and scores better still, −0.150 vs H3\*.

  Lazy beating dense by 0.015 is plausible, not a bug. `decay_mask` puts weight decay on every 2-D leaf, the n-gram tables included. Dense AdamW decays all 2^20 rows every step, plus momentum drift on untouched rows; lazy decays only the rows a batch touches. Gate D4 (lazy == optax for touched rows) passed. The leak check (`causal.txt`) shows no future-token path.
- **D2 (bigram + trigram tables).** The brief's D2 gate is loss only (D1 more than 0.010 below H3\*); D1 dense cleared it by 0.136, so D2 ran. Results at 1200M:
  - D2 dense 4.0283, *worse* than D1 dense (4.0194), at 0.59× H3\*'s full-run tok/s and 8.34 GB logged memory (`wsd_D2_s0.out`; the smoke's 5.25 GB is at smoke batch timing) (the two 2^20 × 128 tables plus dense Adam state).
  - D2 lazy 3.9951, the best Part-D run (−0.160 vs H3\*, −0.128 vs B) at 0.93× tok/s.
  - The lazy-minus-dense gap grows with table count (D1 −0.015, D2 −0.033), which fits the weight-decay explanation below: dense AdamW decays twice as many untouched rows.
  - Per-bucket (`buckets_partD.json`): every arm is below H3\* in every rank bucket; the gain is largest at target ranks ≥ 1024 (about −0.17 to −0.27, peaking in [4096, 16384)), smallest for the 256 most frequent tokens (about −0.07). Almost all of it is top-level; within-cluster loss improves by only 0.01–0.04.
- **Part B: the head wins on wall-clock only.** H3\*-384 is +0.037 at equal tokens (300M) and −0.073 at about equal wall (575M in 32.1 min vs B384's 300M in 35.2 min). The smoke speed ratio was 1.92×; over the full runs, 300M took 17.0 min against B384's 35.2, which is 2.07×.
- **Seed spread is tiny.** H3k8c4 gave 4.1552 on seed 0 and 4.1544 on seed 1 (|Δ| 0.0008).

### Bugs

- **`round10_gates_cd.py` crashed on RB24 (fixed 2026-11-08 12:40).** It ran the two-level-head gates 1 and 2 on every arm. RB24 has a flat head and no `head` params, so it raised KeyError: 'head'. The head gates now run only for arms that have a head. Every gate before the crash had passed, and the rerun passed all of them (ALL GATES OK, `gates_cd_*.txt`). No model code changed.
- **The lazy D1 smoke is 0.044 behind the dense one at step 200** (6.0309 vs 5.9872). That is expected rather than a bug: dense AdamW keeps moving rows that were touched earlier, through their decaying m, and lazy updates do not. Whether lazy is used is decided only by the brief's 0.003 rule at 1200M.
- Everything else: none found. One warning: `round10_buckets.py` logs XLA's "constant folding took long" warning for a scatter-add over the class tables. It only slows compilation, and the B self-check matches the logged loss.
