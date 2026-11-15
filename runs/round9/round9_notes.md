## Notes and deviations (hand-written, 2026-11-07)

**Code.** New heads are in `round9_heads.py`, selected by `cfg.head == "hier9"`. Round-8 code paths are unchanged:
`model.py` only dispatches on the new head value (the pre-edit copy is `runs/round9/model_before_r9.py.bak`).
`round8_gates.py` was rerun after the edit (`runs/round9/r8gates_recheck/`) and diffed against round 8's gates.txt.
Training and eval use round 8's path: the gather form in fp32, remat'd 4096-token chunks, no custom kernels.
One general head covers every arm. The top-level softmax runs over [U[freq]·h ; W_c·h + bc], with W_c = Vc^T or
ū + R. Frequent targets have no within part; their gather points at a valid slot and is zeroed with `jnp.where`.

**Gates** (`runs/round9/gates.txt`, CPU). All OK for tH2tm, tH2a, tH2atm and tH2u.
- Each was checked at init, and with B_r8's trained backbone and embedding plus random head params.
- The reference is built independently: full h·Uᵀ, class means through a dense one-hot matmul, frequent logits
  taken as columns of the full logits, and segment-op LSEs.
- Worst per-token NLL difference: 9.5e-6 (trained backbone). Worst gradient difference relative to max |grad|: 9.1e-7.
- d=384 arms: accounting and backbone bit-identity only (`gates_d384.json`), because the head code is the same.
  A first attempt through `round9_gates.py` hung for 10 min waiting on the HF data stream (3% CPU) and was killed.
  It had nothing to do with the head code.

**Counts and the frequency split** (`round9_freq.py`, `round9_classes.py`).
- Counts are target occurrences over exactly the 20,142 × 64 rows (1,320M targets) the 1200M runs consume.
- Top 4,096 = 79.8% of training targets, matching the brief's ~80%. 242 real tokens never occur.
- Tail: 1,024 balanced k-means clusters, capacity 46. 893 clusters are full; the smallest has 13 tokens.
- Unique training tokens: 9.49M docs in shards 000–012 (parquet footers) × 1,030.6 tokens/doc ≈ 9.78B.
  Part D needs 5.28B including its final decay, so it uses no repeats.

**Round-8 §2b reporting slip, and what it does to H-c.**
- Round 8's §2b table says its Δ@step columns are 5-point smoothed, but the code printed single-step values.
  The round-8 summary is left as it was.
- The brief's "+0.035 at 52M" is that single-step value. Smoothed, it is +0.060.
- The paired train gap keeps growing before the first branch: +0.12 at 65M, +0.143 at the 75M branch step.
- So at 75M, H2 is already +0.143 behind in the stable phase, and decay adds only +0.025 (val Δ +0.168).
- Most of the gap is present before decay, not created by it. This is what A2 shows; Part C was not triggered.

**Seeds.** No round-9 arm finished within ±0.010 of B at 1200M (closest: H2atm, +0.052), so seed 1 was not run.
CB (round 8, −0.0039) is inside ±0.010, but it is an unchanged round-8 reference, not a round-9 arm, so it gets no rerun.
The optional tH2u (untied output embedding) was not run: it passed its gates, but the time went to Parts D–F.
The floor is the default 0.005. Part D's 1200M branch repeats Part B's H2atm result to 0.0001 (4.1746 vs 4.1747).

**GPU time.** 3.34 h of training in round 9: B384 37 min, H2tm 16, H2a 21, H2atm 21, H2atm 4800M 87, H2atm-384 17.
Smokes, Part A and Part F come on top, a few minutes each. Every run returned rc 0, and every data hash matched
round 8's first-3-batches reference. Part D logged no stream restarts, so it used no repeated data.

**Bugs and problems.** No gate failed and no bug was found in round-9 code:
- Part A's B total reproduces the logged metric to 2e-5.
- The Part F exact-greedy path matched brute force on 100% of 4,096 trained-model val positions.

Two problems were found, neither in the head code:
1. *Round-8 reporting bug (above).* The §2b Δ@step columns were single-step values, not the stated 5-point smoothed
   values. This changes the brief's premise: smoothed, "+0.035 at 52M" is +0.060.
2. *Infrastructure.* The HF data stream hung once (in `round9_gates.py` for the d=384 arms). The process was killed;
   its gates_H2tm384_H2a384_H2atm384.json had already been written, and is the file the summary uses.

**Part E method, added after the result.** The pinned equal-wall-clock rule interpolates B384's loss at H2*-384's wall
time. That was undefined: H2atm-384 finished 300M (16.4 min) before B384 reached its first branch (150M, 18.1 min).
After seeing this, I added the assumption-free comparison: H2atm-384 at 300M vs B384 at 150M, which had more
wall-clock. It assumes a decayed branch's loss falls with tokens. The d=384 "regime win" rests on this bound
(−0.117 against a 0.010 threshold).

## Surprises

1. **The tied-mean fix (H-a) helped early, then faded.** H2tm beat H2 by 0.007–0.024 at 75M–600M but finished at
   +0.001 at 1200M. A1 had already put only 21% of H2's gap in the within part.
2. **The frequent-token fix (H-b) fully closes the frequent buckets, and the gap that remains is the rare tail.**
   - H2atm vs B per bucket: top-256 −0.004, 256–1024 −0.007, 1024–4096 +0.019, 4096–16384 +0.287, 16384+ +0.205.
   - Weighted by bucket share, the two tail buckets account for all of the remaining +0.052 (≈ 0.040 + 0.013).
   - The 4096–16384 bucket is slightly *worse* than in round-8 H2 (+0.258). Fewer, larger tail clusters
     (1,024 clusters of ≤ 46) now serve a harder slice of the vocabulary.
   - So the next lever is the tail's cluster structure, not the frequent tokens or the decay.
3. **Tied means add a little on top of the adaptive split**: H2atm − H2a = −0.0046 at 1200M, inside the 0.005 floor.
4. **Matched compute at d=128 is roughly a tie.** H2atm at B's 72 min is −0.0055 (raw; smoke-normalised the same),
   and at equal training FLOPs −0.0016. Both are near the floor. H2atm needs 3.5× the tokens to match B, and only
   gets past B at 4800M (4.1122, −0.0106), which takes 82 min against B's 72.
5. **d=384 is a different regime.** The equal-token gap is the same (+0.053). But H2atm-384 runs 2.17× faster, and at
   300M one doubling of tokens is worth 0.17 nats. In less wall-clock than B384's 150M branch, it is 0.117 lower.
   This is a bound, because the curves do not overlap in wall-clock. It is not an interpolation.
6. **Part F: exact greedy is not faster at the tiny size.**
   - Tiny: +5.7% vs flat greedy (the head is a small share of each step, and the top-3 expansion adds work).
   - 600M config: −7.7%.
   - The fallback fires on 0.54% of steps for the trained H2atm, and the result is 100% exact.
