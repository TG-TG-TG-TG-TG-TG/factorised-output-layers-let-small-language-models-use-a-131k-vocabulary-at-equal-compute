## Notes and deviations (hand-written, 2026-11-06)

**H2 / H2r / H2w loss path: `heads.hier_nll` (gather), not `heads.hier_nll_grouped`.**
- The brief names `hier_nll_grouped` as the training loss. On this JAX 0.9 / RTX 5070 Ti (sm_120) build, `jax.lax.ragged_dot` lowers to a dense product over all classes per token. With 16 384 tokens per micro-batch that is `f32[16384,1024,128]` (8.6 GB), and the first H2 smoke run aborted in XLA autotuning (`runs/round8/smoke/smoke_tH2.out` of the first smoke pass).
  - On CPU the same lowering pushed the gates script to a 22 GB peak; it was OOM-killed on the informational `lm_loss` line, now computed on 4 rows.
  - Even if it fitted, a dense lowering costs about the flat head's FLOPs, so H2 would lose its speed advantage.
- So the hier arms train and evaluate with `heads.hier_nll` (unchanged math), wrapped in remat'd 4096-token chunks (`round8_heads.head_loss_sum`).
- Gates 1–2 show that `hier_nll`, `hier_nll_grouped` and −log p_full[y] agree to 1.9e-6, and that dL/dU agrees to 1.2e-7. The wiring gate checks `lm_loss` against both.
- Recipe, data, LR and eval semantics are unchanged.

**Classes are a bootstrap.**
- `classes_kmeans1024.npy` is `balanced_kmeans` on the embedding of the finished B_r8 run (1200M tokens). The H2 runs therefore start with classes learned by a full flat run.
- Capacity is 50: 937 classes are full, the smallest has 10 tokens, and none is empty.
- The random control has the same size histogram.

**Baseline is B_r8, not round-5 tB.** Round 5 kept no tB checkpoint, so B was rerun.
- B_r8 matches round-5 tB at every branch to within 0.0004 (1200M: 4.1228 vs 4.1232), but not bit for bit.
- B_r8's full-run tok/s (308.5k) is slightly depressed: CPU dry runs competed with its data loader early on. Speed comparisons use the back-to-back smoke runs.

**Pad rows.** B's softmax covers 50 304 rows: the GPT-2 vocab of 50 257 plus 47 pad rows. H2 and CB cover the 50 257 real tokens. On the trained B the 47 pad rows hold a mean probability of 2.9e-6 (`diag_B.json`), so the comparison is not affected.

**Smoke "val nan" is benign.** `train.py` evaluates on the full val set only when a run reaches its full step count, and prints NaN otherwise (`train.py:343`).

**Wrapper bug, fixed before the counted H2 run: tracer leak.**
- `round8_heads.tables` was `lru_cache`d and returned jnp arrays (`heads.class_tables` returns jnp). The first call came inside the training-step scan, so the cached arrays were tracers.
- The first branch eval (a different jit) then crashed with `UnexpectedTracerError` at step 1144. The log is kept in `runs/round8/interrupted/wsd_H2_s0_try1_tracerleak.*`.
- The fix caches only concrete numpy tables (built under `jax.ensure_compile_time_eval`) and converts them per call.
- A scan-then-separate-jit test passes for tH2 and tH2w (`runs/round8/leak_test.out`). The 200-step smoke runs never reach a branch, so they missed this.
- Numerics are unaffected: the crash came before any eval.

**Disk incident (not a result).** Aborted WSL python processes write 1–4 GB core dumps to `%LOCALAPPDATA%\Temp\wsl-crashes` on C:, which has only ~12.6 GB free. Five dumps from my aborted tests filled C: and were deleted. All launch scripts now set `ulimit -c 0`.

**Observations on H2 vs the CPU pilot.**
- **The pilot's early advantage does reproduce here, then reverses** (section 2b, paired train loss on identical batches). H2 is −0.18 at 6.6M tokens and −0.06 at 13M, close to the pilot's −0.069 at 2M. It crosses B at ~27.5M tokens (step 420) and is behind from then on.
- At the decayed branches H2 is +0.168 at 75M, narrowing to +0.091 at 1200M. So this is a crossover, not a bug-like disagreement with the pilot. Training is smooth: no spikes, and the grad-norm max is 1.42 vs B's 1.85.
- H2's tied U grows much larger than B's (σ_max 273 vs 197; σ_min 13.6 vs 4.5). The within-class softmax is invariant to a per-class shift of the word logits, so the word embeddings' scale and the class-mean directions are not pinned by the output loss.
- This is a plausible mechanism for the gap, since U is also the input embedding, but it is untested.
- Random classes are far worse (H2r 4.3974 vs H2 4.2137 at 1200M), so the k-means structure matters a lot.
- **Speed:** H2 trains 4.3× faster than B in the smoke runs (1.34M vs 314k tok/s). The full run took 16.5 min against B's 78.5.
- **Wall-clock matching (section 2b, not a brief metric):** in H2's 984 s, B gets through 256M stable tokens. B's branch-decayed loss at that point is 4.3185, versus H2's 4.2137, so H2 is ahead by 0.10 at equal wall-clock even though it is behind by 0.09 at equal tokens. H2r is still behind (+0.07).
- H2r (1.79M tok/s) is faster than H2 (1.34M) in smoke, at identical FLOPs. A likely cause is the scatter-add in the gather backward: k-means classes put frequent tokens' members in the same rows, so concurrent updates collide.

**CB: slightly better but slower, and not kept.**
- CB beats B at every branch (−0.012 at 75M down to −0.004 at 1200M), and also in paired train loss throughout (section 2b).
- It fails the keep rule on size as well as seeds: −0.0039 is above −2·floor = −0.010, and it misses the −0.005 trigger for a seed-1 run, so no seed 1 was run. One seed also cannot satisfy the "2 seeds" condition.
- Its extra cost is mostly memory-bound: the [n, C] → [n, V] class-bias gather runs in fp32. That gives 0.72× B's training tok/s (104 min wall time) for 1.02× the FLOPs.

**Decode (Step 7).**
- **Tiny proxy:** batch-1 decode is dominated by per-step overhead. The flat head adds only 0.007 ms/tok to the 0.136 ms backbone, so hier_sample gains nothing there (0.148 vs 0.143 ms/tok).
- **600M config:** the flat head is 0.25 of 2.50 ms/tok. hier_sample brings the head to 0.022 ms and cuts the whole model by 9.1%.
- **hier_greedy, random weights:** it needs the 3rd class at all 256 of 256 positions and is unresolved at all 256, as expected for a flat p(c|h). Even so it is −2.1% vs flat greedy.
- **hier_greedy, trained H2:** the 3rd class is needed at 17.2% of positions, and 11.6% stay unresolved after 3 classes. Its token equals the exact argmax at 97.4% of positions (100% of the resolved ones). The exact host greedy expands 1.82 classes on average, 21 at most.

**No d=384 proposal (brief 8.9).** H2 is "worse" by the rule (+0.091 at 1200M against a 0.010 threshold), so the conditional proposal does not apply. If the two-level head is revisited, two cheaper questions come first: whether the gap comes from the tied U's unpinned within-class shift (try untying U for the word level, or adding a class-mean penalty), and whether the wall-clock advantage holds on a matched-time schedule.
