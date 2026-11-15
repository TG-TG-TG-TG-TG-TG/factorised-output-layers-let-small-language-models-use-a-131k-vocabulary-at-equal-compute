## Results files, paper and images

- Results JSON: `runs/round12/round12_results.json` (round 11's, re-run with the seed-1 BLiMP: `runs/round11/round11_results.json`).
- Commands: `runs/round12/commands_extra.txt` (copied below).
- Paper PDF: `paper/main.pdf` (14 pages: body pages 1–9, references page 10, appendices A–F). Source `paper/main.tex`, numbers
  `paper/numbers.tex` (from `paper/make_numbers.py`; 424 macros, 12 unused ones have no value, see below), Markdown export
  `paper/paper.md`, claims table `paper/claims.md` (32 claims).
- Title chosen (both "big vocab for free" and "two-level wins at 128k" hold): *Big Vocabularies for Small Language Models at
  Almost No Head Cost: An Equal-Compute Study of Two-Level Output Layers*.
- Social images (`round12_social.py`), all seven made, each at 1600×900 and 1080×1080, in `runs/round12/social/`:
  `hero_*`, `vocab_*`, `headcost_*`, `knowledge_*`, `schematic_*`, plus the follow-up benchmark's `speed_*` and
  `memory_*`; thread draft `runs/round12/social/thread.md` (7 tweets; tweet 3 now carries the measured training speed-up).
- Follow-up benchmark (2026-11-11, the user's request; reported, no rule pinned): `round12_perf.py` via `r12_perf.sh`,
  raw records `runs/round12/perf.jsonl`, table `runs/round12/perf.md`, paper appendix "Memory and speed". Each
  configuration runs in its own fresh process: configurations sharing a process slowed each other (micro 32 after
  micro 16 ran at about half speed), so the shared-process smoke numbers in `perf_smoke/` are not used. Scoring memory is XLA's compiled estimate, not
  the measured peak: the scoring `peak_gb` is identical at 32 and 64 rows (B 1.83, F128k 1.45 GB), so it is set before
  the call (setup or compile), not by it. Training peaks agree with the estimate to within 0.1 GB.
  The hero has no "flat needs N× the compute" line: the flat curve never reaches the two-level BPB inside the measured FLOPs
  range, and the brief rules out extrapolating.

## Surprises, bugs and deviations

Surprises
- The 131k two-level model is the best model of the project (1.1471 BPB). It beats the 50k two-level model at equal FLOPs even
  with the class-building proxy charged (−0.0047). The flat 131k model is the worst d=384 arm at 6.717e16 (1.2248), worse than
  flat 4k.
- Not pinned, so these are observations: at 131k the two-level model leads flat by 4.85 BLiMP points (74.86 vs 70.01) and
  2.8 LAMBADA points. Its 74.86 is also the highest BLiMP of any model here.
- D2 seed spread is large: the flat bios model stores 712k bits at seed 0 and 617k at seed 1, so the ratio moves from 1.71
  to 2.22. Both clear the pinned 1.2.
- BLiMP seed 1 doubles the near-equal-FLOP gap (+1.03 vs +0.38). The 2-seed mean, +0.70, is still inside the ±1 band, so the
  reading is "no difference".
- LAMBADA accuracy stays below the 50k flat model's (13.2% for H128k vs 14.8%) although its BPB is better. This is consistent with
  the tokenizer effect Part A found (observation, not tested).

Bugs and deviations
- `heads.py`'s docstring gives the rank bounds as d+1 and d+C. `test_heads.py` checks d+2 and d+C+1, i.e. log-prob rank rather
  than logit rank. The paper states the tested bounds and notes the docstring as a documentation bug.
- Footers: the brief says "≤ 31M params". The largest 50k-vocab model has 31.5M, so the footer is computed as "≤ 32M". The 131k
  models are 61–63M, mostly embeddings, and the vocab image says "≤ 63M".
- Round 11's pinned generalisation comparisons put the flat model at 6.39e16 FLOPs (5% fewer). The paper labels them
  "near-equal FLOPs". The D1-matched OOD numbers are reported as observations.
- The three CPU gate OOMs on 131k (vmap fallback gather, per-sample member gather, `jax.random.categorical`'s [N, V] Gumbel
  array) were fixed in `round12_gates.py`; all gates pass. Queue B was restructured so a gate failure cannot cancel the evals.
- The user stopped the GPU at 01:03 (after D1 B384m s1 finished). The F128k run that had just started was voided
  (`runs/round12/void_killed/`), and the queue resumed at 08:51 with `r12_launchC.sh`.
- Unused macros with no value: the Jensen [1k,4k) bucket (tail targets start at rank 8,192, so it is empty), F4k decode (not
  measured), a "proxy charged" variant of the flat verdict (the flat run uses no proxy), and LAMBADA on B384m (not run).
- The H3*-384 seed-1 model has BLiMP only, no OOD BPB. The D1 BPB comes from the logged full-validation loss, like every GPT-2
  row.
- The bpe131072 rows score 23,160,080 of the 23,160,213 val bytes (133 unscored by the byte-aligned evaluator). F128k and
  H128k drop the same bytes. Against the GPT-2 rows the effect is at most ~1e-4 BPB (5.7e-6 of the bytes at <= 20 bits/byte),
  far below the smallest margin (0.0047). Disclosed in the Table 2 caption.
- Part C was not run (optional). Round-12 GPU training time: 3.92 h, plus about 1 h of evals and smokes.
