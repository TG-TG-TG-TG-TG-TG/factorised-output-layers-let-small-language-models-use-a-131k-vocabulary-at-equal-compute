# Round 10 tweets (hand-written drafts; every number is from runs/round10/round10_results.json)

Two round-10 verdicts are positive: d=384 confirmed (`social/d384_wall_1600x900.png`, also `_1080x1080.png`) and input memory helps, via lazy AdamW by the user's decision (`social/input_memory_1600x900.png`, also `_1080x1080.png`).

## Chart tweet (social/d384_wall_1600x900.png)

Same GPU time, lower loss: a two-level output head lets a 30M-param GPT reach 3.712 val loss in 32 min on one RTX 5070 Ti, vs 3.785 for full softmax in 35 min (−0.073 nats). Catch: at equal tokens it is +0.037 worse. The win is speed. One seed.

## Chart tweet (social/input_memory_1600x900.png)

A 2^20-row hashed bigram table added to a tiny GPT's input: −0.150 val loss at 1.3B tokens, same training speed on one RTX 5070 Ti (needs lazy row-sparse AdamW; dense AdamW gave −0.136 but ran 20% slower). 134M lookup params, ~0 extra FLOPs. One seed.

## Thread (5 posts)

1/ Small-model experiment log, round 10 (FineWeb, one RTX 5070 Ti). In a tiny GPT the 50k-word softmax eats most of the compute. A two-level head (8,192 frequent tokens get their own logit, the rest share 4,096 clusters) cuts the head to 0.25× the FLOPs.

2/ At d=384 (30M params) that trains 1.9× faster. In ~the same time it sees 575M tokens instead of 300M and ends at 3.712 vs 3.785 val loss: −0.073 nats at equal wall-clock (chart). At equal tokens it is still +0.037 behind.

3/ Where the gap lives: almost all of it is the top-level choice of a rare-token cluster (target ranks ≥ 4,096). Picking the word inside the cluster costs ~0. More frequent singletons (8,192 vs 4,096) closed ~30% of it; more clusters did not help.

4/ A hashed bigram input table (2^20 rows) gave −0.150 at full speed with lazy sparse AdamW (dense: −0.136, 20% slower). Not claimed: a wider backbone (d=256) gave −0.32, but at 1.12× FLOPs (cap was 1.10×) and 2.4× params.

5/ Caveats: one seed per arm (seed-1 checks moved results by 0.001–0.002), 8–31M-param models (not counting n-gram lookup tables), val loss on 5M held-out tokens, one consumer GPU. Lazy sparse n-gram updates: −0.15/−0.16 at full speed. @ProjectTEST-AI
