# Round 11 tweets

Hand-written drafts. Every number comes from `runs/round11/round11_results.json`.

Three Round 11 verdicts are positive. Each has its own chart:
- Part B: two-level beats an 8k vocabulary at equal FLOPs.
- E1: holds out of domain.
- E4: no knowledge penalty, and more tokens store more knowledge.

The charts are `social/bpb_vs_flops_1600x900.png` and `social/knowledge_bits_1600x900.png` (each also as `_1080x1080.png`). The BPB chart compares different vocabularies at the same width, and says so on the chart.

## Chart tweet (social/bpb_vs_flops_1600x900.png)

Is a small vocabulary a cheaper fix for the softmax cost than a two-level output head? No. At the same training FLOPs (6.7e16, d=384, one RTX 5070 Ti):
- the two-level head reaches 1.156 bits/byte;
- a flat head over an 8k BPE vocab reaches 1.187;
- full 50k softmax reaches 1.179 (at 5% fewer FLOPs).

One seed per arm.

## Chart tweet (social/knowledge_bits_1600x900.png)

Does a cheaper output head forget facts? On synthetic biographies at 100 exposures, the two-level head stores 1.11× (20k people) and 1.02× (100k people) the bits of full softmax. Given the same FLOPs it gets 2.4× the passes and stores 1.71× the bits. One seed per arm.

## Thread (5 posts)

1/ Small-model log, round 11. The question: is a two-level output head both cheap and as good as full softmax? Here, 8,192 frequent tokens get their own logit and the rest share 4,096 clusters. FineWeb-Edu, d=128 / 384 / 512 GPTs, one RTX 5070 Ti.

2/ Cheap: yes. At equal training FLOPs (d=384) it beats full softmax by −0.023 bits/byte on FineWeb-Edu, −0.031 on WikiText-103 and −0.025 on PG-19. It beats a flat head over an 8k BPE vocab by −0.032. Batch-1 decode runs at 0.72× the flat head's time per token.

3/ As good: not at equal tokens. It is still ~+0.03 nats behind, all of it on tokens ranked ≥ 4,096. Re-scoring the top 16 tail clusters exactly gains only −0.005 (two seeds), because the right cluster is in the top 16 just 43% of the time. BLiMP shows no difference (+0.4 points).

4/ Knowledge (synthetic bios, 100 exposures): no penalty. It stores 1.02–1.11× full softmax's bits, and 1.71× when the saved FLOPs go into more passes. Copying rare tokens in context is better at equal FLOPs and worse at equal tokens.

5/ Surprise: the 8k- and 4k-vocab models top LAMBADA (16.8% / 17.3% vs 14.2%) despite worse FineWeb bits/byte. Caveats: mostly one seed (the seed-1 checks are listed in the summary), 8–31M params, one consumer GPU. @ProjectTEST-AI
