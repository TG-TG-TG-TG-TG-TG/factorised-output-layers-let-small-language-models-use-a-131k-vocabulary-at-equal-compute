# Factorised Output Layers Let Small Language Models Use a 131k Vocabulary at Equal Compute

Code, logs and results for the paper (Jakub Michałek, independent researcher).

- **Paper (PDF):** [`paper.pdf`](paper.pdf) (LaTeX source in `paper/`; Markdown version: [`paper/paper.md`](paper/paper.md); every claim with its
  source number: [`paper/claims.md`](paper/claims.md)).
- **Main results (cluster cost charged, adaptive-softmax baseline, compute scaling):**
  [`runs/round13/round13_summary.md`](runs/round13/round13_summary.md), machine-readable in `runs/round13/round13_results.json`;
  design ablation in [`runs/round13/ablation.md`](runs/round13/ablation.md).
- **Earlier results (downstream evaluations, vocabulary study):** [`runs/round12/round12_summary.md`](runs/round12/round12_summary.md) and
  [`runs/round11/round11_summary.md`](runs/round11/round11_summary.md); machine-readable in `runs/round12/round12_results.json`
  and `runs/round11/round11_results.json`.
- **Memory and speed benchmark:** [`runs/round12/perf.md`](runs/round12/perf.md) (raw records `runs/round12/perf.jsonl`).
- **Social images and thread draft:** [`runs/round13/social/`](runs/round13/social/) (made by `python round13_social.py`;
  [`thread.md`](runs/round13/social/thread.md)). The images in `runs/round12/social/` are older and show the uncharged
  −0.020 headline.

## Layout

| path | what |
|---|---|
| `model.py`, `train.py`, `data.py`, `arms.py` | the GPT model, training loop, data pipeline, experiment arms |
| `round8_heads.py`, `round9_heads.py`, `round11_heads.py` | the output heads (the two-level head is `hier9`, with exact re-scoring in round 11) |
| `src/heads/` | an early pilot of the head code (kept for the record) |
| `round<N>_*.py`, `r<N>*.sh` | the scripts of each experiment round (queues, gates, evaluation, analysis, social images) |
| `runs/round<N>/` | raw training/eval logs (`*.jsonl`, `*.out`), results JSONs, summaries and `commands_extra.txt` (every relaunch) |
| `paper/` | LaTeX source, figures, tables, and the scripts that generate the numbers and figures |
| `requirements-freeze.txt` | exact Python packages (Python 3.12.3, JAX 0.9.0.1 with CUDA 12) |

## Reproducing

Runs were done with JAX on one RTX 5070 Ti (16 GB) under WSL2 (Ubuntu). `env.sh` sets the environment (edit the paths for
your machine). The exact commands, including every relaunch, are in `runs/round11/commands_extra.txt`,
`runs/round12/commands_extra.txt` and `runs/round13/commands_extra.txt`.

From the logs in `runs/`, the numbers, tables and figures of the paper are regenerated with:

```bash
python round11_analyze.py
python round12_analyze.py
python round13_analyze.py   # in WSL, next to the round-13 logs
python round13_ablation.py
python paper/make_numbers.py
python paper/make_figures.py
cd paper && tectonic main.tex && python export_md.py
```

Model checkpoints and the tokenized datasets are not included (they are large); the training data is FineWeb-Edu; out-of-domain and probe evaluations use WikiText-103, PG-19, BLiMP and LAMBADA.

## About this repository

It was cut from a larger working folder that also held an unrelated earlier project. What differs from that folder:

- Left out: the earlier project's code, logs and results; pre-change backup copies of source files (`runs/round10/backup/`,
  `runs/round13/bak/`); and `round10_compat.py` with its output `runs/round10/compat.json`, a check of the round-10 code
  against that project's models. `runs/round10/chainA.out` starts at its second step for the same reason. From
  `runs/round5/` only the two tiny-model baselines that the round-8 and round-10 analyses compare against are kept
  (`wsd_tB_s0`, `wsd_tCgcyc_s0`).
- `model.py` and `arms.py` keep only the code paths and arms of this study. For every arm in `arms.py` the trimmed code gives
  bit-identical initial weights, the same loss on a fixed batch, the same FLOP count and the same parameter count as the
  code the runs used (checked on CPU before publishing).
- In the logged configs (`config` in the `*.jsonl` run logs) the keys of the earlier project's model options were removed;
  none had any effect on the runs here, and every logged config key is now a field of the published `Config`. Every other
  logged value is unchanged (checked record by record).
- Local paths are replaced with `/path/to/repo` (the repository) and `~/lm-data` (data and checkpoints; `DATA_DIR` in
  `data.py`, `HF_DATASETS_CACHE` in `env.sh`).
- The regeneration commands above, run in this repository, reproduce every summary, results file and the paper's numbers
  unchanged; the figures differ only in their embedded dates and IDs. The `*.DONE` files mark finished runs (the analysis
  scripts skip a run without one); each holds the run's final record.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Paper, figures and images: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Support this work

Everything here was trained on one RTX 5070 Ti at home. The paper's limits
(runs up to 2× the main budget, 1–2 seeds) are compute limits. If this was
useful to you and you'd like to help fund more experiments:

- Ko-fi: https://ko-fi.com/testtm
- Patreon: https://www.patreon.com/cw/Testtm
