"""Zero-shot multiple choice by continuation log-likelihood (lm-eval-harness style prompts).

Uses only the locally cached HF datasets. acc = argmax of summed logprob, acc_norm = argmax
of logprob per continuation byte. The first N examples of each split are scored.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from model import token_logprobs

CACHE = os.path.expanduser("~/.cache/huggingface/datasets")
MAXLEN = 256
N_PER_TASK = 2000


def _load(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, cache_dir=CACHE, **kw)


def _hellaswag():
    import re
    def clean(t):
        t = t.strip().replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", t).replace("  ", " ")
    for ex in _load("Rowan/hellaswag", split="validation"):
        ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        yield clean(ex["activity_label"] + ": " + ctx), [" " + clean(e) for e in ex["endings"]], int(ex["label"])


def _piqa():
    for ex in _load("baber/piqa", split="validation"):
        yield f"Question: {ex['goal']}\nAnswer:", [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])


def _arc_easy():
    for ex in _load("allenai/ai2_arc", "ARC-Easy", split="test"):
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        yield (f"Question: {ex['question']}\nAnswer:", [" " + t for t in ex["choices"]["text"]],
               labels.index(ex["answerKey"]))


def _sciq():
    for ex in _load("allenai/sciq", split="test"):
        ch = [ex["distractor1"], ex["distractor2"], ex["distractor3"], ex["correct_answer"]]
        yield f"{ex['support'].strip()}\nQuestion: {ex['question']}\nAnswer:", [" " + c for c in ch], 3


TASKS = {"hellaswag": _hellaswag, "piqa": _piqa, "arc_easy": _arc_easy, "sciq": _sciq}


def score_task(params, cfg, examples, batch=32):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    reqs, meta = [], []  # one request per (example, choice)
    for ei, (ctx, choices, gold) in enumerate(examples):
        c_ids = enc.encode_ordinary(ctx)
        for ci, ch in enumerate(choices):
            k_ids = enc.encode_ordinary(ch)
            seq = (c_ids + k_ids)[-(MAXLEN + 1):]
            reqs.append((seq, min(len(k_ids), len(seq) - 1)))
            meta.append((ei, ci, gold, len(ch)))
    fn = jax.jit(lambda p, i, t: token_logprobs(p, i, t, cfg))
    scores = np.zeros(len(reqs))
    for s in range(0, len(reqs), batch):
        chunk = reqs[s:s + batch]
        inp = np.zeros((batch, MAXLEN), np.int32)
        tgt = np.zeros((batch, MAXLEN), np.int32)
        mask = np.zeros((batch, MAXLEN), np.float32)
        for r, (seq, nk) in enumerate(chunk):
            n = len(seq) - 1
            inp[r, :n], tgt[r, :n] = seq[:-1], seq[1:]
            mask[r, n - nk:n] = 1.0
        lp = np.asarray(fn(params, jnp.asarray(inp), jnp.asarray(tgt)))
        scores[s:s + len(chunk)] = (lp * mask).sum(1)[: len(chunk)]
    n_ex = meta[-1][0] + 1
    best = np.full(n_ex, -np.inf); best_n = np.full(n_ex, -np.inf)
    pick = np.zeros(n_ex, int); pick_n = np.zeros(n_ex, int); gold = np.zeros(n_ex, int)
    for (ei, ci, g, nbytes), sc in zip(meta, scores):
        gold[ei] = g
        if sc > best[ei]:
            best[ei], pick[ei] = sc, ci
        if sc / nbytes > best_n[ei]:
            best_n[ei], pick_n[ei] = sc / nbytes, ci
    acc, acc_n = float((pick == gold).mean()), float((pick_n == gold).mean())
    se = lambda p: 1.96 * float(np.sqrt(p * (1 - p) / n_ex))
    return {"n": int(n_ex), "acc": acc, "acc_ci": se(acc), "acc_norm": acc_n, "acc_norm_ci": se(acc_n)}


def run_all(params, cfg, n=N_PER_TASK):
    out = {}
    for name, gen in TASKS.items():
        ex = []
        for e in gen():
            ex.append(e)
            if len(ex) >= n:
                break
        out[name] = score_task(params, cfg, ex)
        print(f"  zeroshot {name}: acc {out[name]['acc']:.3f} acc_norm {out[name]['acc_norm']:.3f} (n={out[name]['n']})", flush=True)
    return out


if __name__ == "__main__":
    import sys
    from arms import ARMS
    from model import init_params
    cfg = ARMS["B"]
    print(run_all(init_params(cfg, jax.random.PRNGKey(0)), cfg, n=int(sys.argv[1]) if len(sys.argv) > 1 else 200))
    os._exit(0)
