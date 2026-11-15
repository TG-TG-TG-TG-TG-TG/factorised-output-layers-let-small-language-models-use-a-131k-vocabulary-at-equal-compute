"""Round 12 Part A: LAMBADA sanity check (rules pinned in round12_analyze.py). GPU, minutes.

A1 whole-passage BPB ([EOT] + encode(passage), every token scored); A2 last-word bits/byte (round 11 E3 definition:
context and " word" encoded separately) split by whether the word appears earlier in the context (whole word, exact,
case-sensitive); A3 ten passages (rng 12 from the 100 with the largest |F8k - H3*| last-word difference) with token splits
and per-token NLL under both tokenizers.

python round12_lambada.py [labels]  -> runs/round12/partA_lambada.json, partA_examples.md
"""
import json
import math
import os
import re
import sys

import numpy as np

import round11_eval as E

OUT = os.path.join(E.HERE, "runs", "round12")
LN2 = math.log(2)
LABELS = ["B384", "H3s384_575M", "F8k384_631M", "F4k384_700M"]


def texts():
    from datasets import load_dataset
    t = load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"]
    return [(x[: x.rfind(" ")], x[x.rfind(" "):]) for x in t]


def appears_earlier(ctx, w):
    return re.search(r"(?<!\w)" + re.escape(w.strip()) + r"(?!\w)", ctx) is not None


def score(m, seqs, bs=64):
    """per-target NLL (nats) list for each sequence [EOT] + ids."""
    out = []
    for i in range(0, len(seqs), bs):
        part = seqs[i:i + bs]
        X, L = E.pad_seqs(part)
        nll = m.nll(np.concatenate([X, np.zeros((len(X), 1), np.int64)], 1))
        for j, s in enumerate(part):
            out.append(np.asarray(nll[j, : len(s) - 1], np.float64))
    return out


def piece_strs(m, ids):
    if m.gpt2:
        import tiktoken
        g = tiktoken.get_encoding("gpt2")
        return [g.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids]
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(E.bpe_path(m.cfg.data) if hasattr(E, "bpe_path") else
                             os.path.join(E.HERE, "runs", "round11", f"{m.cfg.data}.tok.json"))
    return [tk.decode([i]) for i in ids]


def main(labels):
    os.makedirs(OUT, exist_ok=True)
    split = texts()
    earlier = np.array([appears_earlier(c, w) for c, w in split])
    res = {"n": len(split), "n_earlier": int(earlier.sum()), "n_rest": int((~earlier).sum()), "models": {}}
    per = {}
    for lab in labels:
        m = E.Model(lab)
        whole = [[m.eot] + ids for ids in m.encode([c + w for c, w in split])]
        nw = score(m, whole)
        bits_whole = sum(float(x.sum()) for x in nw) / LN2
        nbytes = sum(len((c + w).encode("utf-8")) for c, w in split)
        ctx = m.encode([c for c, _ in split])
        tgt = m.encode([w for _, w in split])
        nl = score(m, [[m.eot] + c + t for c, t in zip(ctx, tgt)])
        lw = np.array([float(x[-len(t):].sum()) / LN2 / len(w.encode("utf-8")) for x, t, (_, w) in zip(nl, tgt, split)])
        per[lab] = {"lw": lw, "tgt": tgt, "nll_t": [x[-len(t):] / LN2 for x, t in zip(nl, tgt)], "m": m}
        res["models"][lab] = {"bpb_whole": bits_whole / nbytes, "bytes_whole": nbytes, "word_bpb": float(lw.mean()),
                              "word_bpb_earlier": float(lw[earlier].mean()), "word_bpb_rest": float(lw[~earlier].mean())}
        print(f"[A] {lab:14s} whole-passage BPB {bits_whole / nbytes:.4f}  last word {lw.mean():.4f} "
              f"(earlier {lw[earlier].mean():.4f} n={earlier.sum()}, rest {lw[~earlier].mean():.4f} n={(~earlier).sum()})", flush=True)
    a, b = "F8k384_631M", "H3s384_575M"
    if a in per and b in per:
        R = res["models"]
        d_whole = R[a]["bpb_whole"] - R[b]["bpb_whole"]
        adv_e = R[b]["word_bpb_earlier"] - R[a]["word_bpb_earlier"]
        adv_r = R[b]["word_bpb_rest"] - R[a]["word_bpb_rest"]
        if d_whole <= -0.15:
            v = "evaluator bug"
        elif abs(d_whole) <= 0.05 and adv_e > 0 and adv_e >= 2 * adv_r:
            v = "real effect"
        else:
            v = "unclear"
        res["verdict"] = {"d_whole_F8k_minus_H3s": d_whole, "adv_earlier": adv_e, "adv_rest": adv_r, "verdict": v}
        print(f"[A] verdict: whole-passage F8k - H3* {d_whole:+.4f}; last-word advantage earlier {adv_e:+.4f} / rest {adv_r:+.4f} -> {v}")
        diff = per[a]["lw"] - per[b]["lw"]
        top = np.argsort(-np.abs(diff))[:100]
        pick = np.random.default_rng(12).choice(top, 10, replace=False)
        L = ["# Round 12 Part A: ten LAMBADA passages with large F8k vs H3* last-word differences", "",
             "Drawn with np.random.default_rng(12) from the 100 passages with the largest |F8k - H3*| last-word bits/byte. "
             "Per-token NLL in bits; `|` separates tokens.", ""]
        for i in sorted(pick):
            c, w = split[i]
            L += [f"## passage {i} (appears earlier: {'yes' if earlier[i] else 'no'}; F8k {per[a]['lw'][i]:.3f} vs H3* {per[b]['lw'][i]:.3f} bits/byte)",
                  "", "> …" + c[-300:].replace("\n", " ") + " **" + w + "**", ""]
            for lab in (b, a):
                pr = per[lab]
                toks = piece_strs(pr["m"], pr["tgt"][i])
                L.append(f"- {lab}: " + " | ".join(f"`{t!r}` {n:.2f}" for t, n in zip(toks, pr["nll_t"][i])))
            L.append("")
        open(os.path.join(OUT, "partA_examples.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "partA_lambada.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else LABELS)
