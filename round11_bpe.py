"""Round 11 Part B.1: byte-level BPE tokenizers (8,192 and 4,096 ids, EOT included) trained on training-stream text only.

Training text: the first >= 2.0e9 UTF-8 bytes of the training document stream (data.TRAIN_FILES = FineWeb-Edu shards
000-012, shuffle seed 0, the order every run sees), streamed and never written to disk. The val shard (013) is never
read here except for the gates.
Tokenizer: HF `tokenizers` BPE, ByteLevel pre-tokenizer with GPT-2's regex (use_regex=True, add_prefix_space=False),
ByteLevel decoder, no normalizer, 256-byte initial alphabet. The trainer learns V-1 ids; the EOT id is V-1 and is NOT a
registered special token (literal "<|endoftext|>" text inside a document stays ordinary text), it is appended between
documents by the data pipeline exactly like tiktoken's encode_ordinary path does for GPT-2.

Gates (written to runs/round11/bpe<V>.json):
  1. decode(encode(doc)) == doc for every val document (the docs behind data.val_rows' 5M GPT-2 tokens).
  2. every id < V - 1 (EOT never produced by encode).
  3. val bytes / token over the val documents (the shared-bytes count is in round11_bytes.py).

python round11_bpe.py train 8192 [4096]   -> runs/round11/bpe8192.tok.json (+ .json report)
python round11_bpe.py gate 8192
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round11")
TRAIN_BYTES = 2.0e9


def tok_path(V):
    return os.path.join(OUT, f"bpe{V}.tok.json")


def load(V):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tok_path(V))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def train_texts(stats):
    from data import TRAIN_FILES, _docs
    n = 0
    for i, t in enumerate(_docs(TRAIN_FILES, seed=0)):
        b = len(t.encode("utf-8"))
        n += b
        stats["docs"], stats["bytes"] = i + 1, n
        yield t
        if n >= TRAIN_BYTES:
            return


def train(Vs):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    os.makedirs(OUT, exist_ok=True)
    # one pass over the stream per tokenizer would re-download; keep the texts in RAM once (~2 GB of str)
    stats = {}
    t0 = time.time()
    texts = list(train_texts(stats))
    print(f"[bpe] {stats['docs']:,} train docs, {stats['bytes'] / 1e9:.3f} GB in {time.time() - t0:.0f} s", flush=True)
    for V in Vs:
        t1 = time.time()
        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=V - 1, min_frequency=2, show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tk.train_from_iterator(texts, trainer=tr, length=len(texts))
        assert tk.get_vocab_size() == V - 1, tk.get_vocab_size()
        tk.save(tok_path(V))
        rep = {"V": V, "eot_id": V - 1, "learned_ids": V - 1, "train_docs": stats["docs"], "train_bytes": stats["bytes"],
               "train_source": "data.TRAIN_FILES (FineWeb-Edu sample/10BT shards 000-012), shuffle seed 0, first docs >= 2e9 bytes",
               "sha256": sha(tok_path(V)), "train_s": time.time() - t1}
        json.dump(rep, open(os.path.join(OUT, f"bpe{V}.json"), "w"), indent=1)
        print(f"[bpe] V={V} trained in {rep['train_s']:.0f} s, sha256 {rep['sha256'][:16]}", flush=True)
        gate(V)


def val_docs():
    """The val documents behind data.val_rows(1024) (5M GPT-2 tokens): the GPT-2 stream split at EOT. The first and
    last pieces are whole/partial docs as the row cut leaves them; both still have to round-trip."""
    import tiktoken
    from data import EOT, val_rows
    enc = tiktoken.get_encoding("gpt2")
    flat = val_rows(1024).reshape(-1)
    cut = np.flatnonzero(flat == EOT)
    pieces = np.split(flat, cut)
    docs = [enc.decode(p[p != EOT].tolist()) for p in pieces]
    return [d for d in docs if d]


def gate(V):
    tk = load(V)
    docs = val_docs()
    enc = tk.encode_batch(docs, add_special_tokens=False)
    bad = [i for i, (d, e) in enumerate(zip(docs, enc)) if tk.decode(e.ids) != d]
    mx = max(max(e.ids) for e in enc if e.ids)
    ntok = sum(len(e.ids) for e in enc)
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    p = os.path.join(OUT, f"bpe{V}.json")
    rep = json.load(open(p)) if os.path.exists(p) else {"V": V}
    rep.update({"gate_roundtrip_docs": len(docs), "gate_roundtrip_failures": len(bad), "gate_max_id": int(mx),
                "gate_ok": not bad and mx < V - 1, "val_docs_bytes": nbytes, "val_docs_tokens": ntok,
                "val_bytes_per_token_docs": nbytes / ntok})
    json.dump(rep, open(p, "w"), indent=1)
    print(f"[bpe] V={V} gate: {len(docs)} val docs, round-trip failures {len(bad)}, max id {mx}, "
          f"{nbytes / ntok:.3f} bytes/token -> {'OK' if rep['gate_ok'] else 'FAIL'}", flush=True)
    if not rep["gate_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train([int(v) for v in sys.argv[2:]])
    else:
        for v in sys.argv[2:]:
            gate(int(v))
