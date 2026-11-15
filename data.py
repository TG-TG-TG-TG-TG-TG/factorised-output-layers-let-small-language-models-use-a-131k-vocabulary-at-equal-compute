"""Streamed FineWeb-Edu token batches: nothing but a ~10 MB val file touches the disk.

Train documents come from shards 000-012 of sample/10BT and val from shard 013, so the two
never overlap. Order is fixed by the shuffle seed and the same for every arm.
"""
import os
import queue
import threading
import time

import numpy as np

REPO = "HuggingFaceFW/fineweb-edu"
TRAIN_FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(13)]
VAL_FILES = ["sample/10BT/013_00000.parquet"]
EOT = 50256
DATA_DIR = os.path.expanduser("~/lm-data")


def _docs(files, seed, skip=0):
    from datasets import load_dataset
    ds = load_dataset(REPO, data_files=files, split="train", streaming=True)
    ds = ds.select_columns(["text"])
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if skip:
        ds = ds.skip(skip)
    for ex in ds:
        yield ex["text"]


def encoder(source=""):
    """(encode_batch(list of str) -> list of id lists, eot id, vocab bound) for a data source.
    "" = GPT-2 (tiktoken encode_ordinary, EOT 50256); "bpe<V>" = round-11 byte-level BPE (runs/round11/bpe<V>.tok.json,
    no special tokens registered, EOT = V - 1 appended between documents, as for GPT-2)."""
    if not source:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return (lambda texts: enc.encode_ordinary_batch(texts, num_threads=8)), EOT, 50257
    assert source.startswith("bpe"), source
    V = int(source[3:])
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(bpe_path(V))
    assert tk.get_vocab_size() == V - 1
    return (lambda texts: [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)]), V - 1, V


def bpe_path(V):
    """round-11 tokenizers live in runs/round11; round 12's (bpe131072) in runs/round12."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    for r in ("round11", "round12"):
        p = os.path.join(base, r, f"bpe{V}.tok.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"bpe{V}.tok.json")


def id_dtype(vmax):
    """uint16 for every vocabulary up to 65,536 ids (all runs before round 12, unchanged); uint32 above."""
    return np.uint16 if vmax <= 65536 else np.uint32


def token_rows(files, seq_len, seed, doc_batch=256, log=print, source=""):
    """Yield (rows_uint16 (k, seq_len+1), docs_consumed) forever, retrying on network
    errors by restarting the stream and skipping the documents already used."""
    encode, eot, vmax = encoder(source)
    row = seq_len + 1
    dt = id_dtype(vmax)
    buf = np.zeros(0, dt)
    consumed = 0
    attempt = 0
    while True:
        try:
            texts = []
            for text in _docs(files, seed, skip=consumed):
                texts.append(text)
                if len(texts) < doc_batch:
                    continue
                ids = encode(texts)
                consumed += len(texts)
                texts = []
                flat = np.fromiter((t for doc in ids for t in (*doc, eot)), dt)
                assert flat.size == 0 or int(flat.max()) < vmax, "token id out of range for this data source"
                buf = np.concatenate([buf, flat])
                k = len(buf) // row
                if k:
                    yield buf[: k * row].reshape(k, row), consumed
                    buf = buf[k * row:]
                attempt = 0
            log(f"[data] stream exhausted after {consumed} docs; restarting")
            consumed = 0
        except Exception as e:  # network hiccup: resume where we were
            attempt += 1
            wait = min(60, 2 ** attempt)
            log(f"[data] stream error after {consumed} docs ({type(e).__name__}: {e}); retry in {wait}s")
            time.sleep(wait)
            buf = np.zeros(0, dt)  # partial rows are dropped; the doc count stays exact


class Batches:
    """Background producer of (batch_rows, seq_len+1) int32 arrays."""

    def __init__(self, batch_rows, seq_len=1024, seed=0, prefetch=8, log=print, source=""):
        self.q = queue.Queue(maxsize=prefetch)
        self.batch_rows = batch_rows
        self.docs = 0
        self.wait_s = 0.0
        if source.startswith("bios"):  # round 11 E4: synthetic biographies (round11_bios.py), GPT-2 tokens
            from round11_bios import bios_rows
            self._gen = bios_rows(source, seq_len, log=log)
        else:
            self._gen = token_rows(TRAIN_FILES, seq_len, seed, log=log, source=source)
        threading.Thread(target=self._fill, daemon=True).start()

    def _fill(self):
        pend = []
        n = 0
        for rows, docs in self._gen:
            pend.append(rows)
            n += len(rows)
            while n >= self.batch_rows:
                cat = np.concatenate(pend)
                self.q.put((cat[: self.batch_rows].astype(np.int32), docs))
                pend, n = [cat[self.batch_rows:]], n - self.batch_rows

    def next(self):
        t = time.perf_counter()
        b, self.docs = self.q.get()
        self.wait_s = time.perf_counter() - t
        return b


def val_rows(seq_len=1024, n_tokens=5_000_000, source=""):
    """Fixed held-out rows from the val shard, cached as one small uint16 file. Round 11: source "bpe<V>" = the
    same val documents re-tokenised (monitoring only; the official BPB is round11_bytes.py), "bios<N>" = round11_bios."""
    if source.startswith("bios"):
        from round11_bios import val_rows_bios
        return val_rows_bios(source, seq_len)
    if source:
        return _val_rows_retok(seq_len, source)
    path = os.path.join(DATA_DIR, f"val_{seq_len}_{n_tokens}.bin")
    row = seq_len + 1
    n_rows = n_tokens // seq_len
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        got = []
        n = 0
        for rows, _ in token_rows(VAL_FILES, seq_len, seed=None):
            got.append(rows)
            n += len(rows)
            if n >= n_rows:
                break
        np.concatenate(got)[:n_rows].tofile(path + ".tmp")
        os.replace(path + ".tmp", path)
    return np.fromfile(path, np.uint16).reshape(-1, row)[:n_rows].astype(np.int32)


def _val_rows_retok(seq_len, source):
    """The GPT-2 val stream decoded back to its documents (split at EOT), re-encoded by source, EOT-joined, cut into
    rows of seq_len + 1 (all whole rows). Cached in DATA_DIR."""
    path = os.path.join(DATA_DIR, f"val_{source}_{seq_len}.bin")
    row = seq_len + 1
    if not os.path.exists(path):
        import tiktoken
        g = tiktoken.get_encoding("gpt2")
        flat = val_rows(seq_len).reshape(-1)
        docs = [g.decode(p[p != EOT].tolist()) for p in np.split(flat, np.flatnonzero(flat == EOT))]
        encode, eot, vmax = encoder(source)
        ids = encode([d for d in docs if d])
        out = np.fromiter((t for doc in ids for t in (*doc, eot)), id_dtype(vmax))
        assert int(out.max()) < vmax
        out[: len(out) // row * row].tofile(path + ".tmp")
        os.replace(path + ".tmp", path)
    return np.fromfile(path, id_dtype(int(source[3:]))).reshape(-1, row).astype(np.int32)


if __name__ == "__main__":
    v = val_rows()
    print("val rows", v.shape, "tokens", v[:, 1:].size)
    b = Batches(64)
    t0 = time.time()
    tok = 0
    for i in range(40):
        tok += b.next()[:, 1:].size
        if i == 4:
            t0, tok = time.time(), 0
    print(f"stream throughput {tok / (time.time() - t0):,.0f} tok/s, docs {b.docs}")
