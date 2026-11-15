"""Round 11 evaluation library and the eval-only tasks: byte-aligned BPB (Part B, E1), BLiMP (E2), LAMBADA (E3),
in-context copying (E5). GPU, minutes. Rules are pinned in round11_analyze.py's docstring.

Models (label: arm, checkpoint in ~/lm-data/ckpt) are in MODELS; a label whose checkpoint is missing is skipped.
Per-token NLL uses the training paths: flat head = bf16 h @ bf16 E^T with fp32 logits over cfg.vocab (as lm_loss),
two-level heads = round9_heads.hier9_nll / round11_heads.nll_tokens in fp32. Greedy / argmax uses the full next-token
distribution: flat = softmax over the real ids (50,257 for GPT-2, V for BPE), two-level = the reference full-vocabulary
log-probs (round9_heads.ref_logprobs_full / round11_heads.ref_logprobs_full, gated against the fast path).

Byte-aligned BPB on a corpus given as GPT-2 rows of 1,025 tokens (row = context token + 1,024 scored targets):
  GPT-2 models: sum NLL over row[1:] / (ln 2 x sum UTF-8 bytes of row[1:]) (EOT = 0 bytes, its NLL counts).
  other tokenizers: per row, the scored text = the bytes of row[1:] split at EOT; each piece is encoded by the model's
  tokenizer (EOT pieces become its EOT id); the context = the row's first GPT-2 token's bytes (decoded with "replace").
  A scored piece that starts (row start) or ends (row end) inside a UTF-8 character loses those 1-3 bytes to the
  context / to nothing: they are counted as unscored bytes and reported. Sequences longer than 1,025 tokens are scored in
  windows of 1,025: the first from position 0, each next one starting at min(covered - 511, n - 1,025), scoring only new
  targets (>= 512 tokens of in-row context).

python round11_eval.py corpora                      # build + cache the E1 corpora (GPT-2 rows) in ~/lm-data/eval
python round11_eval.py bpb  [labels]                # Part B / E1 -> runs/round11/eval_bpb.jsonl
python round11_eval.py validate                     # GPT-2-mode check of the byte-aligned path on B384 / H3s384_300M
python round11_eval.py blimp [labels]               # E2 -> runs/round11/eval_blimp.jsonl
python round11_eval.py lambada [labels]             # E3 -> runs/round11/eval_lambada.jsonl
python round11_eval.py copy [labels]                # E5 -> runs/round11/eval_copy.jsonl
"""
import functools
from dataclasses import replace
import json
import math
import os
import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from arms import ARMS
from data import DATA_DIR, EOT, encoder, val_rows
from model import forward
from round8_heads import load_params
import round9_heads as R9
import round11_heads as R11

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "round11")
EV = os.path.join(DATA_DIR, "eval")
F32, BF = jnp.float32, jnp.bfloat16
LN2 = math.log(2)
MODELS = {
    "B384": ("B", "wsd_B384_300M_s0"),
    "H3s384_300M": ("H3s384", "wsd_H3s384_300M_s0"),
    "H3s384_575M": ("H3s384", "wsd_H3s384_575M_s0"),
    "F8k384_329M": ("F8k384", "wsd_F8k384_631M_s0_branch329M"),
    "F8k384_631M": ("F8k384", "wsd_F8k384_631M_s0"),
    "F4k384_365M": ("F4k384", "wsd_F4k384_700M_s0_branch365M"),
    "F4k384_700M": ("F4k384", "wsd_F4k384_700M_s0"),
    "H3rs16_384_575M": ("H3rs16_384", "wsd_H3rs16_384_575M_s0"),
    "H3rs64_384_575M": ("H3rs64_384", "wsd_H3rs64_384_575M_s0"),
    "B_r8": ("tB", "wsd_B_r8_s0"),
    "H3k8c4": ("tH3k8c4", "wsd_H3k8c4_s0"),
    "H3rs16": ("tH3rs16", "wsd_H3rs16_s0"),
    "H3rs64": ("tH3rs64", "wsd_H3rs64_s0"),
    # seed 1 (E2 seed rule: the BLiMP Δs are within ±1 point of the 1-point threshold)
    "B384_s1": ("B", "wsd_B384_300M_s1"),
    "H3s384_300M_s1": ("H3s384", "wsd_H3s384_575M_s1_branch300M"),
    "H3s384_575M_s1": ("H3s384", "wsd_H3s384_575M_s1"),
    "B_r8_s1": ("tB", "wsd_B_s1"),
    "H3k8c4_s1": ("tH3k8c4", "wsd_H3k8c4_s1"),
    "H3rs16_s1": ("tH3rs16", "wsd_H3rs16_s1"),
}
RANK = np.load(os.path.join(HERE, "runs", "round9", "token_rank.npy"))
EDGES = [0, 256, 1024, 4096, 16384, 50257]
BNAMES = ["[0,256)", "[256,1024)", "[1024,4096)", "[4096,16384)", "[16384,V)"]


def have(label):
    return os.path.exists(os.path.join(DATA_DIR, "ckpt", MODELS[label][1] + ".npz"))


class Model:
    def __init__(self, label):
        self.label = label
        arm, ck = MODELS[label]
        self.cfg = replace(ARMS[arm], attn_impl="cudnn")
        self.params = load_params(self.cfg, ck)
        enc, self.eot, self.vreal = encoder(self.cfg.data if not self.cfg.data.startswith("bios") else "")
        self.encode = enc
        self.gpt2 = not self.cfg.data or self.cfg.data.startswith("bios")
        self.jit()

    def jit(self):
        self._nll = jax.jit(functools.partial(_nll_rows, cfg=self.cfg))
        self._lp = jax.jit(functools.partial(_full_logprobs, cfg=self.cfg, vreal=self.vreal))
        self._h = jax.jit(lambda p, x: forward(p, x, self.cfg))
        self._am = jax.jit(functools.partial(_argmax_lp, cfg=self.cfg, vreal=self.vreal))

    def nll(self, rows):
        """rows int [B, T+1] -> per-token NLL [B, T] (numpy fp32)."""
        return np.asarray(self._nll(self.params, jnp.asarray(rows, jnp.int32)))

    def argmax_lp(self, rows, pos, tgt, P=2048):
        """at (row, position) pairs pos [n, 2]: (argmax of the full next-token distribution, log p(tgt)) as numpy [n].
        One forward per call; positions padded to a multiple of P (fixed shapes, no recompiles); all on the device."""
        h = self._h(self.params, jnp.asarray(rows, jnp.int32))
        n = len(pos)
        npad = -n % P
        pos = np.concatenate([pos, np.zeros((npad, 2), pos.dtype)]) if npad else pos
        tg = np.concatenate([tgt, np.zeros(npad, tgt.dtype)]) if npad else tgt
        am, lp = [], []
        for q in range(0, len(pos), P):
            hs = h[jnp.asarray(pos[q:q + P, 0]), jnp.asarray(pos[q:q + P, 1])]
            a_, l_ = self._am(self.params, hs, jnp.asarray(tg[q:q + P], jnp.int32))
            am.append(np.asarray(a_))
            lp.append(np.asarray(l_))
        return np.concatenate(am)[:n], np.concatenate(lp)[:n]

    def logprobs_at(self, rows, pos):
        """full next-token log-probs [n, vreal] at (row, position) pairs pos [n, 2] (position = index of the input token)."""
        h = self._h(self.params, jnp.asarray(rows, jnp.int32))
        hs = h[jnp.asarray(pos[:, 0]), jnp.asarray(pos[:, 1])]
        return self._lp(self.params, hs)


def _nll_rows(params, rows, cfg):
    inp, tgt = rows[:, :-1], rows[:, 1:]
    h = forward(params, inp, cfg)
    B, T, d = h.shape
    hf, y = h.reshape(-1, d), tgt.reshape(-1)
    n = hf.shape[0]
    ch = 1024 if (cfg.head == "hier9" and cfg.head_rescore) or cfg.vocab > 65536 else 4096  # round 12: 131k vocab -> 1,024
    npad = -n % ch
    hf, y = jnp.pad(hf, ((0, npad), (0, 0))), jnp.pad(y, (0, npad))
    assert cfg.head in ("", "hier9", "adasm"), cfg.head  # round 13: no silent flat fallback
    if cfg.head == "adasm":  # round 13: adaptive softmax (round13_heads.py), fp32
        import round13_heads as R13
        f = lambda a: R13.nll_tokens(a[0].astype(F32), a[1], params, cfg)
    elif cfg.head == "hier9":
        tb = R9.tables(cfg)
        slot, tail, grp, members, mask, pos, _, _ = tb
        Wt, bt = R9.top_matrix(params, cfg, tb)
        Uw = R9.word_matrix(params, cfg)
        if cfg.head_rescore:
            f = lambda a: R11.nll_tokens(a[0].astype(F32), a[1], Uw, Wt, bt, params["head"]["ac"], slot, tail, grp, members, mask, pos, cfg.head_adapt, cfg.head_rescore)[0]
        else:
            f = lambda a: R9.hier9_nll(a[0].astype(F32), a[1], Uw, Wt, bt, slot, tail, grp, members, mask, pos)
    else:
        E = params["emb"].astype(cfg.cdt)

        def f(a):
            z = jnp.einsum("nd,vd->nv", a[0].astype(cfg.cdt), E, preferred_element_type=F32)
            return jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, a[1][:, None], -1)[:, 0]
    out = jax.lax.map(f, (hf.reshape(-1, ch, d), y.reshape(-1, ch)))
    return out.reshape(-1)[:n].reshape(B, T)


def _full_logprobs(params, h, cfg, vreal):
    h = h.astype(F32)
    assert cfg.head in ("", "hier9", "adasm"), cfg.head  # round 13
    if cfg.head == "adasm":
        import round13_heads as R13
        return R13.ref_logprobs_full(params, h, cfg)
    if cfg.head == "hier9":
        if cfg.head_rescore:
            return R11.ref_logprobs_full(params, h, cfg)[0]
        return R9.ref_logprobs_full(params, h, cfg)
    z = jnp.einsum("nd,vd->nv", h.astype(cfg.cdt), params["emb"].astype(cfg.cdt), preferred_element_type=F32)
    return jax.nn.log_softmax(z, -1)[:, :vreal]   # normalised over the padded vocab, as in training


def _argmax_lp(params, h, y, cfg, vreal, ch=512):
    def f(a):
        lp = _full_logprobs(params, a[0], cfg, vreal)
        return jnp.argmax(lp, -1), jnp.take_along_axis(lp, a[1][:, None], 1)[:, 0]
    n, d = h.shape
    am, lp = jax.lax.map(f, (h.reshape(-1, ch, d), y.reshape(-1, ch)))
    return am.reshape(n), lp.reshape(n)


def batched(fn, X, bs):
    return np.concatenate([fn(X[i:i + bs]) for i in range(0, len(X), bs)])


# ---------------------------------------------------------------- corpora (GPT-2 rows of 1,025)
import tiktoken
G2 = tiktoken.get_encoding("gpt2")
NB = np.array([0 if t == EOT else len(G2.decode_single_token_bytes(t)) for t in range(G2.n_vocab)], np.int64)
TB = [b"" if t == EOT else G2.decode_single_token_bytes(t) for t in range(G2.n_vocab)]


def rows_from_docs(docs, row=1025, min_tokens=None):
    flat = []
    n = 0
    for d in docs:
        ids = G2.encode_ordinary(d) + [EOT]
        flat += ids
        n += len(ids)
        if min_tokens and n >= min_tokens + row:
            break
    flat = np.array(flat, np.int64)
    return flat[: len(flat) // row * row].reshape(-1, row)


def wikitext_docs():
    from datasets import load_dataset
    lines = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
    docs, cur = [], []
    for ln in lines:
        if re.match(r"^ = [^=].* = \n$", ln) and cur:
            docs.append("".join(cur).strip("\n"))
            cur = []
        cur.append(ln)
    if cur:
        docs.append("".join(cur).strip("\n"))
    return [d for d in docs if d]


def pg19_docs():
    from datasets import load_dataset
    for ex in load_dataset("emozilla/pg19-test", split="test", streaming=True):
        yield ex["text"]


def corpus(name):
    os.makedirs(EV, exist_ok=True)
    p = os.path.join(EV, f"{name}_rows.npy")
    if not os.path.exists(p):
        if name == "fineweb":
            r = val_rows(1024).astype(np.int64)
        elif name == "wikitext103":
            r = rows_from_docs(wikitext_docs())
        elif name == "pg19":
            r = rows_from_docs(pg19_docs(), min_tokens=1_000_000)
        np.save(p, r)
    return np.load(p)


# ---------------------------------------------------------------- byte-aligned scoring
def valid_utf8_span(b):
    """(lead, body, trail): lead = leading continuation bytes, trail = an incomplete final character, body decodes strictly."""
    i = 0
    while i < len(b) and i < 3 and (b[i] & 0xC0) == 0x80:
        i += 1
    j = len(b)
    k = j - 1
    while k >= i and k >= j - 4 and (b[k] & 0xC0) == 0x80:
        k -= 1
    if k >= i:
        c = b[k]
        need = 1 if c < 0x80 else 2 if c >> 5 == 6 else 3 if c >> 4 == 14 else 4 if c >> 3 == 30 else 1
        if j - k < need:
            j = k
    body = b[i:j]
    body.decode("utf-8")  # strict: raises if the middle is broken
    return b[:i], body, b[j:]


def row_pieces(row):
    """GPT-2 row -> (context bytes, [scored piece bytes or None for EOT], unscored bytes count, scored bytes count)."""
    ctx = TB[row[0]] if row[0] != EOT else None
    pieces, cur = [], []
    for t in row[1:]:
        if t == EOT:
            if cur:
                pieces.append(b"".join(cur))
                cur = []
            pieces.append(None)
        else:
            cur.append(TB[t])
    if cur:
        pieces.append(b"".join(cur))
    lost = 0
    ctx_extra = b""
    # first scored text piece may start inside a character; the last may end inside one
    fi = next((i for i, p in enumerate(pieces) if p is not None), None)
    li = max((i for i, p in enumerate(pieces) if p is not None), default=None)
    if fi is not None:
        for i in sorted({fi, li}):
            lead, body, trail = valid_utf8_span(pieces[i])
            if i == fi and lead:
                ctx_extra, lost = lead, lost + len(lead)
            if i == li and trail:
                lost += len(trail)
            pieces[i] = (lead if i != fi else b"") + body + (trail if i != li else b"")
            if i == fi and i == li:
                pieces[i] = body
    scored = sum(len(p) for p in pieces if p is not None)
    return ctx, ctx_extra, pieces, lost, scored


def build_seq(m, row):
    """model-tokenizer sequence for one GPT-2 row: (ids, n_context_ids, scored bytes, unscored bytes)."""
    ctx, extra, pieces, lost, scored = row_pieces(row)
    if ctx is None and not extra:
        cids = [m.eot]
    else:
        cids = m.encode([((ctx or b"") + extra).decode("utf-8", errors="replace")])[0] if (ctx or extra) else []
        if ctx is None:
            cids = [m.eot] + cids
    texts = [p.decode("utf-8") for p in pieces if p]
    enc = m.encode(texts) if texts else []
    ids, k = list(cids), 0
    for p in pieces:
        if p is None:
            ids.append(m.eot)
        elif p:
            ids += enc[k]
            k += 1
    return ids, len(cids), scored, lost


def windows(n, nctx, W=1025):
    """[(start, first scored target index)] covering targets nctx .. n-1 (target index = position in ids)."""
    out = [(0, nctx)]
    covered = min(n, W) - 1
    while covered < n - 1:
        s = max(0, min(covered - 511, n - W))
        out.append((s, covered + 1))
        covered = s + W - 1
    return out


def bpb_rows(m, rows, bs=16, W=None):
    """(nll_sum nats, scored bytes, unscored bytes, targets) over GPT-2 rows for model m."""
    if m.gpt2:
        nll = m.last_nll = batched(m.nll, rows, bs)
        return float(nll.sum(dtype=np.float64)), int(NB[rows[:, 1:]].sum()), 0, int(rows[:, 1:].size)
    W = W or rows.shape[1]
    wins, wts, sb, lb, nt = [], [], 0, 0, 0
    for r in rows:
        ids, nctx, s, l = build_seq(m, r)
        sb, lb = sb + s, lb + l
        n = len(ids)
        nt += n - nctx
        for st, first in windows(n, nctx, W):
            seg = ids[st:st + W]
            w = np.zeros(W - 1, np.float32)
            w[max(first, st + 1) - st - 1: len(seg) - 1] = 1.0
            wins.append(seg + [0] * (W - len(seg)))
            wts.append(w)
    wins, wts = np.array(wins, np.int64), np.array(wts)
    tot = 0.0
    for i in range(0, len(wins), bs):
        tot += float((m.nll(wins[i:i + bs]) * wts[i:i + bs]).sum(dtype=np.float64))
    assert abs(wts.sum() - nt) < 0.5, (wts.sum(), nt)
    return tot, sb, lb, nt


# ---------------------------------------------------------------- tasks
def jl_append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def task_bpb(labels):
    for lab in labels:
        m = Model(lab)
        for cname in ("fineweb", "wikitext103", "pg19"):
            rows = corpus(cname)
            t = time.time()
            tot, sb, lb, nt = bpb_rows(m, rows)
            rec = {"label": lab, "corpus": cname, "rows": int(len(rows)), "nll_nats": tot, "scored_bytes": sb,
                   "unscored_bytes": lb, "targets": nt, "bpb": tot / (LN2 * sb), "nll_per_target": tot / nt, "s": time.time() - t}
            if m.gpt2:
                nll = m.last_nll
                y = rows[:, 1:]
                b = np.searchsorted(EDGES, RANK[y], side="right") - 1
                rec["bucket_nll_sum"] = [float(nll[b == i].sum()) for i in range(5)]
                rec["bucket_n"] = [int((b == i).sum()) for i in range(5)]
            jl_append(os.path.join(OUT, "eval_bpb.jsonl"), rec)
            print(f"[bpb] {lab:16s} {cname:12s} BPB {rec['bpb']:.4f}  nll/target {rec['nll_per_target']:.4f}  "
                  f"bytes {sb:,} (+{lb} unscored)  {rec['s']:.0f}s", flush=True)


def task_validate():
    """GPT-2-mode check: the byte-aligned path with tiktoken re-encoding vs direct scoring, on FineWeb val."""
    rows = corpus("fineweb")
    out = {}
    for lab in ("B384", "H3s384_300M"):
        m = Model(lab)
        direct = bpb_rows(m, rows)
        m.gpt2 = False
        realigned = bpb_rows(m, rows)
        logged = [json.loads(l) for l in open(os.path.join(OUT, MODELS[lab][1] + ".jsonl")) if '"event": "final"' in l][-1]["val_loss_full"]
        assert direct[1] == 23_160_213, direct[1]
        out[lab] = {"logged_val_loss_full": logged, "direct_minus_logged": direct[0] / direct[3] - logged,
                    "gate_ok": abs(direct[0] / direct[3] - logged) < 1e-3 and abs(realigned[0] / (LN2 * realigned[1]) - direct[0] / (LN2 * direct[1])) < 2e-3,
                    "direct_bpb": direct[0] / (LN2 * direct[1]), "direct_bytes": direct[1], "direct_nll_per_token": direct[0] / direct[3],
                    "aligned_bpb": realigned[0] / (LN2 * realigned[1]), "aligned_bytes": realigned[1], "aligned_unscored": realigned[2]}
        print(lab, out[lab], flush=True)
    json.dump(out, open(os.path.join(OUT, "eval_validate.json"), "w"), indent=1)


def pad_seqs(seqs, mult=64):
    L = max(len(s) for s in seqs)
    L = -(-L // mult) * mult
    return np.array([s + [0] * (L - len(s)) for s in seqs], np.int64), L


def seq_logprob_sums(m, seqs, bs=256):
    """sum of log p over every target of each sequence (sequence = [EOT] + tokens), padded in batches."""
    out = []
    for i in range(0, len(seqs), bs):
        X, L = pad_seqs(seqs[i:i + bs])
        nll = m.nll(np.concatenate([X, np.zeros((len(X), 1), np.int64)], 1))
        for j, s in enumerate(seqs[i:i + bs]):
            out.append(-float(nll[j, : len(s) - 1].sum(dtype=np.float64)))
    return out


def task_blimp(labels):
    from datasets import get_dataset_config_names, load_dataset
    subs = get_dataset_config_names("nyu-mll/blimp")
    data = [(s, load_dataset("nyu-mll/blimp", s, split="train")) for s in subs]
    for lab in labels:
        m = Model(lab)
        t = time.time()
        per = {}
        for s, ds in data:
            good = [[m.eot] + e for e in m.encode(list(ds["sentence_good"]))]
            bad = [[m.eot] + e for e in m.encode(list(ds["sentence_bad"]))]
            lg, lb = seq_logprob_sums(m, good), seq_logprob_sums(m, bad)
            per[s] = {"term": ds[0]["linguistics_term"], "n": len(lg), "correct": int(sum(a > b for a, b in zip(lg, lb)))}
        tot = sum(v["correct"] for v in per.values()) / sum(v["n"] for v in per.values())
        jl_append(os.path.join(OUT, "eval_blimp.jsonl"), {"label": lab, "acc": tot, "subsets": per, "s": time.time() - t})
        print(f"[blimp] {lab:16s} acc {100 * tot:.2f}%  {time.time() - t:.0f}s", flush=True)


def task_lambada(labels):
    from datasets import load_dataset
    texts = load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"]
    split = [(t[: t.rfind(" ")], t[t.rfind(" "):]) for t in texts]
    first_rank = np.array([RANK[G2.encode_ordinary(w)[0]] for _, w in split])
    for lab in labels:
        m = Model(lab)
        t0 = time.time()
        recs = []
        bs = 64
        for i in range(0, len(split), bs):
            part = split[i:i + bs]
            ctx = m.encode([c for c, _ in part])
            tgt = m.encode([w for _, w in part])
            seqs = [[m.eot] + c + w for c, w in zip(ctx, tgt)]
            X, L = pad_seqs(seqs)
            pos = np.array([[j, len(seqs[j]) - len(tgt[j]) - 1 + k] for j in range(len(part)) for k in range(len(tgt[j]))])
            ys = np.array([y for t in tgt for y in t])
            am, lp = m.argmax_lp(X, pos, ys, P=512)
            k = 0
            for j, (c, w) in enumerate(part):
                nv = len(tgt[j])
                nll = -float(lp[k:k + nv].sum())
                ok = bool((am[k:k + nv] == ys[k:k + nv]).all())
                k += nv
                recs.append((ok, nll / LN2 / len(w.encode("utf-8"))))
        acc = np.array([r[0] for r in recs])
        bpbw = np.array([r[1] for r in recs])
        hi = first_rank < 4096
        rec = {"label": lab, "n": len(recs), "acc": float(acc.mean()), "word_bits_per_byte": float(bpbw.mean()),
               "acc_top4096": float(acc[hi].mean()), "acc_rest": float(acc[~hi].mean()), "n_top4096": int(hi.sum()),
               "bpb_top4096": float(bpbw[hi].mean()), "bpb_rest": float(bpbw[~hi].mean()), "s": time.time() - t0}
        jl_append(os.path.join(OUT, "eval_lambada.jsonl"), rec)
        print(f"[lambada] {lab:16s} acc {100 * rec['acc']:.2f}% (top4096 {100 * rec['acc_top4096']:.2f} / rest {100 * rec['acc_rest']:.2f}) "
              f"word bits/byte {rec['word_bits_per_byte']:.4f}  {rec['s']:.0f}s", flush=True)


def task_copy(labels):
    fw = corpus("fineweb")[:2000, 1:513]
    rng = np.random.default_rng(11)
    pool = np.flatnonzero((RANK >= 4096) & (np.arange(len(RANK)) != EOT))
    rnd = rng.choice(pool, (2000, 512))
    for lab in labels:
        m = Model(lab)
        if not m.gpt2:
            print(f"[copy] {lab}: not a GPT-2-vocab model; skipped (pinned rule)")
            continue
        rec = {"label": lab}
        for nm, base in (("E5a", fw), ("E5b", rnd)):
            X = np.concatenate([np.full((len(base), 1), EOT), base, base], 1)       # 1 + 512 + 512 = 1,025
            nll = batched(m.nll, X, 16)                                              # [2000, 1024]: targets base, base
            c1, c2 = nll[:, :512], nll[:, 512:]
            b = np.searchsorted(EDGES, RANK[base], side="right") - 1
            rec[nm] = {"copy1": float(c1.mean()), "copy2": float(c2.mean()),
                       "copy2_first_half": float(c2[:, :256].mean()), "copy2_second_half": float(c2[:, 256:].mean()),
                       "bucket_copy1": [float(c1[b == i].mean()) if (b == i).any() else None for i in range(5)],
                       "bucket_copy2": [float(c2[b == i].mean()) if (b == i).any() else None for i in range(5)],
                       "bucket_n": [int((b == i).sum()) for i in range(5)]}
        jl_append(os.path.join(OUT, "eval_copy.jsonl"), rec)
        print(f"[copy] {lab:16s} E5a copy1 {rec['E5a']['copy1']:.4f} copy2 {rec['E5a']['copy2']:.4f} | "
              f"E5b copy1 {rec['E5b']['copy1']:.4f} copy2 {rec['E5b']['copy2']:.4f}", flush=True)


if __name__ == "__main__":
    task = sys.argv[1]
    labels = [l for l in (sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS) if l in MODELS and have(l)]
    if task == "corpora":
        for c in ("fineweb", "wikitext103", "pg19"):
            r = corpus(c)
            print(c, r.shape, "bytes", int(NB[r[:, 1:]].sum()))
    elif task == "validate":
        task_validate()
    else:
        print(f"[{task}] models: {labels}", flush=True)
        {"bpb": task_bpb, "blimp": task_blimp, "lambada": task_lambada, "copy": task_copy}[task](labels)
