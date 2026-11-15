"""Round 12 analysis (2026-11-10): close the deal on the two-level head. Rules pinned here at 22:31 CET on 2026-11-10,
before any round-12 run or eval started (copy: runs/round12/round12_analyze_docstring_2231.py.txt). Anything decided after
seeing a result is labelled post-hoc in the summary. Seed 0 unless stated. BPB = bits per UTF-8 byte of the shared
FineWeb-Edu val text (23,160,213 bytes); GPT-2-vocabulary arms use the logged val_loss_full x tokens / (bytes ln 2)
(round 11's gate: equal to the byte-aligned evaluator within 2e-5 BPB), BPE arms use round11_eval.task_bpb (byte-aligned).

Part A — LAMBADA sanity check (round12_lambada.py; eval only). Models: B384 (wsd_B384_300M_s0), H3*-384 575M
  (wsd_H3s384_575M_s0), F8k-384 631M, F4k-384 700M. Passages = EleutherAI/lambada_openai test (5,153); last word w = the
  text after the last space (with its leading space), context = the text before it (round 11 E3).
  A1 whole-passage BPB: each passage scored as [EOT] + encode(passage) under the model's own tokenizer (every token
     scored, so every byte is scored); BPB = sum of bits over all passages / sum of UTF-8 bytes.
  A2 last-word bits/byte (round 11 E3's definition: per-passage bits of w's tokens / bytes of w, then the mean over
     passages) split by "appears earlier": the stripped word occurs in the context as a whole word, exact and
     case-sensitive (regex (?<!\\w)WORD(?!\\w)). n is reported for both halves. Advantage = H3* bits/byte - F8k bits/byte
     (positive = F8k better).
  A3 ten passages drawn with np.random.default_rng(12) from the 100 with the largest |F8k - H3*| last-word bits/byte
     difference: token splits under both tokenizers with the per-token NLL (bits).
  Verdict: evaluator bug if BPB_whole(F8k) - BPB_whole(H3*) <= -0.15 (then find it, fix it, re-run E3 for every round-11
     model and report the corrected table); real effect if |that delta| <= 0.05 AND advantage(earlier) >= 2 x
     advantage(rest) with advantage(earlier) > 0; otherwise unclear (numbers reported).

Part B — big vocabulary at equal FLOPs. Tokenizer bpe131072 = round11_bpe.py's recipe (first 2.0 GB of the training
  stream, shuffle seed 0, GPT-2 pre-tokenisation, EOT = id 131,071, not a registered special token); round-trip gate on all
  4,895 val docs; sha256 and val bytes/token recorded; same document order.
  Arms (round-5 part-B d=384 shape, 6 unique blocks, H 6, f 1536, tied embeddings; B384 / H3*-384 recipe: LR 2e-3, WSD,
  10% decay branches):
    F128k-384  flat tied head over 131,072 ids (the round-8 chunked cross-entropy + remat path; chunk 1024 tokens).
    H128k-384  two-level head, H3* recipe: K = 8,192 singletons = the top-K ids by target count over the rows the H128k
               run consumes (stable + final decay), ties by id; C' = 4,096 tail clusters by heads.balanced_kmeans on the
               trained embedding rows of a flat d=128 proxy (arm tB on bpe131072, 50M stable tokens + 10% decay, seed 0)
               (capacity ceil(122,880 / 4,096) = 30); tied-mean cluster logits (ubar + R, bc).
  Budget: total training FLOPs (stable + decay, logged flops_per_token.total) of each arm's final branch within 0.5% of
    H3*-384 575M (6.716845e16); a branch within 0.5% of the 300M point (3.504925e16). H128k also gets a branch at
    6.716845e16 - proxy FLOPs (the "charged" row). Verdicts use the uncharged final branch (H3*-384's classes came from
    an uncharged round-8 B_r8 run); the charged row is reported next to it and each verdict is restated with it.
  Gates (round12_gates.py, before the full runs): G1 normalisation of the full reference (|logsumexp| < 1e-5), G2 fast NLL
    = reference NLL (mean |diff| < 1e-5) and gradients (rel. < 1e-5), G3 F128k chunked CE = unchunked reference within 1e-5
    on a small batch, G4 data hash identical for F128k / H128k smokes; exact greedy (round9_decode.exact_greedy) = argmax of
    the reference on 1,000 val contexts, and exact sampling: the two-stage probabilities p(c)p(w|c) equal the reference
    within 1e-5 on 1,000 contexts plus an empirical TV (200k samples, one context) within 1.5x the TV of 200k direct
    samples from the reference. After training the greedy check is repeated on the trained checkpoint (fallback rate and
    mean clusters visited reported).
  Verdicts (seed 0 vs seed 0): big vocab for free if BPB(H128k) <= BPB(H3*-384 575M) - 0.003; flat can't use it if
    BPB(F128k) >= BPB(B384 matched, D1 seed 0) + 0.003; two-level wins at 128k if BPB(H128k) - BPB(F128k) <= -0.003.
    Any delta within +-0.003 of its threshold runs seed 1 for both sides, and the verdict then uses the 2-seed means.
  E1 (WikiText-103, PG-19), E2 (BLiMP), E3 (LAMBADA; Part A's corrected evaluator if Part A finds a bug) on both arms'
    final branches. Decode: round11_decode.py harness (random bf16 weights) at batch 1 and batch 32 for B384, H3*-384,
    F8k-384, F128k-384, H128k-384; ms/byte = ms/token / val bytes per token.
  Head share of training FLOPs = flops_per_token.head / flops_per_token.total.

Part C (optional, after A, B, D): tH3ctx (d=128). Verdicts: fixes copying if the E5a copy-2 rank >= 4,096 delta vs B_r8
  <= +0.131; closes the gap if total delta vs B_r8 at 1200M <= +0.015 and wall <= 1.5x H3k8c4's.

Part D. D1: B384 at total FLOPs within 0.5% of 6.716845e16, seeds 0 and 1. Headline equal-FLOPs delta = mean over
  seeds {0, 1} of BPB(H3*-384 575M) - mean over seeds {0, 1} of BPB(B384 matched); each seed, the mean and the spread
  (|seed0 - seed1| per side) are reported. H3*-384 575M seed 1 = runs/round11/wsd_H3s384_575M_s1 (chainR7). E1 and E2 on
  the matched B384 models. D2: bioB_100000 (100 exposures) and bioH_100000_eqF (237.4 exposures) at seed 1 (same
  population file, init seed 1); "more tokens -> more knowledge" stands if bits(bioH_eqF) / bits(bioB_100000) >= 1.2 at
  both seeds. D3: chainR7's BLiMP seed-1 numbers folded into round 11's E2 table (round11_analyze.py, unchanged rules).

Seeds: a verdict within +-0.003 BPB (A, B, D1) or +-5% of the ratio (D2: 1.14-1.26) of its threshold runs seed 1 for
  both sides. Smoke every new arm (200 steps); stop and report if a full-run estimate is over 2 h. ulimit -c 0.

python round12_analyze.py  -> runs/round12/round12_summary.md, round12_results.json
"""
import json
import os

import numpy as np

from round9_analyze import BPN, load_run

HERE = os.path.dirname(os.path.abspath(__file__))
RD = {r: os.path.join(HERE, "runs", f"round{r}") for r in (8, 9, 10, 11, 12)}
OUT = RD[12]
GREY, BLUE = "#6f6e69", "#2a78d6"
VAL_BYTES = 23_160_213
GPT2_BPT = 4_999_168 / VAL_BYTES
T575, T300 = 6.716844637929216e16, 3.504924836453376e16


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf8")] if os.path.exists(p) else []


def js(p, default=None):
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else default


def f4(x, s="+.4f"):
    return "-" if x is None else format(x, s)


def R(rnd, name):
    run = load_run(os.path.join(RD[rnd], name + ".jsonl"))
    if run:
        run["name"] = name
    return run


def br(run, tok, tol=0.01):
    if not run:
        return None
    ks = [k for k in run["branch"] if abs(k - tok) <= tol * tok]
    return run["branch"][ks[0]] if ks else None


def last(rows, **kw):
    m = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return m[-1] if m else None


def seed_flag(delta, thr, band=0.003):
    return delta is not None and abs(delta - thr) <= band + 1e-12


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    return plt


R11 = js(os.path.join(RD[11], "round11_results.json"), {})
EB12 = jl(os.path.join(OUT, "eval_bpb.jsonl"))
EB11 = jl(os.path.join(RD[11], "eval_bpb.jsonl"))
BPE = js(os.path.join(OUT, "bpe131072.json"), {})
BPT131 = BPE.get("val_bytes_per_token_docs", 4.9437585274612506)
BPT8K, BPT4K = 3.932876256603333, 3.4727411904724454


def decode_tab():
    out = {}
    for r in jl(os.path.join(OUT, "decode.jsonl")):
        out.setdefault((r["arm"], r["batch"]), []).append(r["ms_per_step"])
    return {k: float(np.mean(v)) for k, v in out.items()}


DEC = decode_tab()


def run_row(lab, run, tok, bpb, bpt, dec_arm, ref_flops):
    """one Part-B / D1 table row for the branch of `run` at `tok` stable tokens."""
    b = br(run, tok)
    if b is None:
        return None
    fpt = run["final"]["flops_per_token"]
    f = fpt["total"] * b["tokens"]
    P = run["final"]["params"]
    d1, d32 = DEC.get((dec_arm, 1)), DEC.get((dec_arm, 32))
    return {"label": lab, "run": run["name"], "stable_tokens": tok, "tokens": b["tokens"], "flops": f,
            "flops_vs_target": f / ref_flops, "bpb": bpb, "wall_min": b["wall"] / 60, "params": P["total"],
            "emb_params": P["embedding"], "nonemb_params": P["non_embedding"], "bytes_per_token": bpt,
            "head_share": fpt["head"] / fpt["total"], "fpt": fpt["total"],
            "dec_ms_tok_b1": d1, "dec_ms_byte_b1": None if d1 is None else d1 / bpt,
            "dec_ms_tok_b32": None if d32 is None else d32 / 32, "dec_ms_byte_b32": None if d32 is None else d32 / 32 / bpt}


def gpt2_bpb(run, tok):
    b = br(run, tok)
    return None if b is None else b["loss"] * BPN


def ev_bpb(label, EB, corpus="fineweb"):
    e = last(EB, label=label, corpus=corpus)
    return None if e is None else e["bpb"]


# ------------------------------------------------------------------ Part A
def part_a(L, res):
    A = js(os.path.join(OUT, "partA_lambada.json"))
    L += ["## Part A — LAMBADA sanity check", ""]
    if not A:
        L += ["Not run.", ""]
        return
    L += [f"EleutherAI/lambada_openai test, {A['n']:,} passages; target word 'appears earlier' in the passage (whole word, exact): "
          f"n = {A['n_earlier']:,}; rest n = {A['n_rest']:,}.", "",
          "| model | whole-passage BPB | last word bits/byte | appears earlier | rest |", "|---|---|---|---|---|"]
    for lab, m in A["models"].items():
        L.append(f"| {lab} | {m['bpb_whole']:.4f} | {m['word_bpb']:.4f} | {m['word_bpb_earlier']:.4f} | {m['word_bpb_rest']:.4f} |")
    v = A["verdict"]
    L += ["", f"**Verdict (pinned): whole-passage BPB(F8k) − BPB(H3*) = {v['d_whole_F8k_minus_H3s']:+.4f}; F8k's last-word advantage "
          f"(H3* − F8k bits/byte) on 'appears earlier' targets {v['adv_earlier']:+.4f} vs {v['adv_rest']:+.4f} on the rest → "
          f"{v['verdict']}.**", "",
          "Ten passages with token splits and per-token NLL: `runs/round12/partA_examples.md`. In all ten, the target is a rare name "
          "that occurs earlier in the passage. GPT-2's 50k vocabulary has it as one rare token (about 15–25 bits); the 8k vocabulary "
          "splits it into a first piece plus continuation pieces that are nearly free to copy from context. B384 ≈ H3* on this split, "
          "so the effect belongs to the vocabulary, not the head (post-hoc reading).", ""]
    res["partA"] = A


# ------------------------------------------------------------------ Part B
def part_b(L, res):
    L += ["## Part B — big vocabulary (V = 131,072) at equal FLOPs", ""]
    cr = js(os.path.join(OUT, "v131072_adapt_report.json"), {})
    fr = js(os.path.join(OUT, "v131072_freq_report.json"), {})
    if BPE:
        L += [f"Tokenizer bpe131072: sha256 `{BPE['sha256']}`, {BPE['train_bytes'] / 1e9:.3f} GB of training text "
              f"({BPE['train_docs']:,} docs), trained in {BPE['train_s']:.0f} s; round trip on all {BPE['gate_roundtrip_docs']:,} val docs: "
              f"{BPE['gate_roundtrip_failures']} failures; val bytes/token {BPE['val_bytes_per_token_docs']:.3f} (GPT-2 {1 / GPT2_BPT:.3f}).", ""]
    if fr:
        L.append(f"Frequent list: target counts over the {fr['batches']:,} batches the H128k run consumes ({fr['targets_counted']:,} targets); "
                 f"top-8,192 share {fr['top_share']['8192']:.4f}, top-4,096 {fr['top_share']['4096']:.4f}; "
                 f"{131072 - fr['nonzero_vocab']:,} ids never seen.")
    if cr:
        L.append("Class report (`runs/round12/v131072_adapt_report.json`): " + json.dumps(cr)[:600])
    proxy = R(12, "wsd_tF128k_proxy_50M_s0")
    pf = None
    if proxy:
        pf = proxy["final"]["flops_per_token"]["total"] * proxy["final"]["tokens"]
        L.append(f"Proxy wsd_tF128k_proxy_50M_s0 (d=128 flat, bpe131072): {proxy['final']['tokens'] / 1e6:.1f}M tokens, {pf:.4e} FLOPs "
                 f"({100 * pf / T575:.1f}% of 6.717e16), {proxy['final']['wall_s'] / 60:.1f} min.")
    G = {g: js(os.path.join(OUT, f"gates_{g}.json")) for g in ("flat", "hier", "trained_wsd_H128k384_575M_s0")}
    SC = js(os.path.join(OUT, "smoke_check.json"), {})
    L += ["", "Gates:", ""]
    for g, v in G.items():
        if not v:
            L.append(f"- [{g}] not run")
            continue
        for k, c in v["checks"].items():
            if isinstance(c, dict) and "ok" in c:
                L.append(f"- [{g}] {'OK' if c['ok'] else 'FAIL'} {k}: {c.get('detail', '')}")
            else:
                L.append(f"- [{g}] {k}: {json.dumps(c)[:300]}")
    if SC:
        h1 = SC["arms"].get("F128k384", {}).get("first_hashes")
        h2 = SC["arms"].get("H128k384", {}).get("first_hashes")
        L.append(f"- [G4] data hash: F128k-384 and H128k-384 smokes saw {'identical' if h1 and h1 == h2 else 'DIFFERENT'} first batches")
        for k, r in SC["arms"].items():
            if not r.get("crashed"):
                L.append(f"- [smoke] {k}: {r['tok_s']:,.0f} tok/s, peak {r['peak_mem_gb']:.2f} GB, est. full run {r['est_full_min']:.0f} min")
        for f in SC.get("fails", []):
            L.append(f"- [smoke] FAIL {f}")
    # ---- table
    H3 = R(10, "wsd_H3s384_575M_s0")
    Bm = R(12, "wsd_B384m_315M_s0")
    F = R(12, "wsd_F128k384_161M_s0")
    H = R(12, "wsd_H128k384_575M_s0")
    r11 = R11.get("partB", {}).get("rows", {})
    g2 = 1 / GPT2_BPT
    spec = [("H3*-384 (300M)", H3, 300e6, gpt2_bpb(H3, 300e6), g2, "H3s384", T300),
            ("H3*-384 (575M)", H3, 575e6, gpt2_bpb(H3, 575e6), g2, "H3s384", T575),
            ("B384 matched (164M)", Bm, 164429824, gpt2_bpb(Bm, 164429824), g2, "B384", T300),
            ("B384 matched (315M)", Bm, 315162624, gpt2_bpb(Bm, 315162624), g2, "B384", T575),
            ("F8k-384 (631M)", R(11, "wsd_F8k384_631M_s0"), 631e6, r11.get("F8k-384 (631M)", {}).get("bpb"), BPT8K, "F8k384", T575),
            ("F4k-384 (700M)", R(11, "wsd_F4k384_700M_s0"), 700e6, r11.get("F4k-384 (700M)", {}).get("bpb"), BPT4K, "F4k384", T575),
            ("F128k-384 (84M)", F, 83886080, ev_bpb("F128k384_84M", EB12), BPT131, "F128k384", T300),
            ("F128k-384 (161M)", F, 160759808, ev_bpb("F128k384_161M", EB12), BPT131, "F128k384", T575),
            ("H128k-384 (300M)", H, 299892736, ev_bpb("H128k384_300M", EB12), BPT131, "H128k384", T300),
            ("H128k-384 (522M, proxy charged)", H, 521797632, ev_bpb("H128k384_522M", EB12), BPT131, "H128k384", T575),
            ("H128k-384 (575M)", H, 574750720, ev_bpb("H128k384_575M", EB12), BPT131, "H128k384", T575)]
    rows = {}
    for s in spec:
        r = run_row(*s)
        if r:
            rows[s[0]] = r
    ch = rows.get("H128k-384 (522M, proxy charged)")
    if ch and pf:
        ch["flops_incl_proxy"] = ch["flops"] + pf
        ch["flops_vs_target"] = ch["flops_incl_proxy"] / T575
    L += ["", "| arm (stable tokens) | total tokens | FLOPs | vs target | BPB | wall min | params (emb / non-emb) | bytes/tok | "
          "decode b1 ms/tok · ms/byte | decode b32 ms/tok · ms/byte | head share |", "|---" * 11 + "|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['tokens'] / 1e6:.1f}M | {r['flops']:.4e} | {r['flops_vs_target']:.4f} | {f4(r['bpb'], '.4f')} | "
                 f"{r['wall_min']:.1f} | {r['params'] / 1e6:.2f}M ({r['emb_params'] / 1e6:.2f} / {r['nonemb_params'] / 1e6:.2f}) | "
                 f"{r['bytes_per_token']:.3f} | {f4(r['dec_ms_tok_b1'], '.4f')} · {f4(r['dec_ms_byte_b1'], '.5f')} | "
                 f"{f4(r['dec_ms_tok_b32'], '.5f')} · {f4(r['dec_ms_byte_b32'], '.6f')} | {100 * r['head_share']:.1f}% |")
    L += ["", "'vs target' = the branch's training FLOPs over 6.717e16 (575M-matched rows) or 3.505e16 (300M-matched rows); the charged "
          "H128k row includes the proxy's FLOPs. GPT-2 rows: BPB from the logged full-val loss. BPE rows: the byte-aligned evaluator "
          "(round 11's numbers for F8k / F4k). Decode: random bf16 weights, `runs/round12/decode.md`.", ""]
    # ---- verdicts
    V = {}
    g = lambda k: rows.get(k, {}).get("bpb")
    h, h3, fl, bm = g("H128k-384 (575M)"), g("H3*-384 (575M)"), g("F128k-384 (161M)"), g("B384 matched (315M)")
    hc = g("H128k-384 (522M, proxy charged)")
    if h is not None and h3 is not None:
        d = h - h3
        V["big_vocab_for_free"] = {"delta": d, "holds": d <= -0.003, "seed1": seed_flag(d, -0.003),
                                   "charged_delta": None if hc is None else hc - h3,
                                   "charged_holds": None if hc is None else hc - h3 <= -0.003}
    if fl is not None and bm is not None:
        d = fl - bm
        V["flat_cant_use_it"] = {"delta": d, "holds": d >= 0.003, "seed1": seed_flag(d, 0.003)}
    if h is not None and fl is not None:
        d = h - fl
        V["two_level_wins_128k"] = {"delta": d, "holds": d <= -0.003, "seed1": seed_flag(d, -0.003),
                                    "charged_delta": None if hc is None else hc - fl,
                                    "charged_holds": None if hc is None else hc - fl <= -0.003}
    names = {"big_vocab_for_free": "big vocab for free: BPB(H128k) − BPB(H3*-384 575M) ≤ −0.003",
             "flat_cant_use_it": "flat can't use it: BPB(F128k) − BPB(B384 matched) ≥ +0.003",
             "two_level_wins_128k": "two-level wins at 128k: BPB(H128k) − BPB(F128k) ≤ −0.003"}
    for k, v in V.items():
        s = f"**Verdict (pinned) — {names[k]}: Δ = {v['delta']:+.4f} → {'holds' if v['holds'] else 'does not hold'}.**"
        if v.get("charged_delta") is not None:
            s += f" With the proxy charged (522M branch): Δ = {v['charged_delta']:+.4f} → {'holds' if v['charged_holds'] else 'does not hold'}."
        if v["seed1"]:
            s += " Within ±0.003 of the threshold: the seed-1 rule applies."
        L += [s, ""]
    for k in names:
        if k not in V:
            L += [f"Verdict — {names[k]}: pending.", ""]
    h300, f84, b164, h3300 = g("H128k-384 (300M)"), g("F128k-384 (84M)"), g("B384 matched (164M)"), g("H3*-384 (300M)")
    m300 = {"H128k − H3* (3.505e16)": None if h300 is None or h3300 is None else h300 - h3300,
            "F128k − B384 matched (3.505e16)": None if f84 is None or b164 is None else f84 - b164,
            "H128k − F128k (3.505e16)": None if h300 is None or f84 is None else h300 - f84}
    L += ["At the 3.505e16 branches, the same differences (reported, not verdicts): "
          + "; ".join(f"{k} {f4(v)}" for k, v in m300.items()) + ".", ""]
    res["partB"] = {"rows": rows, "verdicts": V, "deltas_300": m300, "tokenizer": BPE, "classes": cr, "freq": fr,
                    "proxy_flops": pf, "gates": G, "smoke": SC, "plan": js(os.path.join(OUT, "plan.json"), {})}
    # ---- E1-E3
    L += ["### Part B evals at 6.717e16 (E1 OOD BPB, E2 BLiMP, E3 LAMBADA)", "",
          "| model | FineWeb-Edu | WikiText-103 | PG-19 | BLiMP % | LAMBADA acc % | LAMBADA word bits/byte |", "|---|---|---|---|---|---|---|"]
    BL12, LA12 = jl(os.path.join(OUT, "eval_blimp.jsonl")), jl(os.path.join(OUT, "eval_lambada.jsonl"))
    BL11, LA11 = jl(os.path.join(RD[11], "eval_blimp.jsonl")), jl(os.path.join(RD[11], "eval_lambada.jsonl"))
    ev = {}
    for lab, EB, BL, LA in (("B384", EB11, BL11, LA11), ("H3s384_575M", EB11, BL11, LA11), ("F8k384_631M", EB11, BL11, LA11),
                            ("F4k384_700M", EB11, BL11, LA11), ("B384m_s0", EB12, BL12, LA12),
                            ("F128k384_161M", EB12, BL12, LA12), ("H128k384_575M", EB12, BL12, LA12)):
        e = {c: ev_bpb(lab, EB, c) for c in ("fineweb", "wikitext103", "pg19")}
        b, la = last(BL, label=lab), last(LA, label=lab)
        e.update(blimp=None if b is None else 100 * b["acc"], lambada_acc=None if la is None else 100 * la["acc"],
                 lambada_wbpb=None if la is None else la["word_bits_per_byte"])
        if any(x is not None for x in e.values()):
            ev[lab] = e
            L.append(f"| {lab} | {f4(e['fineweb'], '.4f')} | {f4(e['wikitext103'], '.4f')} | {f4(e['pg19'], '.4f')} | "
                     f"{f4(e['blimp'], '.2f')} | {f4(e['lambada_acc'], '.2f')} | {f4(e['lambada_wbpb'], '.4f')} |")
    L += ["", "B384 / H3s384_575M / F8k / F4k rows are round 11's evals. E1–E3 on the Part B arms are reported; no round-12 rule "
          "was pinned on them.", ""]
    res["partB"]["evals"] = ev
    vocab_plot(rows, res)


def vocab_plot(rows, res):
    flat = [("F4k-384 (700M)", 4096), ("F8k-384 (631M)", 8192), ("B384 matched (315M)", 50257), ("F128k-384 (161M)", 131072)]
    two = [("H3*-384 (575M)", 50257), ("H128k-384 (575M)", 131072)]
    if not any(k in rows and rows[k]["bpb"] is not None for k, _ in flat + two):
        return
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for series, col, nm, mk in ((flat, GREY, "flat tied head", "o"), (two, BLUE, "two-level head (K 8,192 + C 4,096)", "s")):
        pts = [(v, rows[k]["bpb"], rows[k]["head_share"]) for k, v in series if k in rows and rows[k]["bpb"] is not None]
        if not pts:
            continue
        x, y, s = zip(*pts)
        a1.plot(x, y, marker=mk, color=col, lw=2, ms=8, label=nm)
        a2.plot(x, [100 * q for q in s], marker=mk, color=col, lw=2, ms=8, label=nm)
        for xi, yi in zip(x, y):
            a1.annotate(f"{yi:.3f}", (xi, yi), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#333")
    hc = rows.get("H128k-384 (522M, proxy charged)")
    if hc and hc["bpb"] is not None:
        a1.plot([131072], [hc["bpb"]], marker="s", mfc="white", color=BLUE, ms=8, lw=0, label="two-level, proxy FLOPs charged")
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([4096, 8192, 50257, 131072])
        a.set_xticklabels(["4k", "8k", "50k", "131k"])
        a.minorticks_off()
        a.set_xlabel("vocabulary size")
        a.grid(alpha=0.25)
    a1.set_ylabel("FineWeb-Edu val bits per byte")
    a2.set_ylabel("head share of training FLOPs (%)")
    a1.set_title("BPB at 6.717e16 training FLOPs (d=384, seed 0)", fontsize=11)
    a2.set_title("head share of training FLOPs", fontsize=11)
    a1.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"bpb_vs_vocab.{ext}"), dpi=150)
    plt.close(fig)
    res.setdefault("figures", []).append("runs/round12/bpb_vs_vocab.png")


# ------------------------------------------------------------------ Part D
def part_d(L, res):
    L += ["## Part D — two seeds and exact matching", "", "### D1 — B384 matched to 6.717e16 total FLOPs", "",
          "| seed | B384 matched BPB | its FLOPs (vs 6.717e16) | H3*-384 575M BPB | Δ (H3* − B) |", "|---|---|---|---|---|"]
    d1 = {}
    for s in (0, 1):
        Bm = R(12, f"wsd_B384m_315M_s{s}")
        H3 = R(10, "wsd_H3s384_575M_s0") if s == 0 else R(11, "wsd_H3s384_575M_s1")
        b, h = gpt2_bpb(Bm, 315162624), gpt2_bpb(H3, 575e6)
        bb = br(Bm, 315162624)
        fb = None if bb is None else Bm["final"]["flops_per_token"]["total"] * bb["tokens"]
        d1[s] = {"B384m": b, "H3s": h, "flops_B": fb, "delta": None if b is None or h is None else h - b}
        L.append(f"| {s} | {f4(b, '.4f')} | {f4(fb, '.4e')} ({f4(None if fb is None else fb / T575, '.4f')}) | {f4(h, '.4f')} | "
                 f"{f4(d1[s]['delta'])} |")
    if all(d1[s]["B384m"] is not None and d1[s]["H3s"] is not None for s in (0, 1)):
        mb, mh = float(np.mean([d1[s]["B384m"] for s in (0, 1)])), float(np.mean([d1[s]["H3s"] for s in (0, 1)]))
        d1.update(headline_delta=mh - mb, mean_B=mb, mean_H=mh, spread_B=abs(d1[0]["B384m"] - d1[1]["B384m"]),
                  spread_H=abs(d1[0]["H3s"] - d1[1]["H3s"]))
        L += ["", f"**2-seed headline equal-FLOPs Δ = mean BPB(H3*-384 575M) − mean BPB(B384 matched) = {mh:.4f} − {mb:.4f} = "
              f"{mh - mb:+.4f} BPB** (seed spread: B384 {d1['spread_B']:.4f}, H3* {d1['spread_H']:.4f}).", ""]
    else:
        L += ["", "2-seed headline Δ: pending.", ""]
    L += ["E1 / E2 on the matched B384 models:", "", "| model | FineWeb-Edu | WikiText-103 | PG-19 | BLiMP % |", "|---|---|---|---|---|"]
    BL12, BL11 = jl(os.path.join(OUT, "eval_blimp.jsonl")), jl(os.path.join(RD[11], "eval_blimp.jsonl"))
    d1e = {}
    for lab, EB, BL in (("B384m_s0", EB12, BL12), ("B384m_s1", EB12, BL12), ("H3s384_575M", EB11, BL11), ("H3s384_575M_s1", EB11, BL11)):
        e = {c: ev_bpb(lab, EB, c) for c in ("fineweb", "wikitext103", "pg19")}
        b = last(BL, label=lab)
        e["blimp"] = None if b is None else 100 * b["acc"]
        d1e[lab] = e
        L.append(f"| {lab} | {f4(e['fineweb'], '.4f')} | {f4(e['wikitext103'], '.4f')} | {f4(e['pg19'], '.4f')} | {f4(e['blimp'], '.2f')} |")
    L.append("")
    d1["evals"] = d1e
    # D2
    L += ["### D2 — knowledge at equal FLOPs, two seeds", "",
          "| seed | bits bioB_100000 (100 exposures) | bits bioH_100000_eqF (237.4 exposures) | ratio |", "|---|---|---|---|"]
    E4a = jl(os.path.join(RD[11], "eval_bios.jsonl")) + jl(os.path.join(OUT, "eval_bios.jsonl"))
    d2 = {}
    for s, (lb, lh) in ((0, ("bioB_100000", "bioH_100000_eqF")), (1, ("bioB_100000_s1", "bioH_100000_eqF_s1"))):
        b, h = last(E4a, label=lb), last(E4a, label=lh)
        d2[s] = {"bits_B": b and b["bits"], "bits_H_eqF": h and h["bits"], "ratio": (h["bits"] / b["bits"]) if (b and h) else None,
                 "exposures_H": h and h["exposures"]}
        L.append(f"| {s} | {f4(d2[s]['bits_B'], ',.0f')} | {f4(d2[s]['bits_H_eqF'], ',.0f')} | {f4(d2[s]['ratio'], '.3f')} |")
    rs = [d2[s]["ratio"] for s in (0, 1)]
    if all(r is not None for r in rs):
        hold = all(r >= 1.2 for r in rs)
        d2.update(verdict="stands" if hold else "does not stand", mean_ratio=float(np.mean(rs)), holds=hold)
        L += ["", f"**Verdict (pinned): 'more tokens → more knowledge at equal FLOPs' {d2['verdict']} (ratios {rs[0]:.3f} / {rs[1]:.3f}; "
              f"rule: ≥ 1.2 on both seeds).**", ""]
    else:
        L += ["", "D2 verdict: pending.", ""]
    # D3
    e2 = R11.get("E2", {})
    L += ["### D3 — E2 (BLiMP) with seed 1 (from round11_analyze.py, re-run)", ""]
    if e2.get("acc_s1"):
        L += ["| model | seed 0 % | seed 1 % |", "|---|---|---|"]
        for lab, v in e2["acc_s1"].items():
            base = lab[:-3] if lab.endswith("_s1") else lab
            v1 = v.get("overall") if isinstance(v, dict) else v
            L.append(f"| {base} | {f4(e2['acc'].get(base, {}).get('overall'), '.2f')} | {f4(v1, '.2f')} |")
        L.append("")
        for k, v in e2.get("deltas", {}).items():
            L.append(f"- {k}: " + ", ".join(f"{a} {f4(b) if isinstance(b, float) else b}" for a, b in v.items()))
    else:
        L.append("Seed-1 BLiMP numbers not in round11_results.json yet.")
    L.append("")
    res["partD"] = {"D1": d1, "D2": d2, "D3": e2}


# ------------------------------------------------------------------ P5: Jensen gap
def p5(L, res):
    J = js(os.path.join(OUT, "jensen.json"))
    L += ["## Paper P5 measurement — Jensen gap Δ_c(h) of the target's cluster (`runs/round12/jensen.json`)", ""]
    if not J:
        L += ["Not run.", ""]
        return
    L += ["| model | set | tail targets | mean Δ_c | std | mean error z_c − LSE_c | corr(offset, Δ_c) | mean Δ_c by bucket |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, r in J.items():
        for st in ("val", "E5a_copy1", "E5a_copy2"):
            v = r[st]
            L.append(f"| {lab} | {st} | {v['n_tail']:,} | {v['delta_mean_tail']:.3f} | {v['delta_std_tail']:.3f} | {v['error_mean_tail']:+.3f} | "
                     f"{v['corr_offset_delta']:.3f} | "
                     + ", ".join(f"{k} {b['delta_mean']:.2f}" for k, b in v["buckets"].items() if b["delta_mean"] is not None) + " |")
    L.append("")
    res["P5"] = {"jensen": J, "round11_bucket_deltas": R11.get("E1", {}).get("bucket_deltas", {})}


def theory_inputs(L, res):
    """Numbers the paper's theory / cost sections quote (not verdicts): the round-8 heads test (P1, P2, P4) and the
    cost model at a ~600M-parameter shape (Limitations)."""
    import re
    T = {}
    p = os.path.join(RD[8], "test_heads.out")
    if os.path.exists(p):
        s = open(p, encoding="utf8").read()
        m = re.search(r"rank flat / class-bias / hier: (\d+) (\d+) (\d+) \(bounds (\d+), (\d+), (\d+)\)", s)
        g = re.search(r"exact greedy matches argmax: (\d+)/(\d+); classes expanded mean ([\d.]+) of (\d+)", s)
        tv = re.search(r"TV distance to model distribution \(200k samples\): ([\d.]+)", s)
        T["test_heads"] = {"V": 600, "d": 16, "C": 24, "rows": 500,
                           "rank": dict(zip(("flat", "class_bias", "hier"), map(int, m.groups()[:3]))) if m else None,
                           "bound": dict(zip(("flat", "class_bias", "hier"), map(int, m.groups()[3:]))) if m else None,
                           "greedy_ok": int(g.group(1)) if g else None, "greedy_n": int(g.group(2)) if g else None,
                           "greedy_clusters_mean": float(g.group(3)) if g else None, "tv_200k": float(tv.group(1)) if tv else None,
                           "docstring_bounds": {"flat": "d + 1", "class_bias": "d + C + 1", "hier": "d + C"},
                           "note": "test_heads.py checks d + 2 (flat log-probs, bias + normaliser) and d + C + 1 (class-bias, hier); "
                                   "the heads.py docstring says d + 1 and d + C, i.e. it counts logits, not log-probs"}
    fb = js(os.path.join(RD[9], "decode_fallback.json"))
    if fb:
        T["r9_fallback"] = fb
    # cost model: flat head share = 6dV / (6 n (4d^2 + 2df) + 6 n T d + 6dV), f = 4d, T = 1024 (model.flops_per_token)
    shapes = []
    for d, n in ((1024, 48), (1280, 30), (1536, 20), (2048, 12)):
        blocks, attn = 6 * n * (4 * d * d + 8 * d * d), 6 * n * 1024 * d
        for V, hl in ((50257, (8192 + 4096 + 11)), (131072, (8192 + 4096 + 30))):
            fl = 6 * d * V
            two = 6 * d * hl
            shapes.append({"d": d, "layers": n, "V": V, "params_M": (12 * n * d * d + V * d) / 1e6,
                           "flat_head_share": fl / (blocks + attn + fl), "two_level_head_share": two / (blocks + attn + two),
                           "head_ratio": hl / V})
    T["cost_shapes"] = shapes
    d, n = 384, 6  # the experiments' own shape (for the social head-cost bars)
    blocks, attn = 6 * n * 12 * d * d, 6 * n * 1024 * d
    T["cost_d384"] = {f"{kind}_{V // 1000}k": (6 * d * hl) / (blocks + attn + 6 * d * hl)
                      for V, L2 in ((50257, 8192 + 4096 + 11), (131072, 8192 + 4096 + 30))
                      for kind, hl in (("flat", V), ("two_level", L2))}
    T["head_ratio_50k"] = (8192 + 4096 + 11) / 50257
    T["head_ratio_131k"] = (8192 + 4096 + 30) / 131072
    L += ["## Inputs for the paper's theory and cost sections (reported, not verdicts)", ""]
    if "test_heads" in T and T["test_heads"]["rank"]:
        t = T["test_heads"]
        L.append(f"- `runs/round8/test_heads.out` (V {t['V']}, d {t['d']}, C {t['C']}, {t['rows']} random h): log-prob rank flat / "
                 f"class-bias / two-level = {t['rank']['flat']} / {t['rank']['class_bias']} / {t['rank']['hier']}, checked bounds "
                 f"{t['bound']['flat']} / {t['bound']['class_bias']} / {t['bound']['hier']}; exact greedy {t['greedy_ok']}/{t['greedy_n']}, "
                 f"{t['greedy_clusters_mean']} clusters expanded on average (of {t['C']}); sampling TV {t['tv_200k']} (200k samples).")
        L.append(f"- Bug (documentation): {t['note']}.")
    L.append("- Head share of training FLOPs at ~600M-parameter GPT shapes (f = 4d, T = 1024, model.flops_per_token's formula):")
    for s in shapes:
        L.append(f"  - d {s['d']}, {s['layers']} layers, V {s['V']:,} ({s['params_M']:.0f}M params): flat {100 * s['flat_head_share']:.1f}%, "
                 f"two-level {100 * s['two_level_head_share']:.1f}% (head FLOPs ratio {s['head_ratio']:.3f})")
    L.append("")
    res["theory"] = T


def all_runs(res):
    """every finished run in runs/round8 .. runs/round12 (DONE marker): config, params, FLOPs/token, branches."""
    out = []
    for rnd in (8, 9, 10, 11, 12):
        for fn in sorted(os.listdir(RD[rnd])):
            if not fn.endswith(".DONE") or not fn.startswith("wsd_"):
                continue
            name = fn[:-5]
            run = R(rnd, name)
            if not run:
                continue
            c = run["config"]
            cfg = c.get("cfg", {})
            args = c.get("args", {})
            fpt = run["final"]["flops_per_token"]["total"]
            gpt2 = cfg.get("vocab") == 50304 and not cfg.get("data")
            for st, b in sorted(run["branch"].items()):
                out.append({"round": rnd, "run": name, "arm": c.get("arm"), "seed": args.get("seed"), "d": cfg.get("d"),
                            "vocab": cfg.get("vocab"), "head": cfg.get("head") or "flat", "data": cfg.get("data") or "fineweb-edu", "ngram": cfg.get("ngram") or 0,
                            "params": run["final"]["params"]["total"], "fpt": fpt, "stable_tokens": st, "tokens": b["tokens"],
                            "flops": fpt * b["tokens"], "loss": b["loss"], "bpb": b["loss"] * BPN if gpt2 else None,
                            "wall_min": b["wall"] / 60})
    res["all_runs"] = out


PERF_ARMS = ("B384", "H3s384", "F8k384", "F128k384", "H128k384")
PERF_NAME = {"B384": "flat 50k", "H3s384": "two-level 50k", "F8k384": "flat 8k", "F128k384": "flat 131k",
             "H128k384": "two-level 131k"}


def perf(L, res):
    """Follow-up (2026-11-11, user's request; no rule pinned): memory and speed of the d=384 heads (round12_perf.py)."""
    p = os.path.join(OUT, "perf.jsonl")
    if not os.path.exists(p):
        return
    last = {}
    for r in jl(p):
        n = r.get("micro") or r.get("rows") or r.get("batch")
        last[(r["kind"], r["arm"], n)] = r
    P = {"train": {}, "score": {}, "decode": {}}
    for (k, arm, n), r in last.items():
        P[k].setdefault(arm, {})[str(n)] = {x: r.get(x) for x in ("ok", "tok_s", "peak_gb", "mem_est_gb", "s_per_step",
                                                                   "s_per_call", "ms_per_step", "ms_per_token", "error")}
    res["perf"] = P
    L += ["## Follow-up — memory and speed of the d=384 heads (reported, no rule pinned)", "",
          "`round12_perf.py`, one fresh process per training/scoring configuration (configurations sharing a process slowed "
          "each other), random init, real validation rows. Peak = jax `peak_bytes_in_use`; estimate = XLA's compiled "
          "arguments + outputs + temporaries. Decode: random bf16 weights, KV cache, 256 new tokens.", ""]
    for kind, unit in (("train", "micro-batch rows"), ("score", "rows (forward only)"), ("decode", "batch")):
        sizes = sorted({int(n) for a in P[kind].values() for n in a})
        if not sizes:
            continue
        L += [f"### {kind}", "", "| head | " + " | ".join(f"{unit} {n}" for n in sizes) + " |",
              "|---|" + "---|" * len(sizes)]
        for arm in PERF_ARMS:
            cells = []
            for n in sizes:
                r = P[kind].get(arm, {}).get(str(n))
                if r is None:
                    cells.append("-")
                elif not r["ok"]:
                    cells.append(f"OOM (est. {r['mem_est_gb']:.1f} GB)" if r.get("mem_est_gb") else "failed")
                elif kind == "decode":
                    cells.append(f"{r['tok_s']:,.0f} tok/s, {r['ms_per_step']:.3f} ms/step, est. {r['mem_est_gb']:.2f} GB")
                elif kind == "score":  # the measured peak is set before the call (same at 32 and 64 rows)
                    cells.append(f"{r['tok_s']:,.0f} tok/s, est. {r['mem_est_gb']:.2f} GB")
                else:
                    cells.append(f"{r['tok_s']:,.0f} tok/s, peak {r['peak_gb']:.2f} GB")
            L.append(f"| {PERF_NAME[arm]} | " + " | ".join(cells) + " |")
        L.append("")
    # headline ratios at the training micro-batch (16): two-level vs flat at the same vocabulary
    rat = {}
    for V, fa, ha in (("50k", "B384", "H3s384"), ("131k", "F128k384", "H128k384")):
        f, h = P["train"].get(fa, {}).get("16"), P["train"].get(ha, {}).get("16")
        if f and h and f["ok"] and h["ok"]:
            rat[V] = {"tok_s_ratio": h["tok_s"] / f["tok_s"], "peak_saved_gb": f["peak_gb"] - h["peak_gb"],
                      "peak_ratio": h["peak_gb"] / f["peak_gb"]}
            L.append(f"- Training at micro 16, two-level vs flat at {V}: {rat[V]['tok_s_ratio']:.2f}x the tokens/s, "
                     f"peak {h['peak_gb']:.2f} vs {f['peak_gb']:.2f} GB.")
    P["ratios_micro16"] = rat
    L.append("")


def answers(L, res):
    d1 = res.get("partD", {}).get("D1", {})
    V = res.get("partB", {}).get("verdicts", {})
    r11b = R11.get("partB", {})
    L += ["## One-line answers: is the two-level head cheap AND good?", ""]
    hd = d1.get("headline_delta")
    if hd is None:
        s = "pending."
    elif hd < 0:
        s = (f"yes on BPB: at 6.717e16 training FLOPs the two-level head (H3*-384, 575M tokens) is {-hd:.4f} BPB better than the flat "
             f"head matched to the same FLOPs (mean of 2 seeds per side, Δ {hd:+.4f}).")
    else:
        s = (f"no: at 6.717e16 training FLOPs the two-level head is {hd:.4f} BPB worse than the flat head matched to the same FLOPs "
             f"(mean of 2 seeds per side, Δ {hd:+.4f}).")
    L.append("- **Equal FLOPs:** " + s)
    rows = r11b.get("rows") or {}
    et = rows["H3*-384 (300M)"]["bpb"] - rows["B384 (300M)"]["bpb"] if rows else None
    L.append("- **Equal tokens:** " + ("pending." if et is None else
             f"{'no' if et > 0 else 'yes'}: at 300M tokens BPB(H3*) − BPB(B384) = {et:+.4f} (seed 0, round 11); the gap sits in "
             f"targets ranked ≥ 4,096."))
    if V:
        L.append("- **Vocabulary size:** " + "; ".join(f"{k} Δ {v['delta']:+.4f} ({'holds' if v['holds'] else 'does not hold'})" for k, v in V.items()) + ".")
    else:
        L.append("- **Vocabulary size:** pending.")
    L.append("")


# ------------------------------------------------------------------ main
def main():
    L = ["# Round 12 summary — close the deal on the two-level head", "",
         "Decision rules pinned in `round12_analyze.py`'s docstring at 22:31 CET on 2026-11-10 (copy: "
         "`runs/round12/round12_analyze_docstring_2231.py.txt`), before any round-12 run or eval started. Seed 0 unless stated.", ""]
    res = {}
    part_a(L, res)
    part_b(L, res)
    L += ["## Part C — tH3ctx", "", "Not run (optional).", ""]
    part_d(L, res)
    p5(L, res)
    theory_inputs(L, res)
    all_runs(res)
    perf(L, res)
    answers(L, res)
    notes = os.path.join(OUT, "round12_notes.md")
    if os.path.exists(notes):
        L += [open(notes, encoding="utf8").read(), ""]
    cmd = os.path.join(OUT, "commands_extra.txt")
    if os.path.exists(cmd):
        L += ["## Commands (`runs/round12/commands_extra.txt`)", "", "```", open(cmd, encoding="utf8").read().rstrip(), "```", ""]
    json.dump(res, open(os.path.join(OUT, "round12_results.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "round12_summary.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
