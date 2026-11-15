"""Every number in the paper comes from here. Reads runs/round11/round11_results.json and runs/round12/round12_results.json
(written by round11_analyze.py / round12_analyze.py) and writes paper/numbers.tex (LaTeX macros + verdict switches) and
paper/numbers.md (macro, value, source). A value that is not in the JSON yet renders as a bold "??" (\\pending) and is
listed as missing in numbers.md, so the draft builds at any stage.

python paper/make_numbers.py
"""
import json
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R11 = json.load(open(os.path.join(ROOT, "runs", "round11", "round11_results.json"), encoding="utf8"))
P12 = os.path.join(ROOT, "runs", "round12", "round12_results.json")
R12 = json.load(open(P12, encoding="utf8")) if os.path.exists(P12) else {}
R13 = json.load(open(os.path.join(ROOT, "runs", "round13", "round13_results.json"), encoding="utf8"))
ABL = json.load(open(os.path.join(ROOT, "runs", "round13", "ablation.json"), encoding="utf8"))
SRC = {"r11": R11, "r12": R12, "r13": R13, "abl": ABL}
NATS_TO_BPB = 0.311414  # GPT-2 tokens: BPB = nats x 4,999,168 tokens / (23,160,213 bytes x ln 2)

MAC, MD, MISSING = [], [], []
VAL = {}  # macro -> rendered value (for claims.md)
SRCS = {}  # macro -> recorded source ("r11:<json path>", ...), the claims.md evidence column


def get(src, path):
    x = SRC[src]
    for k in path.split("/"):
        if isinstance(x, dict) and k in x:
            x = x[k]
        elif isinstance(x, list) and k.isdigit() and int(k) < len(x):
            x = x[int(k)]
        else:
            return None
    return x


def sci(s):
    """'6.717e+16' -> \\ensuremath{6.717\\times10^{16}}"""
    m, e = s.split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


SEEN = set()


def put(name, v, fmt=".4f", src="", scale=1.0, suffix=""):
    assert name.isalpha(), name
    assert name not in SEEN, f"macro {name} defined twice"
    SEEN.add(name)
    SRCS[name] = src
    if v is None:
        MAC.append(f"\\newcommand{{\\{name}}}{{\\pending}}")
        MD.append(f"| \\{name} | MISSING | {src} |")
        MISSING.append(name)
        VAL[name] = "??"
        return None
    if isinstance(v, str):
        MAC.append(f"\\newcommand{{\\{name}}}{{{v}}}")
        MD.append(f"| \\{name} | {v} | {src} |")
        VAL[name] = v
        return v
    x = v * scale
    if fmt.endswith("d"):
        x = int(round(x))
    s = format(x, fmt)
    if "e" in fmt:
        t = sci(s)
    else:
        t = s.replace("-", "\\textminus{}", 1) if s.startswith("-") else s
        t = t.replace(",", "{,}")
    MAC.append(f"\\newcommand{{\\{name}}}{{{t}{suffix}}}")
    sfx = suffix.replace("\\%", "%")
    MD.append(f"| \\{name} | {s}{sfx} | {src} |")
    VAL[name] = f"{s}{sfx}"
    return x


def g(name, src, path, fmt=".4f", scale=1.0, suffix=""):
    return put(name, get(src, path), fmt, f"{src}:{path}", scale, suffix)


def d(name, a, b, fmt="+.4f", src=""):
    put(name, None if a is None or b is None else a - b, fmt, src)


def switch(name, val, src):
    MAC.append(f"\\newif\\if{name}")
    if val:
        MAC.append(f"\\{name}true")
    MD.append(f"| \\if{name} | {val} | {src} |")


# ---------------------------------------------------------------- cost model / setup
plan = "partB/plan"
for arm, nm in (("B", "B"), ("H3s384", "Hs"), ("F8k384", "Feight"), ("F128k384", "Fbig"), ("H128k384", "Hbig")):
    g(f"Share{nm}", "r12", f"{plan}/{arm}/head_share", ".1f", 100, "\\%")
    g(f"Fpt{nm}", "r12", f"{plan}/{arm}/fpt", ".1f", 1e-6, "M")
g("HeadRatioFifty", "r12", "theory/head_ratio_50k", ".3f")
g("HeadRatioBig", "r12", "theory/head_ratio_131k", ".3f")
g("HeadRatioSmallTest", "r11", "partC/H3k8c4/head_flops_vs_B", ".3f")
shapes = get("r12", "theory/cost_shapes") or []
fl50 = [s for s in shapes if s["V"] == 50257]
if fl50:
    put("ShareSixHundredLo", min(s["flat_head_share"] for s in fl50), ".0f", "r12:theory/cost_shapes", 100, "\\%")
    put("ShareSixHundredHi", max(s["flat_head_share"] for s in fl50), ".0f", "r12:theory/cost_shapes", 100, "\\%")
    for s in fl50:
        tag = {1024: "A", 1280: "B", 1536: "C", 2048: "D"}[s["d"]]
        put(f"ShareShape{tag}", s["flat_head_share"], ".1f", "r12:theory/cost_shapes", 100, "\\%")
        put(f"ShareShapeTwo{tag}", s["two_level_head_share"], ".1f", "r12:theory/cost_shapes", 100, "\\%")
        put(f"ParamsShape{tag}", s["params_M"], ".0f", "r12:theory/cost_shapes", 1, "M")
    big = [s for s in shapes if s["V"] == 131072]
    for s in big:
        tag = {1024: "A", 1280: "B", 1536: "C", 2048: "D"}[s["d"]]
        put(f"ShareShapeBig{tag}", s["flat_head_share"], ".1f", "r12:theory/cost_shapes", 100, "\\%")
g("TargetFlops", "r12", "partB/rows/H3*-384 (575M)/flops", ".3e")
# model sizes over every run in the paper (n-gram / lookup arms are out of scope and excluded)
AR = [x for x in (get("r12", "all_runs") or []) if not x.get("ngram")]
pm = lambda f: max((x["params"] for x in AR if f(x)), default=None)  # noqa: E731
put("ParamsMaxFifty", pm(lambda x: x["vocab"] in (50257, 50304) and x["d"] <= 384), ".1f", "r12:all_runs, d<=384, 50k", 1e-6, "M")
put("ParamsWideMax", pm(lambda x: x["d"] == 512), ".1f", "r12:all_runs, d=512", 1e-6, "M")
put("ParamsBigVMax", pm(lambda x: x["vocab"] == 131072 and x["d"] == 384), ".1f", "r12:all_runs, 131k, d=384", 1e-6, "M")
# the one d=512 50k two-level run at the main budget (2026-11-14: reported next to the "131k beats 50k" comparison)
_w512 = [x for x in AR if x["run"] == "wsd_H3s512_359M_s0" and x.get("bpb") is not None and abs(x["flops"] / 6.716845e16 - 1) < 0.005]
put("BpbWideFifty", _w512[0]["bpb"] if _w512 else None, ".4f", "r12:all_runs wsd_H3s512_359M_s0 at 1x")

# ---------------------------------------------------------------- theory
th = "theory/test_heads"
for k, nm in (("flat", "Flat"), ("class_bias", "Cb"), ("hier", "Hier")):
    g(f"Rank{nm}", "r12", f"{th}/rank/{k}", "d")
    g(f"Bound{nm}", "r12", f"{th}/bound/{k}", "d")
g("TestGreedyOk", "r12", f"{th}/greedy_ok", "d")
g("TestGreedyN", "r12", f"{th}/greedy_n", "d")
g("TestGreedyClusters", "r12", f"{th}/greedy_clusters_mean", ".2f")
g("TestTV", "r12", f"{th}/tv_200k", ".4f")
g("PosNine", "r12", "theory/r9_fallback/positions", ",d")
g("FallbackRateNine", "r12", "theory/r9_fallback/fallback_rate", ".2f", 100, "\\%")
fr9 = get("r12", "theory/r9_fallback/fallback_rate")
put("ClustersNine", None if fr9 is None else 3 * (1 - fr9) + 1024 * fr9, ".2f", "r12:theory/r9_fallback (3(1-r)+1024 r)")
tg = "partB/gates/trained_wsd_H128k384_575M_s0/checks"
g("GreedyExactBig", "r12", f"{tg}/exact_match", ".4f")
g("GreedyPosBig", "r12", f"{tg}/positions", ",d")
g("FallbackBig", "r12", f"{tg}/fallback_rate", ".2f", 100, "\\%")
g("ClustersBig", "r12", f"{tg}/mean_clusters_expanded", ".1f")
g("FallbackHierOne", "r12", "partB/gates/hier/checks/greedy_fallback_h1", ".3f")
g("FallbackHierEight", "r12", "partB/gates/hier/checks/greedy_fallback_h8", ".3f")

# P5
for lab, nm in (("H3k8c4", "Small"), ("H3s384_575M", "Mid")):
    for st, sn in (("val", "Val"), ("E5a_copy1", "CopyOne"), ("E5a_copy2", "CopyTwo")):
        base = f"P5/jensen/{lab}/{st}"
        g(f"Jensen{nm}{sn}", "r12", f"{base}/delta_mean_tail", ".3f")
        g(f"JensenErr{nm}{sn}", "r12", f"{base}/error_mean_tail", "+.3f")
        g(f"JensenCorr{nm}{sn}", "r12", f"{base}/corr_offset_delta", ".2f")
    for i, b in enumerate(("[1024,4096)", "[4096,16384)", "[16384,V)")):
        g(f"JensenBucket{nm}{'ABC'[i]}", "r12", f"P5/jensen/{lab}/val/buckets/{b}/delta_mean", ".3f")
        g(f"JensenBucketErr{nm}{'ABC'[i]}", "r12", f"P5/jensen/{lab}/val/buckets/{b}/error_mean", "+.3f")

# ---------------------------------------------------------------- (a) equal-compute frontier, round 11 + D1
rb = "partB/rows"
g("BpbBThree", "r11", f"{rb}/B384 (300M)/bpb")
g("FlopsBThree", "r11", f"{rb}/B384 (300M)/flops", ".2e")
g("BpbHsThree", "r11", f"{rb}/H3*-384 (300M)/bpb")
g("BpbHsFive", "r11", f"{rb}/H3*-384 (575M)/bpb")
g("WallHsFive", "r11", f"{rb}/H3*-384 (575M)/wall_min", ".1f")
g("WallBThree", "r11", f"{rb}/B384 (300M)/wall_min", ".1f")
g("ParamsHs", "r11", f"{rb}/H3*-384 (575M)/params", ".1f", 1e-6, "M")
g("ParamsB", "r11", f"{rb}/B384 (300M)/params", ".1f", 1e-6, "M")
g("EqTokDelta", "r11", "E1/deltas/equal tokens (300M)/fineweb", "+.4f")
g("RoundElevenEqF", "r11", "E1/deltas/equal FLOPs (6.72e16 vs 6.39e16)/fineweb", "+.4f")
g("SmallEqTokNats", "r11", "partC/H3k8c4/d_vs_B_1200", "+.4f")
g("WideDelta", "r11", "partD/d_vs_H3s384_575", "+.4f")
D1 = "partD/D1"
for s in ("0", "1"):
    nm = "Zero" if s == "0" else "One"
    g(f"DOneB{nm}", "r12", f"{D1}/{s}/B384m")
    g(f"DOneH{nm}", "r12", f"{D1}/{s}/H3s")
    g(f"DOneD{nm}", "r12", f"{D1}/{s}/delta", "+.4f")
    g(f"DOneF{nm}", "r12", f"{D1}/{s}/flops_B", ".4e")
g("DOneMeanB", "r12", f"{D1}/mean_B")
g("DOneMeanH", "r12", f"{D1}/mean_H")
g("DOneSpreadB", "r12", f"{D1}/spread_B")
g("DOneSpreadH", "r12", f"{D1}/spread_H")
# (the headline and its switches are in the round-13 block: charged proxy clusters)
# envelope points
env = get("r11", "partA/envelope_flops") or []
put("EnvelopePoints", len(env) or None, "d", "r11:partA/envelope_flops")

# ---------------------------------------------------------------- (b) small vocabulary
g("BpbFeight", "r11", f"{rb}/F8k-384 (631M)/bpb")
g("BpbFfour", "r11", f"{rb}/F4k-384 (700M)/bpb")
g("BptFeight", "r11", f"{rb}/F8k-384 (631M)/bytes_per_token", ".3f")
g("BptFfour", "r11", f"{rb}/F4k-384 (700M)/bytes_per_token", ".3f")
g("BptGpt", "r11", f"{rb}/H3*-384 (575M)/bytes_per_token", ".3f")
g("SmallVocabDelta", "r11", "partB/delta_bpb_575", "+.4f")
g("SmallVocabDeltaThree", "r11", "partB/delta_bpb_300", "+.4f")
a = get("r11", f"{rb}/F4k-384 (700M)/bpb")
b = get("r11", f"{rb}/H3*-384 (575M)/bpb")
d("SmallVocabDeltaFour", b, a, "+.4f", "r11 H3* 575M - F4k 700M")

# ---------------------------------------------------------------- (c) big vocabulary
g("BptBig", "r12", "partB/tokenizer/val_bytes_per_token_docs", ".3f")
g("BigTrainS", "r12", "partB/tokenizer/train_s", ".0f")
g("BigRoundTrip", "r12", "partB/tokenizer/gate_roundtrip_failures", "d")
g("BigRoundTripDocs", "r12", "partB/tokenizer/gate_roundtrip_docs", ",d")
g("TopShareK", "r12", "partB/freq/top_share/8192", ".3f")
nz = get("r12", "partB/freq/nonzero_vocab")
put("UnseenIds", None if nz is None else 131072 - nz, ",d", "r12:131072 - partB/freq/nonzero_vocab")
pf = g("ProxyFlops", "r12", "partB/proxy_flops", ".2e")
put("ProxyShare", None if pf is None else pf / 6.716844637929216e16, ".1f", "r12:partB/proxy_flops / 6.717e16", 100, "\\%")
for lab, nm in (("F128k-384 (161M)", "Fbig"), ("H128k-384 (575M)", "Hbig"), ("H128k-384 (522M, proxy charged)", "Hcharged"),
                ("B384 matched (315M)", "Bm"), ("H3*-384 (575M)", "HsRow"), ("F128k-384 (84M)", "FbigThree"),
                ("H128k-384 (300M)", "HbigThree"), ("B384 matched (164M)", "BmThree"), ("H3*-384 (300M)", "HsRowThree"),
                ("F8k-384 (631M)", "FeightRow"), ("F4k-384 (700M)", "FfourRow")):
    base = f"{rb}/{lab}"
    g(f"Bpb{nm}", "r12", f"{base}/bpb")
    g(f"Tok{nm}", "r12", f"{base}/tokens", ".0f", 1e-6, "M")
    g(f"Flops{nm}", "r12", f"{base}/flops", ".3e")
    g(f"Vs{nm}", "r12", f"{base}/flops_vs_target", ".4f")
    g(f"Wall{nm}", "r12", f"{base}/wall_min", ".1f")
    g(f"Params{nm}", "r12", f"{base}/params", ".2f", 1e-6, "M")
    g(f"Emb{nm}", "r12", f"{base}/emb_params", ".2f", 1e-6, "M")
    g(f"Nonemb{nm}", "r12", f"{base}/nonemb_params", ".2f", 1e-6, "M")
    g(f"HS{nm}", "r12", f"{base}/head_share", ".1f", 100, "\\%")
    g(f"DecA{nm}", "r12", f"{base}/dec_ms_tok_b1", ".3f")
    g(f"DecAB{nm}", "r12", f"{base}/dec_ms_byte_b1", ".4f")
    g(f"DecB{nm}", "r12", f"{base}/dec_ms_tok_b32", ".4f")
    g(f"DecBB{nm}", "r12", f"{base}/dec_ms_byte_b32", ".5f")
V = get("r12", "partB/verdicts") or {}
for k, nm in (("big_vocab_for_free", "Free"), ("flat_cant_use_it", "FlatCant"), ("two_level_wins_128k", "Wins")):
    v = V.get(k)
    put(f"V{nm}Delta", None if not v else v["delta"], "+.4f", f"r12:partB/verdicts/{k}/delta")
    put(f"V{nm}Text", None if not v else ("holds" if v["holds"] else "does not hold"), src=f"r12:partB/verdicts/{k}/holds")
    put(f"V{nm}ChargedDelta", None if not v or v.get("charged_delta") is None else v["charged_delta"], "+.4f",
        f"r12:partB/verdicts/{k}/charged_delta")
    switch(f"v{nm.lower()}", bool(v and v["holds"]), f"r12:partB/verdicts/{k}/holds")
    switch(f"v{nm.lower()}known", bool(v), f"r12:partB/verdicts/{k} present")
for k, nm in (("H128k − H3* (3.505e16)", "Free"), ("F128k − B384 matched (3.505e16)", "FlatCant"), ("H128k − F128k (3.505e16)", "Wins")):
    put(f"Three{nm}", get("r12", "partB/deltas_300").get(k) if get("r12", "partB/deltas_300") else None, "+.4f",
        f"r12:partB/deltas_300/{k}")
# title (2026-11-13): head-centred framing, written only because round-13 experiment 4's pinned gate passed.
# 2026-11-14: experiment 5's pinned rule gave TIE, and its pinned consequence removes the head-centred title claim
# ("the title claim goes"; TIE and ADAPTIVE SOFTMAX WINS both): the head-centred title stays only if exp 5 PASSes.
_p = (R13.get("exp4") or {}).get("verdict") == "PASS"
_v5 = str((R13.get("exp5") or {}).get("verdict", ""))
put("PaperTitle", "A Two-Level Output Layer Makes a 131k Vocabulary Pay Off in Small Language Models at Equal Compute"
    if _p and _v5.startswith("PASS") else
    "Factorised Output Layers Let Small Language Models Use a 131k Vocabulary at Equal Compute"
    if _p or _v5 else "An Equal-Compute Study of Two-Level Output Layers in Small Language Models",
    src="r13:exp4/verdict == PASS, exp5/verdict")
ev = "partB/evals"
for lab, nm in (("F128k384_161M", "Fbig"), ("H128k384_575M", "Hbig"), ("B384m_s0", "Bm")):
    for c, cn in (("wikitext103", "Wiki"), ("pg19", "Pg"), ("blimp", "Blimp"), ("lambada_acc", "Lam"), ("lambada_wbpb", "LamB")):
        g(f"Ev{nm}{cn}", "r12", f"{ev}/{lab}/{c}", ".2f" if c in ("blimp", "lambada_acc") else ".4f")

# ---------------------------------------------------------------- (d) generalization
e1 = "E1/deltas/equal FLOPs (6.72e16 vs 6.39e16)"
g("OodWiki", "r11", f"{e1}/wikitext103", "+.4f")
g("OodPg", "r11", f"{e1}/pg19", "+.4f")
g("OodWikiTok", "r11", "E1/deltas/equal tokens (300M)/wikitext103", "+.4f")
g("OodPgTok", "r11", "E1/deltas/equal tokens (300M)/pg19", "+.4f")
for c, nm in (("wikitext103", "OodWikiM"), ("pg19", "OodPgM")):  # two-level vs the D1 FLOP-matched flat model (seed 0)
    h, b = get("r12", f"partD/D1/evals/H3s384_575M/{c}"), get("r12", f"partD/D1/evals/B384m_s0/{c}")
    put(nm, None if h is None or b is None else h - b, "+.4f", f"r12:partD/D1/evals H3s384_575M - B384m_s0 ({c})")
for lab, nm in (("B384", "B"), ("H3s384_575M", "Hs"), ("F8k384_631M", "Feight")):
    g(f"Wiki{nm}", "r11", f"E1/bpb/{lab}/wikitext103")
    g(f"Pg{nm}", "r11", f"E1/bpb/{lab}/pg19")
    g(f"Blimp{nm}", "r11", f"E2/acc/{lab}/overall", ".2f")
    g(f"Lam{nm}", "r11", f"E3/rows/{lab}/acc", ".1f", 100, "\\%")
    g(f"LamB{nm}", "r11", f"E3/rows/{lab}/word_bits_per_byte", ".3f")
for k, nm in (("equal tokens (300M)", "Tok"), ("equal FLOPs (6.72e16 vs 6.39e16)", "Flops"), ("d=128, equal tokens (1200M)", "Small"),
              ("F8k vs H3*, equal FLOPs", "Feight")):
    g(f"BlimpD{nm}", "r11", f"E2/deltas/{k}/delta", "+.2f")
    g(f"BlimpDMean{nm}", "r11", f"E2/deltas/{k}/delta_mean", "+.2f")
    g(f"LamD{nm}", "r11", f"E3/deltas/{k}/d_acc", "+.2f")
    g(f"LamBD{nm}", "r11", f"E3/deltas/{k}/d_bpb", "+.3f")
e2s1 = get("r11", "E2/acc_s1") or {}
for lab, nm in (("B384_s1", "B"), ("H3s384_575M_s1", "Hs"), ("H3s384_300M_s1", "HsThree"), ("B_r8_s1", "Br"), ("H3k8c4_s1", "Hk")):
    v = e2s1.get(lab)
    put(f"BlimpOne{nm}", (v.get("overall") if isinstance(v, dict) else v) if v is not None else None, ".2f", f"r11:E2/acc_s1/{lab}")
# grammar not hurt: two-level BLiMP no more than 1 point (the pinned band) below flat at every seed measured (equal FLOPs)
bl = [(get("r11", "E2/acc/H3s384_575M/overall"), get("r11", "E2/acc/B384/overall"))]
s1h, s1b = e2s1.get("H3s384_575M_s1"), e2s1.get("B384_s1")
if s1h is not None and s1b is not None:
    bl.append(((s1h.get("overall") if isinstance(s1h, dict) else s1h), (s1b.get("overall") if isinstance(s1b, dict) else s1b)))
switch("blimpok", all(h is not None and b is not None and h - b > -1.0 for h, b in bl), "r11:E2 two-level - flat > -1 point, all seeds")
A = "partA"
g("LamN", "r12", f"{A}/n", ",d")
g("LamEarlierN", "r12", f"{A}/n_earlier", ",d")
g("LamRestN", "r12", f"{A}/n_rest", ",d")
g("LamWholeFeight", "r12", f"{A}/models/F8k384_631M/bpb_whole")
g("LamWholeHs", "r12", f"{A}/models/H3s384_575M/bpb_whole")
g("LamWholeB", "r12", f"{A}/models/B384/bpb_whole")
g("LamWholeD", "r12", f"{A}/verdict/d_whole_F8k_minus_H3s", "+.4f")
g("LamAdvEarlier", "r12", f"{A}/verdict/adv_earlier", "+.3f")
g("LamAdvRest", "r12", f"{A}/verdict/adv_rest", "+.3f")
put("LamVerdict", get("r12", f"{A}/verdict/verdict"), src="r12:partA/verdict/verdict")

# ---------------------------------------------------------------- (e) knowledge
e4 = "E4/rows"
for lab, nm in (("bioB_20000", "BTwenty"), ("bioH_20000", "HTwenty"), ("bioB_100000", "BHundred"), ("bioH_100000", "HHundred"),
                ("bioH_100000_eqF", "HEqF")):
    g(f"Bits{nm}", "r11", f"{e4}/{lab}/bits", ",.0f")
    g(f"BitsNE{nm}", "r11", f"{e4}/{lab}/bits_per_nonemb_param", ".2f")
    g(f"Exp{nm}", "r11", f"{e4}/{lab}/exposures", ".1f")
r = get("r11", "E4/ratio_H_over_B")
put("KnowRatioTwenty", r.get("20000") if isinstance(r, dict) else None, ".3f", "r11:E4/ratio_H_over_B/20000")
put("KnowRatioHundred", r.get("100000") if isinstance(r, dict) else None, ".3f", "r11:E4/ratio_H_over_B/100000")
g("KnowEqF", "r11", "E4/eqF_ratio", ".3f")
g("DTwoRatioZero", "r12", "partD/D2/0/ratio", ".3f")
g("DTwoRatioOne", "r12", "partD/D2/1/ratio", ".3f")
# 2026-11-14 (outside review): the bios two-level models reuse the round-10 FineWeb-Edu clusters, built from the trained
# output vectors of wsd_B_r8_s0; that run's training FLOPs are not charged in D2. Reported as a multiple of the flat bios
# run's total training FLOPs (final records: tokens x flops_per_token/total, both runs).
def _final_any(path):
    for line in open(os.path.join(ROOT, *path.split("/")), encoding="utf8"):
        d = json.loads(line)
        if d.get("event") == "final":
            return d


_fr, _fb = _final_any("runs/round8/wsd_B_r8_s0.jsonl"), _final_any("runs/round11/wsd_bioB_100000_505M_s0.jsonl")
put("ClusterCostBios", _fr["tokens"] * _fr["flops_per_token"]["total"] / (_fb["tokens"] * _fb["flops_per_token"]["total"]),
    ".1f", "runs/round8/wsd_B_r8_s0 final FLOPs / runs/round11/wsd_bioB_100000_505M_s0 final FLOPs")
g("DTwoMean", "r12", "partD/D2/mean_ratio", ".3f")
g("DTwoBitsBOne", "r12", "partD/D2/1/bits_B", ",.0f")
g("DTwoBitsHOne", "r12", "partD/D2/1/bits_H_eqF", ",.0f")
switch("dtwo", bool(get("r12", "partD/D2/holds")), "r12:partD/D2/holds")
switch("dtwoknown", get("r12", "partD/D2/holds") is not None, "r12:partD/D2/holds present")

# ---------------------------------------------------------------- (f) copying
for k, nm in (("equal tokens (300M)", "Tok"), ("equal FLOPs (6.72e16 vs 6.39e16)", "Flops"), ("d=128, equal tokens (1200M)", "Small")):
    base = f"E5/deltas/{k}"
    g(f"CopyAOne{nm}", "r11", f"{base}/E5a_d_copy1_rare", "+.3f")
    g(f"CopyATwo{nm}", "r11", f"{base}/E5a_d_copy2_rare", "+.3f")
    g(f"CopyBOne{nm}", "r11", f"{base}/E5b_d_copy1", "+.2f")
    g(f"CopyBTwo{nm}", "r11", f"{base}/E5b_d_copy2", "+.2f")
for i, bk in enumerate("ABCDE"):
    g(f"BucketTok{bk}", "r11", f"E1/bucket_deltas/H3s384_300M-B384:fineweb/{i}", "+.3f")
    g(f"BucketSmall{bk}", "r11", f"E1/bucket_deltas/H3k8c4-B_r8:fineweb/{i}", "+.3f")
    g(f"BucketFlops{bk}", "r11", f"E1/bucket_deltas/H3s384_575M-B384:fineweb/{i}", "+.3f")

# ---------------------------------------------------------------- (g) negative results
g("RsDelta", "r11", "partC/tH3rs16/d_vs_H3k8c4_1200", "+.4f")
g("RsDeltaOne", "r11", "partC/seed1/d_vs_H3k8c4_s1", "+.4f")
g("RsMean", "r11", "partC/seed1/d_vs_H3k8c4_mean", "+.4f")
g("RsHit", "r11", "partC/hits/H3rs16/hit_fraction", ".0f", 100, "\\%")
g("RsVsB", "r11", "partC/tH3rs16/d_vs_B_1200", "+.4f")
g("RsMid", "r11", "partC/H3rs16_384/d_vs_H3s384_575", "+.4f")

# ---------------------------------------------------------------- (h) wall-clock and decode
# round-12 decode (same numbers as Table 2): B384 architecture vs H3*-384, batch 1
g("DecB", "r12", "partB/rows/B384 matched (315M)/dec_ms_tok_b1", ".3f")
g("DecHs", "r12", "partB/rows/H3*-384 (575M)/dec_ms_tok_b1", ".3f")
db, dh = get("r12", "partB/rows/B384 matched (315M)/dec_ms_tok_b1"), get("r12", "partB/rows/H3*-384 (575M)/dec_ms_tok_b1")
put("DecRatio", None if db is None else dh / db, ".2f", "r12 decode b1 H3*/B384")

# ---------------------------------------------------------------- round 13 (charged 50k proxy, compute scaling, adaptive softmax)
# runs/round13/round13_results.json (round13_analyze.py) and runs/round13/ablation.json (round13_ablation.py)
E1 = "exp1/1x"
for s, nm in (("0", "Zero"), ("1", "One")):
    g(f"MainFlat{nm}", "r13", f"{E1}/seeds/{s}/flat/bpb")
    g(f"MainTwo{nm}", "r13", f"{E1}/seeds/{s}/ch/bpb")
    g(f"MainTwoUnch{nm}", "r13", f"{E1}/seeds/{s}/unch/bpb")
    g(f"MainD{nm}", "r13", f"{E1}/seeds/{s}/d_ch", "+.4f")
    g(f"HalfD{nm}", "r13", f"exp1/0.5x/seeds/{s}/d_ch", "+.4f")
hd = g("Headline", "r13", f"{E1}/d_ch_mean", "+.4f")
put("HeadlineAbs", None if hd is None else abs(hd), ".4f", "r13:exp1/1x/d_ch_mean (abs)")
g("HeadlineUnch", "r13", f"{E1}/d_unch_mean", "+.4f")
g("HeadlineOld", "r12", f"{D1}/headline_delta", "+.4f")  # history: round-12 clusters from a full-budget flat model, uncharged
g("MainSpreadFlat", "r13", f"{E1}/spread_flat")
g("MainSpreadTwo", "r13", f"{E1}/spread_ch")
fz, fo = get("r13", f"{E1}/seeds/0/flat/bpb"), get("r13", f"{E1}/seeds/1/flat/bpb")
tz, to = get("r13", f"{E1}/seeds/0/ch/bpb"), get("r13", f"{E1}/seeds/1/ch/bpb")
put("MainFlatMean", (fz + fo) / 2, ".4f", "r13:exp1/1x mean flat")
tl50 = put("MainTwoMean", (tz + to) / 2, ".4f", "r13:exp1/1x mean two-level charged")
g("HalfDMean", "r13", "exp1/0.5x/d_ch_mean", "+.4f")
g("ClusterSourceEffect", "r13", "exp1/cluster_source_effect", "+.4f")
g("MainTokFlat", "r13", f"{E1}/seeds/0/flat/tokens", ".0f", 1e-6, "M")
g("MainTokTwo", "r13", f"{E1}/seeds/0/ch/tokens", ".0f", 1e-6, "M")
g("ProxyFiftyTok", "r13", "proxy/tokens", ".0f", 1e-6, "M")
g("ProxyFiftyFlops", "r13", "proxy/flops", ".2e")
g("ProxyFiftyShare", "r13", "proxy/share_of_1x", ".1f", 100, "\\%")
switch("headlinewin", hd is not None and hd <= -0.003, "r13:exp1/1x/d_ch_mean <= -0.003")
switch("headlineknown", hd is not None, "r13:exp1/1x/d_ch_mean present")
# experiment 2: gap vs training FLOPs (seed 0; 2-seed means at 0.5x and 1x)
for p, nm in (("0.5x", "Half"), ("1x", "One"), ("2x", "Two")):
    g(f"Gap{nm}", "r13", f"exp2/{p}/gap_ch", "+.4f")
    g(f"GapUnch{nm}", "r13", f"exp2/{p}/gap_unch", "+.4f")
    g(f"GapFlops{nm}", "r13", f"exp2/{p}/flops", ".3e")
    g(f"GapFlat{nm}", "r13", f"exp2/{p}/flat/bpb")
    g(f"GapTwoLevel{nm}", "r13", f"exp2/{p}/ch/bpb")
g("FitSlope", "r13", "exp2/fit/slope_per_doubling", "+.4f")
fs = g("FitFstar", "r13", "exp2/fit/F_star_x", ".1f")
put("FitFstarFlops", None if fs is None else fs * 6.716844637929216e16, ".1e", "r13:exp2/fit/F_star_x x 6.717e16")
g("ReproFlat", "r13", "exp2/repro_1x_flat", "+.4f")
# experiment 3, 50k: adaptive softmax
X3 = "exp3/50k"
for p, pn in (("1x", "One"), ("0.5x", "Half")):
    for s, nm in (("0", "Zero"), ("1", "One")):
        g(f"AsFifty{pn}{nm}", "r13", f"{X3}/{p}/seeds/{s}/as/bpb")
        g(f"AsTl{pn}{nm}", "r13", f"{X3}/{p}/seeds/{s}/d_tl_as", "+.4f")
    g(f"AsTl{pn}Mean", "r13", f"{X3}/{p}/d_tl_as_mean", "+.4f")
    g(f"AsFlat{pn}Mean", "r13", f"{X3}/{p}/d_as_flat_mean", "+.4f")
    g(f"AsTlUnch{pn}Mean", "r13", f"{X3}/{p}/d_tlunch_as_mean", "+.4f")
a50 = [get("r13", f"{X3}/1x/seeds/{s}/as/bpb") for s in "01"]
as50 = put("AsFiftyMean", sum(a50) / 2, ".4f", "r13:exp3/50k/1x mean adaptive softmax")
put("VerdictFifty", get("r13", f"{X3}/1x/verdict"), src="r13:exp3/50k/1x/verdict")
g("AsTok", "r13", f"{X3}/1x/seeds/0/as/tokens", ".0f", 1e-6, "M")
g("FptAs", "r13", "exp3/fpt/total", ".1f", 1e-6, "M")
fa, ha = get("r13", "exp3/fpt/total"), get("r13", "exp3/fpt/head")
put("ShareAs", ha / fa, ".1f", "r13:exp3/fpt/head / total", 100, "\\%")
_GA = {f: json.load(open(os.path.join(ROOT, "runs", "round13", f), encoding="utf8")) for f in ("gates_as.json", "gates_as_c49.json")}
put("FptAsBigOne", _GA["gates_as.json"]["AS128k384"]["A5"]["total"], ".1f", "runs/round13/gates_as.json AS128k384/A5/total", 1e-6, "M")
put("FptAsBigTwo", _GA["gates_as_c49.json"]["AS128k384c49"]["A5"]["total"], ".1f",
    "runs/round13/gates_as_c49.json AS128k384c49/A5/total", 1e-6, "M")
# experiment 5 (2026-11-14): the full-width arm
_GW = json.load(open(os.path.join(ROOT, "runs", "round13", "gates_as_c49w.json"), encoding="utf8"))["AS128k384c49w"]["A5"]
put("FptAsBigW", _GW["total"], ".1f", "runs/round13/gates_as_c49w.json AS128k384c49w/A5/total", 1e-6, "M")
put("ShareAsBigW", _GW["head_train"] / _GW["total"], ".1f", "runs/round13/gates_as_c49w.json A5 head_train / total", 100, "\\%")
put("HeadParamsAsBigW", _GW["head_params"], ".1f", "runs/round13/gates_as_c49w.json A5/head_params", 1e-6, "M")


def _params13(run):  # first logged "params" record of a round-13 run
    for line in open(os.path.join(ROOT, "runs", "round13", run + ".jsonl"), encoding="utf8"):
        d = json.loads(line)
        if "params" in d:
            return d["params"]["total"]


for nm, run in (("ParamsAsFifty", "wsd_AS384_624M_s0"), ("ParamsAsBigOne", "wsd_AS128k384_620M_s0"),
                ("ParamsAsBigTwo", "wsd_AS128k384c49_612M_s0"), ("ParamsAsBigW", "wsd_AS128k384c49w_540M_s0")):
    put(nm, _params13(run), ".1f", f"runs/round13/{run}.jsonl params/total", 1e-6, "M")
# the 131k two-level model at the same precision (seed 1 run; the same model as round 12's seed 0)
put("ParamsTlBig", _params13("wsd_H128k384_575M_s1"), ".1f", "runs/round13/wsd_H128k384_575M_s1.jsonl params/total", 1e-6, "M")


def _final13(run):  # the "final" record of a round-13 run (tokens = stable + decay of the last branch)
    for line in open(os.path.join(ROOT, "runs", "round13", run + ".jsonl"), encoding="utf8"):
        d = json.loads(line)
        if d.get("event") == "final":
            return d


put("TokAw", _final13("wsd_AS128k384c49w_540M_s0")["tokens"], ".0f", "runs/round13/wsd_AS128k384c49w_540M_s0.jsonl final/tokens",
    1e-6, "M")
# experiments 3 and 4, 131k (byte-aligned BPB; two-level charged at 1x, uncharged at 0.5x)
X4 = "exp4/bpb"
for s, nm in (("0", "Zero"), ("1", "One")):
    for k, kn in (("tl", "Tl"), ("a1", "AsA"), ("a2", "AsB"), ("tl05", "TlHalf"), ("a105", "AsAHalf"), ("a205", "AsBHalf")):
        for c, cn in (("fineweb", ""), ("wikitext103", "Wiki"), ("pg19", "Pg")):
            if k.endswith("05") and c != "fineweb":
                continue
            g(f"Big{kn}{cn}{nm}", "r13", f"{X4}/{k}/{s}/{c}")
g("BigFlat", "r13", "exp3/131k/1x/flat")
g("BigFlatHalf", "r13", "exp3/131k/0.5x/flat")
g("BigFlatWiki", "r13", "exp4/flat_s0/wikitext103")
g("BigFlatPg", "r13", "exp4/flat_s0/pg19")
dz = g("BigDZero", "r13", "exp4/delta_s0", "+.4f")
do = g("BigDOne", "r13", "exp4/delta_s1", "+.4f")
put("BigDMean", None if dz is None or do is None else (dz + do) / 2, "+.4f", "r13:exp4 mean of delta_s0, delta_s1")
put("BigGate", "PASS" if get("r13", "exp4/verdict") == "PASS" else get("r13", "exp4/verdict"), src="r13:exp4/verdict")
switch("bigpass", get("r13", "exp4/verdict") == "PASS", "r13:exp4/verdict == PASS")
B = lambda k, s, c="fineweb": get("r13", f"{X4}/{k}/{s}/{c}")  # noqa: E731
for s, nm in (("0", "Zero"), ("1", "One")):
    put(f"BigDOld{nm}", B("tl", s) - B("a1", s), "+.4f", f"r13:exp4 tl - a1, seed {s}")
    put(f"BigCfg{nm}", B("a2", s) - B("a1", s), "+.4f", f"r13:exp4 a2 - a1, seed {s}")
    put(f"BigDHalf{nm}", B("tl05", s) - B("a205", s), "+.4f", f"r13:exp4 tl05 - a205, seed {s} (two-level uncharged)")
    put(f"BigDWiki{nm}", B("tl", s, "wikitext103") - B("a2", s, "wikitext103"), "+.4f", f"r13:exp4 tl - a2 wikitext103, seed {s}")
    put(f"BigDPg{nm}", B("tl", s, "pg19") - B("a2", s, "pg19"), "+.4f", f"r13:exp4 tl - a2 pg19, seed {s}")
put("BigDFlat", B("tl", "0") - get("r13", "exp3/131k/1x/flat"), "+.4f", "r13:exp4 tl seed 0 - exp3 131k flat")
tl131 = put("BigTlMean", (B("tl", "0") + B("tl", "1")) / 2, ".4f", "r13:exp4 mean two-level 131k")
as131 = put("BigAsBMean", (B("a2", "0") + B("a2", "1")) / 2, ".4f", "r13:exp4 mean adaptive softmax cfg 2 131k")
# across vocabularies at 1x (observations: not pinned in round 13; seed s against seed s)
for s, nm in (("0", "Zero"), ("1", "One")):
    put(f"VocTl{nm}", B("tl", s) - get("r13", f"{E1}/seeds/{s}/ch/bpb"), "+.4f", f"r13:exp4 tl - exp1 two-level charged, seed {s}")
    put(f"VocAs{nm}", min(B("a1", s), B("a2", s)) - get("r13", f"{X3}/1x/seeds/{s}/as/bpb"), "+.4f",
        f"r13:best-of-two AS 131k - AS 50k, seed {s}")
put("VocTlMean", tl131 - tl50, "+.4f", "r13:two-level 131k mean - two-level 50k charged mean")
put("VocAsMean", as131 - as50, "+.4f", "r13:adaptive softmax cfg 2 131k mean - adaptive softmax 50k mean")
put("VocAsOneMean", (B("a1", "0") + B("a1", "1")) / 2 - as50, "+.4f", "r13:adaptive softmax cfg 1 131k mean - 50k mean")
put("VocFlat", get("r13", "exp3/131k/1x/flat") - fz, "+.4f", "r13:flat 131k - flat 50k, seed 0")
# experiment 5 (rule pinned 2026-11-13 21:40; result 2026-11-14): full-width adaptive softmax at 131k
W = lambda s, c="fineweb": get("r13", f"exp5/bpb/{s}/{c}")  # noqa: E731
for s, nm in (("0", "Zero"), ("1", "One")):
    g(f"BigAw{nm}", "r13", f"exp5/bpb/{s}/fineweb")
    g(f"BigAwWiki{nm}", "r13", f"exp5/bpb/{s}/wikitext103")
    g(f"BigAwPg{nm}", "r13", f"exp5/bpb/{s}/pg19")
    g(f"BigAwHalf{nm}", "r13", f"exp5/bpb_05/{s}")
    g(f"BigDw{nm}", "r13", f"exp5/delta_s{s}", "+.4f")
    g(f"BigWid{nm}", "r13", f"exp5/width_effect/{s}", "+.4f")
    put(f"BigDwWiki{nm}", B("tl", s, "wikitext103") - W(s, "wikitext103"), "+.4f", f"r13:exp4 tl - exp5 wikitext103, seed {s}")
    put(f"BigDwPg{nm}", B("tl", s, "pg19") - W(s, "pg19"), "+.4f", f"r13:exp4 tl - exp5 pg19, seed {s}")
    put(f"BigDwHalf{nm}", B("tl05", s) - get("r13", f"exp5/bpb_05/{s}"), "+.4f",
        f"r13:exp4 tl05 - exp5 bpb_05, seed {s} (two-level uncharged)")
put("BigDwMean", (get("r13", "exp5/delta_s0") + get("r13", "exp5/delta_s1")) / 2, "+.4f", "r13:exp5 mean of delta_s0, delta_s1")
put("BigWidMean", (get("r13", "exp5/width_effect/0") + get("r13", "exp5/width_effect/1")) / 2, "+.4f",
    "r13:exp5 mean width effect")
aw131 = put("BigAwMean", (W("0") + W("1")) / 2, ".4f", "r13:exp5 mean full-width adaptive softmax 131k")
put("VocAwMean", aw131 - as50, "+.4f", "r13:full-width adaptive softmax 131k mean - adaptive softmax 50k mean (reduced width)")
put("AwBestFifty", aw131 - min(tl50, as50), "+.4f", "r13:full-width AS 131k mean - best 50k mean (two-level or AS)")
put("TlBestFifty", tl131 - min(tl50, as50), "+.4f", "r13:two-level 131k mean - best 50k mean (two-level or AS)")
put("AwDFlat", W("0") - get("r13", "exp3/131k/1x/flat"), "+.4f", "r13:exp5 seed 0 - exp3 131k flat")
put("WideVerdict", str(get("r13", "exp5/verdict")), src="r13:exp5/verdict")
switch("widetie", str(get("r13", "exp5/verdict")).startswith("TIE"), "r13:exp5/verdict starts with TIE")
# ablation (d=128, equal tokens; BPB)
for k, nm in (("kmeans", "Kmeans"), ("kmeans+tied", "Tied"), ("kmeans+freq4k", "Freq"), ("freq4k+tied", "FreqTied"),
              ("freq8k+tied", "Eight"), ("freq4k+tied+4kcl", "FourCl"), ("final", "Final"), ("final+rescore16", "Rescore")):
    g(f"Abl{nm}", "abl", f"{k}/points/1200000000/delta_vs", "+.4f")
sp = [v["points"]["1200000000"]["spread"] for v in ABL.values() if v["points"]["1200000000"]["spread"] is not None]
put("AblSpreadLo", min(sp), ".4f", "abl: min two-seed spread at 1,200M")
put("AblSpreadHi", max(sp), ".4f", "abl: max two-seed spread at 1,200M")
g("AblFinalVsFlat", "abl", "final/points/1200000000/delta_vs_flat", "+.4f")
# nats -> BPB / bits conversions of round-11 numbers (GPT-2 tokens: BPB = nats x NATS_TO_BPB; per-token bits = nats / ln 2)
LN2 = 0.6931471805599453
for nm, path in (("SmallEqTokBpb", "partC/H3k8c4/d_vs_B_1200"), ("WideDeltaBpb", "partD/d_vs_H3s384_575"),
                 ("RsDeltaBpb", "partC/tH3rs16/d_vs_H3k8c4_1200"), ("RsDeltaOneBpb", "partC/seed1/d_vs_H3k8c4_s1"),
                 ("RsMeanBpb", "partC/seed1/d_vs_H3k8c4_mean"), ("RsMidBpb", "partC/H3rs16_384/d_vs_H3s384_575")):
    put(nm, get("r11", path), "+.4f", f"r11:{path} x NATS_TO_BPB", NATS_TO_BPB)
for k, nm in (("equal tokens (300M)", "Tok"), ("equal FLOPs (6.72e16 vs 6.39e16)", "Flops")):
    base = f"E5/deltas/{k}"
    for key, kn in (("E5a_d_copy1_rare", "CopyAOneBits"), ("E5a_d_copy2_rare", "CopyATwoBits"), ("E5b_d_copy2", "CopyBTwoBits")):
        put(f"{kn}{nm}", get("r11", f"{base}/{key}"), "+.2f", f"r11:{base}/{key} / ln 2", 1 / LN2)
for i, bk in enumerate("DE"):
    put(f"BucketSmallBits{bk}", get("r11", f"E1/bucket_deltas/H3k8c4-B_r8:fineweb/{3 + i}"), "+.2f",
        f"r11:E1/bucket_deltas/H3k8c4-B_r8:fineweb/{3 + i} / ln 2", 1 / LN2)
for st, sn in (("val", "Val"), ("E5a_copy2", "CopyTwo")):
    put(f"JensenBitsSmall{sn}", get("r12", f"P5/jensen/H3k8c4/{st}/delta_mean_tail"), ".2f",
        f"r12:P5/jensen/H3k8c4/{st}/delta_mean_tail / ln 2", 1 / LN2)

# ---------------------------------------------------------------- tables (generated LaTeX, \input by main.tex)
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)


def cell(v, fmt):
    if v is None:
        return "--"
    s = format(v, fmt)
    return s.replace("-", "\\textminus{}", 1) if s.startswith("-") else s


def fl(v):
    return "--" if v is None else sci(format(v, ".2e"))


runs = [r for r in (R12.get("all_runs") or []) if not r.get("ngram")]
L = ["\\begin{longtable}{llrrrrrrrr}", "\\toprule",
     "folder & run & seed & $d$ & $V$ & stable tok. & tokens & FLOPs & loss (nats) & BPB \\\\", "\\midrule", "\\endhead"]
for r in runs:
    nm = r["run"].replace("_", "\\_")
    L.append(f"\\texttt{{round{r['round']}}} & \\texttt{{{nm}}} & {r['seed']} & {r['d']} & {(50257 if r['vocab'] == 50304 else r['vocab']):,} & {r['stable_tokens'] / 1e6:.0f}M & "
             f"{r['tokens'] / 1e6:.0f}M & {fl(r['flops'])} & {cell(r['loss'], '.4f')} & {cell(r['bpb'], '.4f')} \\\\".replace(",", "{,}"))
L += ["\\bottomrule", "\\end{longtable}"]
open(os.path.join(TAB, "runs.tex"), "w", encoding="utf8", newline="\n").write("\n".join(L) + "\n")
put("RunCount", len({r["run"] for r in runs}) or None, "d", "r12:all_runs (n-gram arms excluded)")
put("BranchCount", len(runs) or None, "d", "r12:all_runs")

rows = get("r12", "partB/rows") or {}
L = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
     "arm (stable tokens) & tokens & FLOPs vs target & BPB & wall (min) & params (emb / non-emb) & bytes/tok & "
     "decode ms/tok (b1 / b32) & head share \\\\", "\\midrule"]
PLAIN = {"H3*-384": "two-level 50k, old clusters","B384 matched": "flat 50k", "F8k-384": "flat 8k", "F4k-384": "flat 4k",
         "F128k-384": "flat 131k", "H128k-384": "two-level 131k"}
for k, r in rows.items():
    lab = k
    for a_, b_ in PLAIN.items():
        lab = lab.replace(a_, b_)
    lab = lab.replace("proxy charged", "clusters charged")
    L.append(f"{lab} & {r['tokens'] / 1e6:.0f}M & {r['flops_vs_target']:.4f} & {cell(r['bpb'], '.4f')} & {r['wall_min']:.1f} & "
             f"{r['params'] / 1e6:.1f}M ({r['emb_params'] / 1e6:.1f} / {r['nonemb_params'] / 1e6:.1f}) & {r['bytes_per_token']:.3f} & "
             f"{cell(r['dec_ms_tok_b1'], '.3f')} / {cell(r['dec_ms_tok_b32'], '.4f')} & {100 * r['head_share']:.1f}\\% \\\\")
if not rows:
    L.append("\\multicolumn{9}{c}{pending} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "partb.tex"), "w", encoding="utf8", newline="\n").write("\n".join(L) + "\n")

G = get("r12", "partB/gates") or {}
L = ["\\begin{tabular}{p{0.16\\linewidth}p{0.06\\linewidth}p{0.68\\linewidth}}", "\\toprule", "head (131k) & result & check and measured value \\\\",
     "\\midrule"]
GSET = {"flat": "flat", "hier": "two-level", "trained": "two-level, trained"}  # 2026-11-13: plain names, display only
for gs, v in G.items():
    if not v:
        L.append(f"{gs.replace('_', chr(92) + '_')} & -- & not run \\\\")
        continue
    for k, c in v["checks"].items():
        if isinstance(c, dict) and "ok" in c:
            det = c["detail"].replace("_", "\\_").replace("%", "\\%").replace("|", "$|$").replace("#", "\\#")
            kk = re.sub(r"^G\d+ ", "", k).replace("F128k ", "").replace("_", chr(92) + "_").replace("==", "=")
            L.append(f"{GSET.get(gs.split('_')[0], gs.split('_')[0])} & {'OK' if c['ok'] else 'FAIL'} & {kk}: {det} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "gates.tex"), "w", encoding="utf8", newline="\n").write("\n".join(L) + "\n")

# head comparison (round 13): FineWeb-Edu val BPB at 0.5x and 1x training FLOPs, per seed
e1, e3, e4 = R13["exp1"], R13["exp3"], R13["exp4"]["bpb"]
f4 = lambda v: cell(v, ".4f")  # noqa: E731
L = ["\\begin{tabular}{llcccccc}", "\\toprule",
     "vocabulary & output layer & seed & 0.5$\\times$ & 1$\\times$ & WikiText-103 (1$\\times$) & PG-19 (1$\\times$) \\\\", "\\midrule"]
for sd in "01":
    L.append(f"50k & flat & {sd} & {f4(e1['0.5x']['seeds'][sd]['flat']['bpb'])} & {f4(e1['1x']['seeds'][sd]['flat']['bpb'])} & -- & -- \\\\")
for sd in "01":
    L.append(f"50k & adaptive softmax & {sd} & {f4(e3['50k']['0.5x']['seeds'][sd]['as']['bpb'])} & "
             f"{f4(e3['50k']['1x']['seeds'][sd]['as']['bpb'])} & -- & -- \\\\")
for sd in "01":
    L.append(f"50k & two-level & {sd} & {f4(e1['0.5x']['seeds'][sd]['ch']['bpb'])} & {f4(e1['1x']['seeds'][sd]['ch']['bpb'])} & -- & -- \\\\")
L.append("\\midrule")
fl0 = R13["exp4"]["flat_s0"]
L.append(f"131k & flat & 0 & {f4(e3['131k']['0.5x']['flat'])} & {f4(e3['131k']['1x']['flat'])} & {f4(fl0['wikitext103'])} & "
         f"{f4(fl0['pg19'])} \\\\")
for k, half, nm in (("a1", "a105", "adaptive softmax, setup 1"), ("a2", "a205", "adaptive softmax, setup 2"),
                    ("aw", "aw05", "adaptive softmax, full width"), ("tl", "tl05", "two-level")):
    for sd in "01":
        if k == "aw":  # experiment 5
            x, h = R13["exp5"]["bpb"][sd], f4(R13["exp5"]["bpb_05"][sd])
        else:
            x = e4[k][sd]
            h = f4(e4[half][sd]["fineweb"]) + ("$^\\dagger$" if k == "tl" else "")
        L.append(f"131k & {nm} & {sd} & {h} & {f4(x['fineweb'])} & {f4(x['wikitext103'])} & {f4(x['pg19'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "heads.tex"), "w", encoding="utf8", newline="\n").write("\n".join(L).replace(",", "{,}") + "\n")

# ablation (d=128, equal tokens; round13_ablation.py)
NAMES = {"flat": "(a) flat head", "random": "(b) 1,024 random clusters", "kmeans": "(c) 1,024 $k$-means clusters",
         "kmeans+tied": "(d) (c) + tied-mean cluster logit", "kmeans+freq4k": "(e) (c) + 4,096 frequent singletons",
         "freq4k+tied": "(f) (e) + tied-mean cluster logit", "freq8k+tied": "(g) (f) with 8,192 singletons",
         "freq4k+tied+4kcl": "(h) (f) with 4,096 clusters", "final": "(i) (g) with 4,096 clusters: final head",
         "final+rescore16": "(j) (i) + exact re-scoring, 16 clusters"}
L = ["\\begin{tabular}{lccccc}", "\\toprule",
     "row & vs & seeds & BPB & $\\Delta$ BPB vs row & training FLOPs \\\\", "\\midrule"]
for k, r in ABL.items():
    p = r["points"]["1200000000"]
    L.append(f"{NAMES[k]} & {NAMES[r['vs']][:3] if r['vs'] else '--'} & {len(r['runs'])} & {f4(p['mean'])} & "
             f"{cell(p['delta_vs'], '+.4f')} & {fl(p['flops'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "ablation.tex"), "w", encoding="utf8", newline="\n").write(
    "\n".join(L).replace(",", "{,}").replace("{,}024", "{,}024") + "\n")

# round-13 runs (the runs behind the headline, the compute study and the adaptive-softmax comparison)
R13RUNS = []
R13RUNS.append(("wsd_tBproxy_50M_s0", "50k cluster proxy (flat, $d=128$)", "0", None))
for sd in "01":
    R13RUNS.append((e1["1x"]["seeds"][sd]["flat"]["run"], "flat 50k (0.5$\\times$, 1$\\times$ branches)", sd, e1["1x"]["seeds"][sd]["flat"]["bpb"]))
    R13RUNS.append((e1["1x"]["seeds"][sd]["unch"]["run"], "two-level 50k, proxy clusters", sd, e1["1x"]["seeds"][sd]["ch"]["bpb"]))
    R13RUNS.append((e3["50k"]["1x"]["seeds"][sd]["as"]["run"], "adaptive softmax 50k", sd, e3["50k"]["1x"]["seeds"][sd]["as"]["bpb"]))
R13RUNS.append((R13["exp2"]["2x"]["flat"]["run"], "flat 50k, 2$\\times$", "0", R13["exp2"]["2x"]["flat"]["bpb"]))
R13RUNS.append((R13["exp2"]["2x"]["unch"]["run"], "two-level 50k, 2$\\times$", "0", R13["exp2"]["2x"]["ch"]["bpb"]))
R13RUNS.append(("wsd_AS128k384_620M_s0", "adaptive softmax 131k, setup 1", "0", e4["a1"]["0"]["fineweb"]))
R13RUNS.append(("wsd_AS128k384_620M_s1", "adaptive softmax 131k, setup 1", "1", e4["a1"]["1"]["fineweb"]))
R13RUNS.append(("wsd_AS128k384c49_612M_s0", "adaptive softmax 131k, setup 2", "0", e4["a2"]["0"]["fineweb"]))
R13RUNS.append(("wsd_AS128k384c49_612M_s1", "adaptive softmax 131k, setup 2", "1", e4["a2"]["1"]["fineweb"]))
R13RUNS.append(("wsd_AS128k384c49w_540M_s0", "adaptive softmax 131k, full width", "0", R13["exp5"]["bpb"]["0"]["fineweb"]))
R13RUNS.append(("wsd_AS128k384c49w_540M_s1", "adaptive softmax 131k, full width", "1", R13["exp5"]["bpb"]["1"]["fineweb"]))
R13RUNS.append(("wsd_H128k384_575M_s1", "two-level 131k (seed 0 is in \\texttt{round12})", "1", e4["tl"]["1"]["fineweb"]))
L = ["\\begin{tabular}{llcc}", "\\toprule", "run (\\texttt{runs/round13}) & arm & seed & BPB at the main point \\\\", "\\midrule"]
for nm, arm, sd, bp in R13RUNS:
    L.append(f"\\texttt{{{nm.replace('_', chr(92) + '_')}}} & {arm} & {sd} & {f4(bp)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "runs13.tex"), "w", encoding="utf8", newline="\n").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- follow-up: memory and speed (round12_perf.py; reported)
PF = get("r12", "perf") or {}
PA = (("B384", "flat 50k", "Bf"), ("H3s384", "two-level 50k", "Hf"), ("F8k384", "flat 8k", "Fe"),
      ("F128k384", "flat 131k", "Fb"), ("H128k384", "two-level 131k", "Hb"))
NUMW = {"1": "One", "16": "Sixteen", "32": "ThirtyTwo", "64": "SixtyFour", "128": "OneTwoEight"}
for kind, kn in (("train", "Tr"), ("score", "Sc"), ("decode", "De")):
    for arm, _, an in PA:
        for n, nw in NUMW.items():
            base = f"perf/{kind}/{arm}/{n}"
            if get("r12", base) is None:
                continue
            g(f"Perf{kn}{an}{nw}Tok", "r12", f"{base}/tok_s", ",.0f")
            g(f"Perf{kn}{an}{nw}Mem", "r12", f"{base}/{'peak_gb' if kind == 'train' else 'mem_est_gb'}", ".2f")  # score peak is set before the call
for V, vn in (("50k", "Fifty"), ("131k", "Big")):
    g(f"PerfRatio{vn}", "r12", f"perf/ratios_micro16/{V}/tok_s_ratio", ".2f")
    g(f"PerfSaved{vn}", "r12", f"perf/ratios_micro16/{V}/peak_saved_gb", ".2f")


def perf_cell(kind, arm, n):
    r = get("r12", f"perf/{kind}/{arm}/{n}")
    if r is None:
        return "--"
    if not r.get("ok"):
        return "OOM"
    m = r.get("peak_gb") if kind == "train" else r.get("mem_est_gb")
    return f"{r['tok_s'] / 1e3:,.0f}k / {m:.2f}"


if PF:
    L = []
    for kind, title, sizes in (("train", "training step, micro-batch rows", ("16", "32", "64")),
                               ("score", "forward-only scoring, rows", ("32", "64")),
                               ("decode", "KV-cache decode, batch", ("1", "32", "128"))):
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\")
        L.append("head & " + " & ".join(sizes) + " \\\\ \\midrule")
        for arm, nm, _ in PA:
            L.append(f"{nm} & " + " & ".join(perf_cell(kind, arm, n) for n in sizes) + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    T = ["\\begin{tabular}{lrrr}", "\\toprule"] + L + ["\\end{tabular}"]
    open(os.path.join(TAB, "perf.tex"), "w", encoding="utf8", newline="\n").write("\n".join(T).replace(",", "{,}") + "\n")
switch("perfknown", bool(PF), "r12 perf")


def perf_ratio(name, kind, n, fa, ha, key="tok_s"):
    f, h = get("r12", f"perf/{kind}/{fa}/{n}/{key}"), get("r12", f"perf/{kind}/{ha}/{n}/{key}")
    put(name, None if f is None or h is None else h / f, ".2f", f"r12:perf/{kind}/{{{ha}/{fa}}}/{n}/{key}")


for vn, fr, hr in (("Fifty", "B384 matched (315M)", "H3*-384 (575M)"), ("Big", "F128k-384 (161M)", "H128k-384 (575M)")):
    a_, b_ = get("r12", f"partB/rows/{fr}/fpt"), get("r12", f"partB/rows/{hr}/fpt")  # FLOPs/token: flat / two-level
    put(f"PerfPred{vn}", None if a_ is None or b_ is None else a_ / b_, ".2f", f"r12:partB/rows/{{{fr}/{hr}}}/fpt")
for vn, fa, ha in (("Fifty", "B384", "H3s384"), ("Big", "F128k384", "H128k384")):
    perf_ratio(f"PerfScRatio{vn}", "score", "64", fa, ha)
    perf_ratio(f"PerfScMemRatio{vn}", "score", "64", ha, fa, "mem_est_gb")  # flat / two-level
    perf_ratio(f"PerfDeRatio{vn}", "decode", "1", fa, ha)
    perf_ratio(f"PerfDeRatioB{vn}", "decode", "128", fa, ha)
    perf_ratio(f"PerfTrRatioSixtyFour{vn}", "train", "64", fa, ha)

# ---------------------------------------------------------------- claims.md (claim, section, number, seeds, pinned, evidence)
# pinned: "yes" = decision rule written in an analysis docstring before the run; "no" = post-hoc, worded as an observation;
# "n/a" = a derivation or a reported measurement with no decision rule.
R11J, R12J = "runs/round11/round11_results.json", "runs/round12/round12_results.json"
R13J, ABLJ = "runs/round13/round13_results.json", "runs/round13/ablation.json"
CLAIMS = [
    # ---- main results (round 13)
    ("At 50k and equal training FLOPs ({TargetFlops}), with the cluster proxy's FLOPs charged, two-level minus flat is {Headline} BPB (2-seed mean; seeds {MainDZero} / {MainDOne}).",
     "abstract, 5.1", "2", "yes", R13J + " exp1"),
    ("History: with clusters from a full-budget flat model whose cost was not charged, an earlier draft reported {HeadlineOld}.",
     "5.1", "2", "yes", R12J + " partD/D1"),
    ("At 50k the two-level head ties adaptive softmax: two-level minus adaptive softmax is {AsTlOneMean} BPB at 1x (seeds {AsTlOneZero} / {AsTlOneOne}), verdict '{VerdictFifty}'.",
     "abstract, 5.1", "2", "yes", R13J + " exp3/50k"),
    ("At 50k adaptive softmax minus flat is {AsFlatOneMean} BPB at 1x (2 seeds).", "5.1", "2", "yes", R13J + " exp3/50k"),
    ("At 131k and 1x, two-level minus the better of adaptive-softmax setups 1-2 is {BigDZero} (seed 0) and {BigDOne} (seed 1); pinned gate {BigGate}.",
     "5.2", "2", "yes", R13J + " exp4"),
    ("Observation: the second adaptive-softmax setup improves on the first by {BigCfgZero} / {BigCfgOne} BPB at 131k (seeds 0 / 1).",
     "5.2", "2", "no", R13J + " exp4"),
    ("At 131k and 1x, two-level (charged) minus flat is {BigDFlat} BPB (seed 0; flat 131k has one seed).", "5.2", "1", "yes",
     R13J + " exp3/131k"),
    ("Observation: going from 50k to 131k at 1x changes two-level by {VocTlMean} BPB (2 seeds), the better adaptive softmax by {VocAsMean} (2 seeds) and flat by {VocFlat} (seed 0).",
     "5.2", "1-2", "no", R13J + " exp1, exp3, exp4"),
    ("At 131k and 1x, two-level minus the full-width adaptive softmax is {BigDwZero} (seed 0) and {BigDwOne} (seed 1); pinned rule: {WideVerdict}.",
     "abstract, 5.2", "2", "yes", R13J + " exp5"),
    ("Observation: removing the tail projection improves adaptive softmax (setup 2) by {BigWidZero} / {BigWidOne} BPB at 131k (seeds 0 / 1).",
     "abstract, 5.2", "2", "no", R13J + " exp5/width_effect"),
    ("Observation: the 131k full-width adaptive softmax and two-level models are {AwBestFifty} / {TlBestFifty} BPB below the best d = 384 50k model of the main comparison at the same budget (2-seed means); the d = 512 50k two-level run reaches {BpbWideFifty} (seed 0).",
     "abstract, 5.2", "2", "no", R13J + " exp1, exp3, exp4, exp5"),
    ("Observation: at 131k two-level is ahead of the full-width adaptive softmax on WikiText-103 ({BigDwWikiZero} / {BigDwWikiOne}) and PG-19 ({BigDwPgZero} / {BigDwPgOne}).",
     "5.2", "2", "no", R13J + " exp4, exp5"),
    ("Observation: at 131k and 0.5x (two-level uncharged), two-level minus the full-width adaptive softmax is {BigDwHalfZero} / {BigDwHalfOne}.",
     "5.3", "2", "no", R13J + " exp4, exp5"),
    ("Parameters at 131k: two-level {ParamsTlBig}, full-width adaptive softmax {ParamsAsBigW} (its head {HeadParamsAsBigW}).",
     "abstract, 4", "n/a", "n/a", "runs/round13/wsd_AS128k384c49w_540M_s0.jsonl; runs/round13/gates_as_c49w.json"),
    ("Observation: at 131k two-level is ahead of the second adaptive-softmax setup on WikiText-103 ({BigDWikiZero} / {BigDWikiOne}) and PG-19 ({BigDPgZero} / {BigDPgOne}).",
     "5.2", "2", "no", R13J + " exp4"),
    ("Observation: at 131k and 0.5x (two-level uncharged), two-level minus the second adaptive-softmax setup is {BigDHalfZero} / {BigDHalfOne}.",
     "5.3", "2", "no", R13J + " exp4"),
    ("The 50k two-level minus flat gap shrinks with compute: {GapHalf} (0.5x), {GapOne} (1x), {GapTwo} (2x), seed 0; 2-seed means {HalfDMean} and {Headline} at 0.5x and 1x.",
     "abstract, 5.3", "1-2", "yes", R13J + " exp2"),
    ("Observation: a line through the three points crosses zero near {FitFstar}x the main budget; this is an extrapolation, not a result.",
     "5.3", "1", "no", R13J + " exp2/fit"),
    # ---- vocabulary, equal tokens, ablation
    ("At equal FLOPs the two-level head (old clusters) beats flat 8k / 4k vocabularies by {SmallVocabDelta} / {SmallVocabDeltaFour} BPB.",
     "5.4", "1", "yes", R11J + " partB"),
    ("At equal tokens (300M, d = 384) the two-level head is {EqTokDelta} BPB behind the flat head; at d = 128 and 1.2B tokens, {SmallEqTokBpb} BPB.",
     "5.5", "1", "yes", R11J),
    ("Observation (ablation, d = 128, equal tokens; existing runs, no rule pinned): k-means vs random clusters {AblKmeans}; 4,096 frequent singletons {AblFreq}; tied-mean cluster logits {AblTied}; re-scoring {AblRescore} BPB.",
     "5.6", "1-2", "no", ABLJ),
    ("(g) Exact re-scoring of the top 16 clusters vs the plain two-level head: {RsDeltaBpb} / {RsDeltaOneBpb} BPB (seeds 0 / 1), mean {RsMeanBpb}; pinned verdict 'no help'.",
     "5.6", "2", "yes", R11J + " partC"),
    # ---- beyond loss (old-cluster 50k model, round 11/12)
    ("Observation: against the FLOP-matched flat model (seed 0) the two-level head (old clusters) differs by {OodWikiM} BPB on WikiText-103 and {OodPgM} on PG-19.",
     "5.7", "1", "no", R12J + " partD/D1/evals"),
    ("Near-equal FLOPs (flat at 6.39e16, 5% fewer): {OodWiki} / {OodPg} BPB (pinned rule: holds).", "5.7 footnote", "1", "yes", R11J + " E1"),
    ("BLiMP: two-seed mean differences {BlimpDMeanTok} (equal tokens) and {BlimpDMeanFlops} (near-equal FLOPs) points, inside the +-1 band: no difference.",
     "5.7", "2", "yes", R11J + " E2"),
    ("LAMBADA: the 8k model's last-word advantage is {LamAdvEarlier} bits/byte when the word appears earlier and {LamAdvRest} otherwise; verdict '{LamVerdict}'.",
     "5.7", "1", "yes", R12J + " partA"),
    ("At equal exposures the two-level head stores {KnowRatioTwenty}x (20k people) and {KnowRatioHundred}x (100k people) the bits of the flat head.",
     "5.7", "1", "yes", R11J + " E4"),
    ("At equal FLOPs the two-level head stores {DTwoRatioZero}x (seed 0) and {DTwoRatioOne}x (seed 1) the bits of the flat head; its clusters are not charged (building them took {ClusterCostBios}x the flat run's FLOPs).",
     "5.7", "2", "yes", R12J + " partD/D2"),
    ("Rare-target gap at equal tokens on a repeated window: {CopyAOneBitsTok} bits/token (copy 1), {CopyATwoBitsTok} (copy 2).",
     "5.7", "1", "yes", R11J + " E5"),
    # ---- cost, speed, decoding
    ("The two-level head costs {HeadRatioFifty}x the flat head's FLOPs at V = 50k and {HeadRatioBig}x at V = 131k.",
     "3", "n/a", "n/a", R12J + " theory/head_ratio_*"),
    ("At d = 384 the flat head is {ShareB} of training FLOPs; the two-level head {ShareHs}; adaptive softmax {ShareAs}.",
     "3", "n/a", "n/a", R12J + " partB/plan; " + R13J + " exp3/fpt"),
    ("Observation (no rule pinned): training at 16 rows, the two-level head runs at {PerfRatioFifty}x (50k) and {PerfRatioBig}x (131k) the flat head's tokens/s on this machine.",
     "5.8, appendix", "n/a", "no", R12J + " perf"),
    ("Exact greedy on the trained 131k two-level model: exact-match {GreedyExactBig} on {GreedyPosBig} positions, {ClustersBig} clusters expanded.",
     "3, appendix", "1", "yes", "runs/round12/gates_trained_wsd_H128k384_575M_s0.json"),
    ("Observation: at ~600M parameters with a 50k vocabulary the flat head is {ShareShapeA}-{ShareShapeD} of training FLOPs (cost model).",
     "6", "n/a", "no", R12J + " theory/cost_shapes"),
]
import re  # noqa: E402

CL = ["# Claims (generated by paper/make_numbers.py)", "",
      "pinned: yes = decision rule written in an analysis docstring before the run; no = post-hoc, worded as an observation; "
      "n/a = derivation or reported measurement. Numbers marked ?? are not in the results JSON yet.", "",
      "| # | claim | section | number | seeds | pinned | evidence file |", "|---|---|---|---|---|---|---|"]
for i, (text, sec, seeds, pinned, ev) in enumerate(CLAIMS, 1):
    names = re.findall(r"\{([A-Za-z]+)\}", text)
    for nm in names:
        assert nm in VAL, f"claims.md: unknown macro {nm}"
    assert pinned != "no" or text.startswith("Observation"), text
    shown = re.sub(r"\{([A-Za-z]+)\}", lambda m: VAL[m.group(1)], text)
    nums = "; ".join(f"\\{nm} = {VAL[nm]}" for nm in names)
    # evidence: each macro's recorded source (results JSON + key path), plus the raw file the analysis read, if named
    files = {"r11": R11J, "r12": R12J, "r13": R13J, "abl": ABLJ}
    evs = []
    for nm in names:
        sr = SRCS.get(nm, "")
        pre, _, path = sr.partition(":")
        e = f"{files[pre]} → {path}" if pre in files and path else (sr or "?")
        if e not in evs:
            evs.append(e)
    if not ev.startswith(("runs/round11/round11_results", "runs/round12/round12_results", "runs/round13/")):
        evs.append("raw: " + ev)
    evc = "<br>".join(f"`{e}`" for e in evs)
    CL.append(f"| {i} | {shown.replace('|', '/')} | {sec} | {nums} | {seeds} | {pinned} | {evc} |")
open(os.path.join(HERE, "claims.md"), "w", encoding="utf8", newline="\n").write("\n".join(CL) + "\n")

# ---------------------------------------------------------------- write
pre =["% generated by paper/make_numbers.py -- do not edit", "\\providecommand{\\pending}{\\textbf{??}}"]
open(os.path.join(HERE, "numbers.tex"), "w", encoding="utf8", newline="\n").write("\n".join(pre + MAC) + "\n")
open(os.path.join(HERE, "numbers.md"), "w", encoding="utf8", newline="\n").write(
    "# Paper numbers (generated by paper/make_numbers.py)\n\n"
    f"{len(MD)} entries, {len(MISSING)} missing: {', '.join(MISSING) or 'none'}\n\n| macro | value | source |\n|---|---|---|\n"
    + "\n".join(MD) + "\n")
print(f"{len(MD)} macros, {len(MISSING)} missing")
