"""Export paper/main.tex to paper/paper.md (no pandoc on this machine; a converter for the constructs main.tex uses).

Macros come from paper/numbers.tex (run make_numbers.py first), \\if switches are resolved, citations become author-year
from refs.bib, figures point at figs/*.png, and the generated tables are converted to Markdown tables. Math stays as $...$.

python paper/export_md.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(HERE, p), encoding="utf8").read()  # noqa: E731


def group(s, i):
    """s[i] == '{': return (content, index after the matching '}')."""
    assert s[i] == "{", s[i:i + 30]
    depth, j = 0, i
    while True:
        if s[j] == "\\":
            j += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1


# ---------------------------------------------------------------- macros and switches
NUM = rd("numbers.tex")
MACROS, SWITCH = {}, {}
i = 0
while True:
    i = NUM.find("\\newcommand{\\", i)
    if i < 0:
        break
    name, j = group(NUM, i + len("\\newcommand"))
    val, i = group(NUM, j)
    MACROS[name[1:]] = val
for m in re.finditer(r"\\newif\\if([A-Za-z]+)", NUM):
    SWITCH[m.group(1)] = False
for m in re.finditer(r"\\([A-Za-z]+)true", NUM):
    if m.group(1) in SWITCH:
        SWITCH[m.group(1)] = True
MACROS["pending"] = "**??**"

src = rd("main.tex")
src = re.sub(r"(?<!\\)%.*", "", src)  # comments
# layout-only wrappers that have no Markdown meaning
src = src.replace("\\resizebox{\\linewidth}{!}{\\input{tables/partb}}", "\\input{tables/partb}")
src = src.replace("{\\sloppy ", "").replace(".\\par}", ".")
body = src[src.index("\\begin{document}") + len("\\begin{document}"):src.index("\\end{document}")]


def resolve_ifs(s):
    out, stack, k = [], [], 0  # stack of (active_before, cond, in_else)
    active = True
    pat = re.compile(r"\\if([A-Za-z]+)(?![A-Za-z])|\\else(?![A-Za-z])|\\fi(?![A-Za-z])")
    for m in pat.finditer(s):
        if m.group(1) is not None and m.group(1) not in SWITCH:
            continue
        if active:
            out.append(s[k:m.start()])
        k = m.end()
        tok = m.group(0)
        if m.group(1) is not None:
            stack.append((active, SWITCH[m.group(1)]))
            active = active and SWITCH[m.group(1)]
        elif tok.startswith("\\else"):
            before, cond = stack[-1]
            active = before and not cond
        else:
            active = stack.pop()[0]
    out.append(s[k:])
    assert not stack, "unbalanced \\if"
    return "".join(out)


def expand(s):
    for _ in range(5):
        s2 = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: MACROS.get(m.group(1), m.group(0)) if m.group(1) in MACROS else m.group(0), s)
        if s2 == s:
            break
        s = s2
    return s


body = resolve_ifs(body)
body = expand(body)
title = expand(MACROS["PaperTitle"])

# ---------------------------------------------------------------- citations
BIB = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", rd("refs.bib"), re.S):
    key, fields = m.group(1).strip(), m.group(2)
    f = {}
    for fm in re.finditer(r"(\w+)\s*=\s*\{", fields):
        if any(a <= fm.start() < b for a, b in f.get("_spans", [])):
            continue
        v, j = group(fields, fm.end() - 1)
        f.setdefault("_spans", []).append((fm.start(), j))
        v = re.sub(r"\\[`'^\"~=.uvHc]\s*\{?\\?([A-Za-z])\}?", lambda x: {"'": {"E": "É", "e": "é", "i": "í", "a": "á"}}
                   .get("'", {}).get(x.group(1), x.group(1)), v)  # accents: keep the base letter (é/í/á/É for \')
        f[fm.group(1).lower()] = re.sub(r"[{}]", "", v).strip()
    authors = [a.strip() for a in f.get("author", "").split(" and ")]
    last = [a.split(",")[0] if "," in a else a.split()[-1] for a in authors if a]
    short = last[0] if len(last) == 1 else (f"{last[0]} and {last[1]}" if len(last) == 2 else f"{last[0]} et al.")
    BIB[key] = dict(short=short, year=f.get("year", "n.d."), full=f)
CITED = []


def cite(kind, keys):
    ks = [k.strip() for k in keys.split(",")]
    for k in ks:
        assert k in BIB, f"unknown citation {k}"
        if k not in CITED:
            CITED.append(k)
    if kind == "citet":
        return "; ".join(f"{BIB[k]['short']} ({BIB[k]['year']})" for k in ks)
    return "(" + "; ".join(f"{BIB[k]['short']}, {BIB[k]['year']}" for k in ks) + ")"


body = re.sub(r"\\(citep|citet)\{([^}]*)\}", lambda m: cite(m.group(1), m.group(2)), body)

# ---------------------------------------------------------------- labels -> numbers
labels, sec_n, fig_n, tab_n, app = {}, 0, 0, 0, False
for m in re.finditer(r"\\appendix|\\section\{[^}]*\}(\\label\{([^}]*)\})?|\\begin\{figure\}|\\begin\{table\}|\\label\{([^}]*)\}", body):
    t = m.group(0)
    if t == "\\appendix":
        app, sec_n = True, 0
    elif t.startswith("\\section"):
        sec_n += 1
        if m.group(2):
            labels[m.group(2)] = chr(64 + sec_n) if app else str(sec_n)
    elif t.startswith("\\begin{figure"):
        fig_n += 1
        cur = str(fig_n)
    elif t.startswith("\\begin{table"):
        tab_n += 1
        cur = str(tab_n)
    elif m.group(3):
        labels[m.group(3)] = cur
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "??"), body)


# ---------------------------------------------------------------- tables
def tab_to_md(tex):
    tex = re.sub(r"\\(toprule|midrule|bottomrule|endhead|endfirsthead|endfoot|endlastfoot|hline)", "", tex)
    m = re.search(r"\\begin\{(tabular|longtable)\}", tex)
    if m:  # drop the column spec (may nest braces, e.g. p{0.06\linewidth})
        j = m.end()
        while j < len(tex) and tex[j] == "{":
            j = group(tex, j)[1]
        tex = tex[:m.start()] + tex[j:]
    tex = re.sub(r"\\end\{(tabular|longtable)\}", "", tex)
    rows = [r.strip() for r in tex.split("\\\\") if r.strip()]
    rows = [[c.strip() for c in r.split("&")] for r in rows]
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    seen, uniq = set(), []
    for r in rows:  # longtable repeats its header on every page
        if r == rows[0] and seen:
            continue
        seen.add(tuple(r))
        uniq.append(r)
    out = ["| " + " | ".join(uniq[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(r) + " |" for r in uniq[1:]]
    return "\n\n" + "\n".join(out) + "\n\n"


body = re.sub(r"\\input\{(tables/[^}]*)\}", lambda m: tab_to_md(expand(rd(m.group(1) + ".tex"))), body)


# ---------------------------------------------------------------- environments
def figure(m):
    f = m.group(0)
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", f)
    cap = re.search(r"\\caption\{", f)
    c = group(f, cap.end() - 1)[0] if cap else ""
    num = labels.get((re.search(r"\\label\{([^}]*)\}", f) or [None, ""])[1], "")
    kind = "Figure" if "figure" in f[:15] else "Table"
    pics = "\n".join(f"![]({p.replace('.pdf', '.png')})" for p in imgs)
    if kind == "Table" and "\\begin{tabular}" in f:  # an inline tabular (generated tables were \input and converted above)
        f = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", lambda t: tab_to_md(t.group(0)), f, flags=re.S)
        f = re.sub(r"\\small|\\footnotesize", "", f)
    rest = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}|\\caption\{.*|\\label\{[^}]*\}|\\centering|\\hfill|"
                  r"\\scriptsize|\\setlength\{[^}]*\}\{[^}]*\}|\\begin\{(figure|table)\}(\[[^\]]*\])?|\\end\{(figure|table)\}", "",
                  f, flags=re.S) if kind == "Table" else ""
    return f"\n\n{pics}{rest.strip()}\n\n*{kind} {num}: {c.strip()}*\n\n"


body = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{(figure|table)\}", figure, body, flags=re.S)
prop_n = [0]


def prop(m):
    prop_n[0] += 1
    return f"\n\n**Proposition {prop_n[0]} ({m.group(1)}).** "


body = re.sub(r"\\begin\{proposition\}\[([^\]]*)\]", prop, body)
body = body.replace("\\end{proposition}", "\n\n")
rem_n = [0]


def rem(m):  # 2026-11-13: remarks replace the propositions
    rem_n[0] += 1
    return f"\n\n**Remark {rem_n[0]} ({m.group(1)}).** "


body = re.sub(r"\\begin\{remark\}\[([^\]]*)\]", rem, body)
body = body.replace("\\end{remark}", "\n\n")
fn_n, fns = [0], []


def footnote(m):  # footnotes -> [^n] with the text collected at the end of the body
    fn_n[0] += 1
    fns.append(f"[^{fn_n[0]}]: {m.group(1)}")
    return f"[^{fn_n[0]}]"


body = re.sub(r"\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}", footnote, body)
if fns:
    body += "\n\n" + "\n\n".join(fns) + "\n"
body = body.replace("\\begin{proof}", "\n\n*Proof.* ").replace("\\end{proof}", " ∎\n\n")
body = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n\n", body).replace("\\end{abstract}", "\n")


def lists(s):
    out, stack = [], []
    for line in s.split("\n"):
        st = line.strip()
        if st.startswith("\\begin{itemize}") or st.startswith("\\begin{enumerate}"):
            stack.append(["- ", 0] if "itemize" in st else ["1. ", 0])
            out.append("")
            continue
        if st.startswith("\\end{itemize}") or st.startswith("\\end{enumerate}"):
            stack.pop()
            out.append("")
            continue
        if stack and st.startswith("\\item"):
            stack[-1][1] += 1
            mark = "- " if stack[-1][0] == "- " else f"{stack[-1][1]}. "
            out.append("  " * (len(stack) - 1) + mark + st[len("\\item"):].strip())
            continue
        out.append(("  " * len(stack) + st) if stack and st else line)
    return "\n".join(out)


body = lists(body)
body = re.sub(r"\\begin\{center\}|\\end\{center\}|\\setlength\{[^}]*\}\{[^}]*\}", "", body)
body = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n\n$$" + " ".join(m.group(1).split()) + "$$\n\n", body, flags=re.S)


# ---------------------------------------------------------------- inline commands (outside math)
def inline(s):
    s = re.sub(r"\\section\{([^}]*)\}(\\label\{[^}]*\})?", lambda m: f"\n\n## {m.group(1)}\n\n", s)
    s = re.sub(r"\\subsection\{([^}]*)\}(\\label\{[^}]*\})?", lambda m: f"\n\n### {m.group(1)}\n\n", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    for cmd, l, r in (("paragraph", "\n\n**", "**"), ("emph", "*", "*"), ("textbf", "**", "**"), ("texttt", "`", "`"),
                      ("textit", "*", "*"), ("url", "<", ">")):
        while True:
            m = re.search(r"\\" + cmd + r"\{", s)
            if not m:
                break
            c, j = group(s, m.end() - 1)
            if cmd == "texttt":
                c = c.replace("\\_", "_")
            s = s[:m.start()] + l + c + r + s[j:]
    s = re.sub(r"\\ensuremath\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", lambda m: "$" + m.group(1) + "$", s)
    rep = [("\\textminus{}", "−"), ("\\noindent", ""), ("\\maketitle", ""), ("\\appendix", "\n\n# Appendix\n\n"),
           ("\\bibliographystyle{plainnat}", ""), ("\\bibliography{refs}", "\n\n@@REFS@@\n\n"), ("\\%", "%"), ("\\_", "_"),
           ("\\#", "#"), ("\\&", "&"), ("{,}", ","), ("\\S", "§"), ("---", "—"), ("--", "–"), ("``", "“"), ("''", "”"),
           ("i.e.\\ ", "i.e. "), ("\\,", " "), ("~", " "), ("\\dots", "…"), ("\\scriptsize", "")]
    for a, b in rep:
        s = s.replace(a, b)
    s = re.sub(r"\{\}", "", s)
    return s


MATH = []


def stash(m):
    MATH.append(m.group(0))
    return f"@@M{len(MATH) - 1}@@"


body = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", stash, body, flags=re.S)  # keep math verbatim
body = inline(body)
body = re.sub(r"@@M(\d+)@@", lambda m: MATH[int(m.group(1))], body)
body = re.sub(r"^\|(—\|)+$", lambda m: "|" + "---|" * m.group(0).count("—"), body, flags=re.M)  # undo the dash rule
body = re.sub(r"\$\\ensuremath\{(.*?)\}\$", r"$\1$", body)


# ---------------------------------------------------------------- references
def ref(k):
    f = BIB[k]["full"]
    venue = f.get("journal") or f.get("booktitle") or f.get("howpublished") or f.get("note") or ""
    return f"- {f.get('author', '').replace(' and ', ', ')} ({f.get('year', '')}). {f.get('title', '')}. {venue}".strip()


refs = "## References\n\n" + "\n".join(ref(k) for k in sorted(CITED, key=lambda k: BIB[k]["short"]))
body = body.replace("@@REFS@@", refs)
body = re.sub(r"[ \t]+\n", "\n", body)
body = re.sub(r"\n{3,}", "\n\n", body)
head = f"# {title}\n\nJakub Michałek, independent researcher\n\n*November 2026. Exported from `paper/main.tex` by `paper/export_md.py`; the PDF is the reference version.*\n"
out = head + body
left = sorted(set(re.findall(r"\\[A-Za-z]+", re.sub(r"\$.*?\$", "", out, flags=re.S))))
open(os.path.join(HERE, "paper.md"), "w", encoding="utf8", newline="\n").write(out)
print(f"wrote paper.md ({len(out.split())} words); LaTeX commands left outside math: {left or 'none'}")
