"""Round 11 E4: synthetic biographies (a small bioS, Physics of LM Part 3.1/3.3), seeded and reproducible.

Population (python round11_bios.py make  -> runs/round11/bios/people.npz + pools.json, sha256 in bios_meta.json):
  100,000 people drawn once with np.random.default_rng(20261109); N = 20,000 uses the first 20,000 of them.
  Full name = first x middle x last, each pool 400 synthetic names (syllable combinations, distinct across pools),
  sampled without replacement, so every full name is unique. Gender (pronoun only, not counted as knowledge) uniform.
  Attributes, uniform and independent:
    birth date  day 1-28 x 12 months x years 1900-2099          67,200 values
    birth city  200 synthetic city names
    university  300 synthetic university names
    major       100 fields of study
    employer    250 synthetic company names
  Information per person = log2(67200 * 200 * 300 * 100 * 250) = 46.52 bits.
Exposure: one bio = 5 sentences (one per attribute), each from >= 5 template variants, sentence order shuffled; the full
name is the subject of the first sentence, pronouns after. Epoch e (e = 0, 1, ...) renders every person once, in the
order np.random.default_rng([20261109, e]).permutation(N), with variants and sentence order from the same generator.
Bios are GPT-2 encode_ordinary tokens joined by EOT and packed into rows of 1025 tokens, like FineWeb.
Exposures of a run = (stable + decay tokens) / (N x mean tokens per bio incl. its EOT): the decay counts.
Canonical probe for attribute a: EOT + "<full name> " + variant 0 of a's sentence up to the value; the value is scored.

python round11_bios.py make | plan | sample
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(HERE, "runs", "round11", "bios")
SEED = 20261109
N_MAX = 100_000
EOT = 50256
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ATTRS = ["birth_date", "birth_city", "university", "major", "employer"]
SIZES = {"birth_date": 28 * 12 * 200, "birth_city": 200, "university": 300, "major": 100, "employer": 250}

# {N} full name or pronoun subject (He/She), {P} possessive determiner (Name's / His / Her), {p} his/her, {V} the value
TEMPLATES = {
    "birth_date": ["{N} was born on {V}.", "{N} came into the world on {V}.", "{P} date of birth is {V}.",
                   "{N} celebrates {p} birthday on {V}.", "{N} entered life on {V}.", "{P} birthday falls on {V}."],
    "birth_city": ["{N} was born in {V}.", "{N} spent {p} early years in {V}.", "{P} birthplace is {V}.",
                   "{N} hails from {V}.", "{N} grew up in {V}.", "{N} originates from {V}."],
    "university": ["{N} studied at {V}.", "{N} graduated from {V}.", "{N} received {p} education at {V}.",
                   "{N} attended {V}.", "{N} completed {p} degree at {V}.", "{P} alma mater is {V}."],
    "major": ["{N} majored in {V}.", "{N} focused on {V} during {p} studies.", "{P} field of study was {V}.",
              "{N} earned a degree in {V}.", "{N} specialized in {V}.", "{N} chose to study {V}."],
    "employer": ["{N} works for {V}.", "{N} is employed by {V}.", "{N} has a job at {V}.", "{P} employer is {V}.",
                 "{N} worked at {V}.", "{N} draws a salary from {V}."],
}
# the birthday template drops the year? no: every variant states the full date (all 46.5 bits are always present)

MAJORS_BASE = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science", "Economics", "History",
               "Philosophy", "Psychology", "Sociology", "Linguistics", "Literature", "Music", "Art History",
               "Architecture", "Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Geology",
               "Astronomy", "Statistics", "Political Science", "Anthropology", "Nursing", "Medicine", "Law",
               "Education", "Journalism", "Marketing", "Finance", "Accounting", "Agriculture", "Ecology",
               "Oceanography", "Neuroscience", "Biochemistry", "Genetics", "Pharmacology", "Nutrition",
               "Sports Science", "Theology", "Archaeology", "Geography", "Urban Planning", "Graphic Design",
               "Film Studies", "Theater", "Dance", "Classics", "Robotics"]


def _syllable_names(rng, n, onsets, nuclei, codas):
    out, seen = [], set()
    while len(out) < n:
        k = rng.integers(2, 4)
        s = "".join(rng.choice(onsets) + rng.choice(nuclei) for _ in range(k)) + rng.choice(codas)
        s = s.capitalize()
        if 4 <= len(s) <= 10 and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def make_pools():
    rng = np.random.default_rng(SEED + 1)
    pools = {}
    taken = set()
    specs = {"first": (["b", "d", "j", "l", "m", "n", "r", "s", "t", "v", "k"], ["a", "e", "i", "o", "ia", "ae"], ["", "n", "l", "s", "a"]),
             "middle": (["c", "g", "h", "p", "w", "z", "f", "br", "tr"], ["a", "o", "u", "ei", "y"], ["", "th", "n", "x", "e"]),
             "last": (["k", "st", "gr", "m", "d", "w", "h", "sh", "pl"], ["a", "e", "o", "u", "ou"], ["rd", "ng", "lt", "ck", "ss", "nd"]),
             "city": (["al", "b", "c", "dr", "f", "l", "m", "n", "p", "v"], ["a", "e", "i", "o", "u"], ["", "n", "r", "s"]),
             "uni": (["b", "c", "d", "h", "l", "m", "r", "s", "w"], ["a", "e", "i", "o", "ey"], ["", "n", "ll", "rd"]),
             "emp": (["k", "q", "v", "x", "z", "tr", "pr", "gl"], ["a", "e", "i", "o", "y"], ["", "x", "ix", "on", "a"])}
    for name, (o, nu, co) in specs.items():
        cand = [s for s in _syllable_names(rng, 1200, o, nu, co) if s not in taken]
        pools[name] = cand
        taken |= set(cand)
    city_sfx = ["", " City", "ton", "ville", " Falls", "burg", "field", " Harbor"]
    pools["birth_city"] = []
    for s in pools["city"]:
        c = s + city_sfx[len(pools["birth_city"]) % len(city_sfx)]
        if c not in pools["birth_city"]:
            pools["birth_city"].append(c)
        if len(pools["birth_city"]) == 200:
            break
    uni_fmt = ["University of {}", "{} University", "{} Institute of Technology", "{} College", "{} State University"]
    pools["university"] = [uni_fmt[i % 5].format(s) for i, s in enumerate(pools["uni"][:300])]
    emp_fmt = ["{} Industries", "{} Labs", "{} Systems", "{} Group", "{} Corporation"]
    pools["employer"] = [emp_fmt[i % 5].format(s) for i, s in enumerate(pools["emp"][:250])]
    pools["major"] = MAJORS_BASE + ["Applied " + m for m in MAJORS_BASE]
    for k in ("first", "middle", "last"):
        pools[k] = pools[k][:400]
    for k in ("city", "uni", "emp"):
        del pools[k]
    assert all(len(pools[k]) == 400 for k in ("first", "middle", "last"))
    assert (len(pools["birth_city"]), len(pools["university"]), len(pools["major"]), len(pools["employer"])) == (200, 300, 100, 250)
    for k in pools:
        assert len(set(pools[k])) == len(pools[k]), k
    return pools


def make():
    os.makedirs(BD, exist_ok=True)
    pools = make_pools()
    rng = np.random.default_rng(SEED)
    flat = rng.choice(400 ** 3, N_MAX, replace=False)
    names = np.stack([flat // 160000, (flat // 400) % 400, flat % 400], 1)
    attrs = np.stack([rng.integers(0, SIZES[a], N_MAX) for a in ATTRS], 1)
    gender = rng.integers(0, 2, N_MAX)
    np.savez(os.path.join(BD, "people.npz"), names=names, attrs=attrs, gender=gender)
    json.dump(pools, open(os.path.join(BD, "pools.json"), "w"), indent=0)
    sha = {f: hashlib.sha256(open(os.path.join(BD, f), "rb").read()).hexdigest() for f in ("people.npz", "pools.json")}
    meta = {"seed": SEED, "N_max": N_MAX, "sizes": SIZES, "bits_per_person": float(np.log2(np.prod([float(SIZES[a]) for a in ATTRS]))),
            "sha256": sha, "templates": TEMPLATES}
    json.dump(meta, open(os.path.join(BD, "bios_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("bits_per_person", "sha256")}, indent=1))


class World:
    def __init__(self, N):
        z = np.load(os.path.join(BD, "people.npz"))
        self.N = N
        self.names, self.attrs, self.gender = z["names"][:N], z["attrs"][:N], z["gender"][:N]
        self.pools = json.load(open(os.path.join(BD, "pools.json")))

    def full_name(self, i):
        f, m, l = self.names[i]
        return f"{self.pools['first'][f]} {self.pools['middle'][m]} {self.pools['last'][l]}"

    def value(self, i, a):
        v = int(self.attrs[i, ATTRS.index(a)])
        if a == "birth_date":
            day, mon, yr = v % 28 + 1, (v // 28) % 12, 1900 + v // (28 * 12)
            return f"{MONTHS[mon]} {day}, {yr}"
        return self.pools[a][v]

    def sentence(self, i, a, variant, first):
        he = "She" if self.gender[i] else "He"
        name = self.full_name(i)
        N = name if first else he
        P = name + "'s" if first else ("Her" if self.gender[i] else "His")
        return TEMPLATES[a][variant].format(N=N, P=P, p="her" if self.gender[i] else "his", V=self.value(i, a))

    def bio(self, i, rng):
        order = rng.permutation(5)
        var = rng.integers(0, len(TEMPLATES["major"]), 5)
        return " ".join(self.sentence(i, ATTRS[k], int(var[k]), j == 0) for j, k in enumerate(order))

    def epoch_texts(self, e):
        rng = np.random.default_rng([SEED, e])
        perm = rng.permutation(self.N)
        return [self.bio(int(i), rng) for i in perm]

    def probe(self, i, a):
        """(prompt, value) of the canonical probe: variant 0 with the full name as subject, cut before the value."""
        t = TEMPLATES[a][0]
        pre = t.split("{V}")[0].format(N=self.full_name(i), P=self.full_name(i) + "'s", p="her" if self.gender[i] else "his")
        return pre.rstrip(" "), " " + self.value(i, a)


def _n_of(source):
    assert source.startswith("bios")
    return int(source[4:])


def bios_rows(source, seq_len, log=print):
    """Yield (rows (k, seq_len + 1) uint16, bios consumed) forever: epoch 0, 1, ... of the population."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    W = World(_n_of(source))
    row = seq_len + 1
    buf = np.zeros(0, np.uint16)
    used, e = 0, 0
    while True:
        texts = W.epoch_texts(e)
        for s in range(0, len(texts), 4096):
            ids = enc.encode_ordinary_batch(texts[s:s + 4096], num_threads=8)
            used += len(ids)
            flat = np.fromiter((t for doc in ids for t in (*doc, EOT)), np.uint16)
            buf = np.concatenate([buf, flat])
            k = len(buf) // row
            if k:
                yield buf[: k * row].reshape(k, row), used
                buf = buf[k * row:]
        e += 1


def val_rows_bios(source, seq_len):
    """Monitoring rows: one extra rendering (rng seed SEED + 7) of the first min(N, 5000) people (memorisation check)."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    W = World(_n_of(source))
    rng = np.random.default_rng(SEED + 7)
    texts = [W.bio(i, rng) for i in range(min(W.N, 5000))]
    flat = np.fromiter((t for doc in enc.encode_ordinary_batch(texts, num_threads=8) for t in (*doc, EOT)), np.uint16)
    row = seq_len + 1
    return flat[: len(flat) // row * row].reshape(-1, row).astype(np.int32)


def tokens_per_bio(N=N_MAX, epochs=2):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    W = World(N)
    n = [len(x) + 1 for e in range(epochs) for x in enc.encode_ordinary_batch(W.epoch_texts(e), num_threads=8)]
    return float(np.mean(n))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    elif sys.argv[1] == "sample":
        W = World(20000)
        rng = np.random.default_rng(0)
        for i in range(3):
            print(W.bio(i, rng))
            for a in ATTRS:
                print("   probe:", W.probe(i, a))
    elif sys.argv[1] == "plan":
        tpb = tokens_per_bio()
        print(json.dumps({"tokens_per_bio_incl_eot": tpb}, indent=1))
