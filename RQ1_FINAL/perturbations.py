import re, string
import numpy as np, pandas as pd

_clean = lambda s: re.sub(r"\s+", " ", re.sub(r"\(.*?\)", " ", str(s).lower())).strip()


def build_esco_lexicon(path="data/esco_skills.csv", min_len=3, max_words=5):
    d = pd.read_csv(path); concept_labels, owner = {}, {}
    for i, r in d.iterrows():
        labs = {_clean(r.preferredLabel)} | {_clean(x) for x in str(r.altLabels).split("\n") if x.strip() and str(r.altLabels) != "nan"}
        labs = {l for l in labs if min_len <= len(l) and len(l.split()) <= max_words and re.fullmatch(r"[a-z0-9 +#./-]+", l)}
        concept_labels[i] = sorted(labs)
        for l in labs: owner.setdefault(l, set()).add(i)
    lex = {}
    for l, o in owner.items():
        if len(o) == 1:
            c = next(iter(o)); alts = [x for x in concept_labels[c] if x != l]
            if alts: lex[l] = alts
    return lex


def esco_synonyms(text, lex, rate, rng, max_n=5, min_words=1):
    w = str(text).split(); out, i, n_match, n_sub = [], 0, 0, 0
    while i < len(w):
        hit = None
        for n in range(min(max_n, len(w) - i), 0, -1):
            key = " ".join(w[i:i + n]).strip(string.punctuation + " ").lower()
            if key in lex: hit = (n, key); break
        if hit and len(hit[1]) >= 3 and len(hit[1].split()) >= min_words:
            n_match += 1
            if rng.random() < rate:
                alts = lex[hit[1]]; out.append(alts[rng.integers(len(alts))]); n_sub += 1; i += hit[0]; continue
        out.append(w[i]); i += 1
    return " ".join(out), n_match, n_sub


def typos(text, rate, rng):
    L = string.ascii_lowercase; out = []; n = 0
    for w in str(text).split():
        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            k = rng.integers(1, len(w) - 1); op = rng.integers(4)
            if op == 0: w = w[:k] + w[k + 1] + w[k] + w[k + 2:] if k + 1 < len(w) else w
            elif op == 1: w = w[:k] + w[k + 1:]
            elif op == 2: w = w[:k] + w[k] + w[k:]
            else: w = w[:k] + L[rng.integers(26)] + w[k + 1:]
            n += 1
        out.append(w)
    return " ".join(out), n


def reorder(text, rng):
    s = [x for x in re.split(r"(?<=[.!?])\s+", str(text)) if x]
    if len(s) < 3: return str(text)
    idx = rng.permutation(len(s)); return " ".join(s[i] for i in idx)
