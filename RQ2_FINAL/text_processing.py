import re
import numpy as np

SECTION_HEADINGS = (
    "experience", "work experience", "education", "skills", "projects",
    "summary", "objective", "certifications", "responsibilities",
    "requirements", "about the role", "qualifications",
)


def head_only(text, n_tokens=256):
    toks = str(text).split()
    return " ".join(toks[:n_tokens])


def head_tail(text, n_tokens=256):
    toks = str(text).split()
    if len(toks) <= n_tokens:
        return " ".join(toks)
    head = n_tokens // 2
    tail = n_tokens - head
    return " ".join(toks[:head] + toks[-tail:])


def section_aware(text, n_tokens=256):
    """Keep heading-tagged spans first, then fill remaining budget from the head."""
    raw = str(text)
    lower = raw.lower()
    spans = []
    for h in SECTION_HEADINGS:
        pat = re.compile(rf"(?:^|\n|\. )\s*{re.escape(h)}\s*[:\n]", re.I)
        for m in pat.finditer(lower):
            spans.append(m.start())
    if not spans:
        return head_tail(raw, n_tokens)
    chunks = []
    starts = sorted(set(spans))
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(raw), st + 800)
        chunks.append(raw[st:end])
    merged = " ".join(chunks)
    return head_only(merged, n_tokens)


def word_chunks(text, chunk_words=180, overlap=20):
    words = str(text).split()
    if len(words) <= chunk_words:
        return [" ".join(words)]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + chunk_words]))
        start += max(1, chunk_words - overlap)
    return out


def pool_chunk_vectors(vectors, how="mean"):
    arr = np.asarray(vectors)
    if how == "mean":
        return arr.mean(axis=0)
    if how == "max":
        return arr.max(axis=0)
    raise ValueError(how)


STRATEGIES = ("head_only", "head_tail", "section_aware", "full")


def apply_strategy(text, strategy, n_tokens=256):
    if strategy == "head_only":
        return head_only(text, n_tokens)
    if strategy == "head_tail":
        return head_tail(text, n_tokens)
    if strategy == "section_aware":
        return section_aware(text, n_tokens)
    if strategy == "full":
        return str(text)
    raise ValueError(strategy)
