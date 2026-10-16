import os, re, time, hashlib
import numpy as np, pandas as pd

SEED_DIR = "data/eval_v7/frozen_seeds"
OUT_DIR = "results/paraphrases"
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LEN_RATIO = (0.6, 1.5); JACCARD = (0.15, 0.85); PROTECTED_MIN = 0.85
CHUNK_WORDS = (60, 130)
PROMPT = ("Rewrite the following {kind} excerpt in different words and with different sentence structure, keeping the meaning exactly the same. "
          "Do not add, remove or change any fact. Keep every skill, tool, technology name, number, date, job title, company and place name exactly as written. "
          "Output only the rewritten excerpt, nothing else.\n\nExcerpt:\n{chunk}")
SYSTEM = "You are a careful copy editor."
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()
NUMWORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}     # "5 years" -> "five years" keeps the fact
_tok = lambda s: [t for t in (x.strip(".") for x in re.findall(r"[a-z0-9+#.]+", str(s).lower())) if t]


# ---------------------------------------------------------------- texts and chunks
def collect_texts(seed_dir=SEED_DIR, seeds=range(5)):
    R, J = {}, {}
    for s in seeds:
        d = pd.read_csv(f"{seed_dir}/pairs_seed{s}.csv"); d = d[d.split == "test"]
        for t in d.resume_text.unique(): R[sha(t)] = t
        for t in d.job_text.unique(): J[sha(t)] = t
    return R, J


def chunk_text(text, lo=CHUNK_WORDS[0], hi=CHUNK_WORDS[1]):
    """Greedy grouping of sentences into chunks of lo..hi words; a sentence longer than hi words is cut into hi-word pieces. Joining the chunks with a space gives back the text."""
    units = []
    for s in re.split(r"(?<=[.!?])\s+", str(text).strip()):
        w = s.split()
        if not w: continue
        for i in range(0, len(w), hi): units.append(w[i:i + hi])
    chunks, cur = [], []
    for u in units:
        if cur and len(cur) + len(u) > hi: chunks.append(cur); cur = []
        cur = cur + u
        if len(cur) >= lo: chunks.append(cur); cur = []
    if cur:
        if chunks and len(cur) < lo // 2: chunks[-1] = chunks[-1] + cur
        else: chunks.append(cur)
    return [" ".join(c) for c in chunks]


# ---------------------------------------------------------------- acceptance filter
def esco_single_words(path="data/esco_skills.csv"):
    d = pd.read_csv(path); out = set()
    for l in d.preferredLabel.astype(str):
        l = re.sub(r"\(.*?\)", " ", l.lower()).strip()
        if re.fullmatch(r"[a-z0-9+#.]{3,}", l): out.add(l)
    return out


def protected_tokens(text, skills):
    out = set()
    for raw in str(text).split():
        w = raw.strip(".,;:()[]{}\"'")
        lw = w.lower()
        if not lw: continue
        if re.fullmatch(r"\d+([.,]\d+)?%?", lw) or (w.isalpha() and w.isupper() and len(w) >= 2) or any(c in w for c in "+#") or lw in skills: out.add(lw)
    return out


def check(orig, para, skills):
    """Returns (accepted, reason, length_ratio, jaccard_distance, protected_retention)."""
    para = str(para or "").strip(); ow, pw = str(orig).split(), para.split()
    if not pw: return False, "empty", 0.0, np.nan, np.nan
    ratio = len(pw) / max(1, len(ow)); A, B = set(_tok(orig)), set(_tok(para)); jd = 1 - len(A & B) / max(1, len(A | B))
    prot = protected_tokens(orig, skills); ptoks = {NUMWORDS.get(t, t) for t in _tok(para)}; pst = {t.rstrip("s") for t in ptoks}
    ret = (sum(p in ptoks or p.rstrip("s") in pst or any(p in x for x in ptoks) for p in prot) / len(prot)) if prot else 1.0
    if not LEN_RATIO[0] <= ratio <= LEN_RATIO[1]: return False, "length", ratio, jd, ret
    if jd < JACCARD[0]: return False, "too_similar", ratio, jd, ret
    if jd > JACCARD[1]: return False, "too_different", ratio, jd, ret
    if ret < PROTECTED_MIN: return False, "protected_lost", ratio, jd, ret
    return True, "ok", ratio, jd, ret


def clean_output(s):
    import html
    s = html.unescape(str(s)).strip()
    lines = s.split("\n")
    if len(lines) > 1 and re.match(r"(?i)^(here|sure|certainly|rewritten|paraphrase|excerpt)\b.*:\s*$", lines[0].strip()): s = "\n".join(lines[1:]).strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'": s = s[1:-1].strip()
    return s


# ---------------------------------------------------------------- generators
def stub_generate(chunks, kind, sampling=False):
    """Deterministic fake 'paraphrase' used ONLY to test the plumbing without a GPU (never for results)."""
    out = []
    for c in chunks:
        w = c.split(); k = 0
        for i, x in enumerate(w):
            if x.isalpha() and x.islower() and len(x) > 3:
                k += 1
                if k % 3 == 0: w[i] = x + "ish"
        out.append(" ".join(w))
    return out


class HFGenerator:
    def __init__(self, model_id=MODEL_ID, batch_size=8):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.bs = torch, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        try: self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        except TypeError: self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        print(f"[ok] loaded {model_id} | commit {getattr(self.model.config, '_commit_hash', None)} | dtype {next(self.model.parameters()).dtype}", flush=True)

    def __call__(self, chunks, kind, sampling=False):
        torch, tok = self.torch, self.tok; order = sorted(range(len(chunks)), key=lambda i: len(chunks[i])); res = [None] * len(chunks)
        for b in range(0, len(order), self.bs):
            idx = order[b:b + self.bs]
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(kind=kind, chunk=chunks[i])}], tokenize=False, add_generation_prompt=True) for i in idx]
            enc = tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            max_new = int(max(len(chunks[i].split()) for i in idx) * 2.2) + 24
            kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sampling else dict(do_sample=False)
            if sampling: torch.manual_seed(0)
            with torch.no_grad(): out = self.model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.05, pad_token_id=tok.pad_token_id, **kw)
            for j, i in enumerate(idx): res[i] = clean_output(tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return res


# ---------------------------------------------------------------- pipeline
def generate(generator=None, out_dir=OUT_DIR, seed_dir=SEED_DIR, sides=("resume", "job"), limit=None, log_every=40):
    """Resumable: chunks that already have a final status in chunks.csv are not generated again."""
    os.makedirs(out_dir, exist_ok=True); skills = esco_single_words(); R, J = collect_texts(seed_dir)
    texts = {}
    if "resume" in sides: texts.update({h: ("resume", t) for h, t in R.items()})
    if "job" in sides: texts.update({h: ("job", t) for h, t in J.items()})
    if limit: texts = dict(list(texts.items())[:limit])
    path = f"{out_dir}/chunks.csv"; rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    have = {(r["chunk_id"], r["attempt"]) for r in rows}
    all_chunks = {}
    for h, (kind, t) in texts.items():
        for i, c in enumerate(chunk_text(t)): all_chunks[f"{h}:{i}"] = (kind, h, i, c)
    print(f"{len(texts)} texts -> {len(all_chunks)} chunks ({sum(len(v[3].split()) for v in all_chunks.values())} words)", flush=True)
    gen = generator or HFGenerator()
    t0 = time.time()
    for attempt, sampling in ((1, False), (2, True)):
        accepted = {r["chunk_id"] for r in rows if r["accepted"]}
        todo = [cid for cid in all_chunks if cid not in accepted and (cid, attempt) not in have]
        print(f"attempt {attempt}: {len(todo)} chunks to generate", flush=True)
        for kind_name in ("resume", "job"):
            ids = [cid for cid in todo if all_chunks[cid][0] == kind_name]
            for b in range(0, len(ids), 64):
                part = ids[b:b + 64]; outs = gen([all_chunks[c][3] for c in part], "CV" if kind_name == "resume" else "job posting", sampling=sampling)
                for cid, p in zip(part, outs):
                    kind, h, i, c = all_chunks[cid]; ok, why, ratio, jd, ret = check(c, p, skills)
                    rows.append(dict(chunk_id=cid, text_hash=h, role=kind, idx=i, attempt=attempt, accepted=int(ok), reason=why, length_ratio=ratio, jaccard_dist=jd, protected_retention=ret, original=c, paraphrase=p))
                pd.DataFrame(rows).to_csv(path, index=False)
                if attempt == 1 and b == 0 and kind_name == "resume":          # fail fast if the generator is broken (e.g. fp16 overflow, wrong template): do not burn an hour
                    first = pd.DataFrame(rows[-len(part):])
                    if first.accepted.mean() < 0.05:
                        print(first[["reason", "original", "paraphrase"]].head(3).to_string(), flush=True)
                        raise RuntimeError(f"first {len(part)} chunks: acceptance {first.accepted.mean():.0%}; the generator output looks wrong, see the examples above")
                if (b // 64) % max(1, log_every // 64) == 0 or b + 64 >= len(ids): print(f"  {kind_name} attempt {attempt}: {b + len(part)}/{len(ids)} chunks ({time.time() - t0:.0f}s)", flush=True)
        have = {(r["chunk_id"], r["attempt"]) for r in rows}
    return build_texts(rows, texts, all_chunks, out_dir)


def build_texts(rows, texts, all_chunks, out_dir=OUT_DIR):
    best = {}
    for r in rows:
        if r["accepted"]: best.setdefault(r["chunk_id"], r["paraphrase"])
    out = []
    for h, (kind, t) in texts.items():
        parts, w_para, w_all = [], 0, 0
        for i, c in enumerate(chunk_text(t)):
            cid = f"{h}:{i}"; w = len(c.split()); w_all += w
            if cid in best: parts.append(best[cid]); w_para += w
            else: parts.append(c)
        out.append(dict(text_hash=h, role=kind, original=t, paraphrased=" ".join(parts), share_words_paraphrased=w_para / max(1, w_all), n_chunks=len(parts)))
    D = pd.DataFrame(out); D.to_csv(f"{out_dir}/paraphrased_texts.csv", index=False)
    C = pd.DataFrame(rows); fin = C.sort_values("attempt").groupby("chunk_id").apply(lambda g: int(g.accepted.max()), include_groups=False) if len(C) else pd.Series(dtype=int)
    note = (f"Paraphrases: model {MODEL_ID}, greedy decoding (attempt 1), one retry with sampling T=0.7 (attempt 2); original chunk kept if both attempts were rejected.\n"
            f"Acceptance rules: length ratio {LEN_RATIO}, Jaccard distance {JACCARD}, protected-token retention >= {PROTECTED_MIN}.\n"
            f"Chunks: {len(fin)}; accepted: {int(fin.sum())} ({fin.mean():.1%}). Texts: {len(D)}; mean share of words paraphrased: resumes {D[D.role == 'resume'].share_words_paraphrased.mean():.3f}, jobs {D[D.role == 'job'].share_words_paraphrased.mean():.3f}.\n"
            "The automatic rules do NOT prove that meaning is preserved: a human check of a random sample is required (paraphrase_human_check.py) and its result must be reported next to every paraphrase number.\n")
    open(f"{out_dir}/NOTE.txt", "w").write(note); print(note, flush=True); return D


def load_map(path=f"{OUT_DIR}/paraphrased_texts.csv"):
    D = pd.read_csv(path); return dict(zip(D.original, D.paraphrased))
