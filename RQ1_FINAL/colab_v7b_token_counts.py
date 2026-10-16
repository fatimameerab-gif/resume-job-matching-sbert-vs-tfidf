
import os, sys, types, glob, zipfile
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS (change these, then run) =======================
ENCODERS_TO_RUN = "all"                  # or e.g. "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2"
OUT_DIR = "results/colab_v7b"
TASKA_DIR = None                         # if TaskA is not found automatically, put its folder path here, e.g. "/content/taskA_en"
# ==================================================================================

from colab_common import ENCODERS

# Used ONLY if the model itself cannot be loaded; flagged 'fallback_unverified' in the output.
FALLBACK_LIMITS = {"SBERT-MiniLM-L6-v2": 256, "SBERT-mpnet-base-v2": 384, "e5-base-v2": 512,
                   "bge-base-en-v1.5": 512, "gte-base": 512, "JobBERT-v2": 512}


def _read_dir(path):
    out = []
    for fname in sorted(os.listdir(path)):
        fp = os.path.join(path, fname)
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                out.append(f.read())
    return out


def _taskA_pair(d):
    """Returns (queries_dir, corpus_dir) if folder d holds both (names matched case-insensitively:
    'queries*' and 'corpus*', e.g. Queries/, corpus/, corpus_elements/), else None. qrels is not needed here."""
    try:
        subs = [n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))]
    except OSError:
        return None
    q = next((n for n in subs if n.lower().startswith("quer")), None)
    c = next((n for n in subs if n.lower().startswith("corpus")), None)
    return (os.path.join(d, q), os.path.join(d, c)) if q and c else None


def _search(roots):
    for r in roots:
        if r and os.path.isdir(r):
            if _taskA_pair(r):
                return _taskA_pair(r)
            for root, dirs, files in os.walk(r):
                if _taskA_pair(root):
                    return _taskA_pair(root)
    return None


def _zip_candidates():
    zs = set()
    for pat in ("*.zip", "data/*.zip", "/content/*.zip", "**/*.zip"):
        for z in glob.glob(pat, recursive=True):
            if "taska" in os.path.basename(z).lower().replace("_", "").replace("-", ""):
                zs.add(z)
    return sorted(zs)


def find_taskA(taska_dir=None):
    """Returns (queries_dir, corpus_dir) or None. Order: explicit TASKA_DIR -> data/taskA_en, taskA_en, anywhere below '.'
    -> any *taskA*.zip (extracted to data/taskA_en_unzipped)."""
    found = _search([taska_dir, "data/taskA_en", "taskA_en", "."])
    if found:
        return found
    for z in _zip_candidates():
        print(f"TaskA looks like a zip ({z}) -> extracting to data/taskA_en_unzipped ...", flush=True)
        with zipfile.ZipFile(z) as f:
            f.extractall("data/taskA_en_unzipped")
        found = _search(["data/taskA_en_unzipped"])
        if found:
            return found
    return None


def _diagnose(taska_dir):
    print("\n" + "!" * 78 + "\n!!! TaskA NOT FOUND -> taskA_* rows will be MISSING from the output.", flush=True)
    print("!!! Working folder:", os.getcwd())
    for d in (".", "data", "/content"):
        if os.path.isdir(d):
            print(f"!!! contents of {d!r}:", sorted(os.listdir(d))[:40])
    zs = _zip_candidates()
    print("!!! zip files whose name contains 'taskA':", zs or "NONE")
    for z in zs[:2]:
        try:
            print(f"!!! first entries inside {z}:", zipfile.ZipFile(z).namelist()[:15])
        except Exception as e:
            print(f"!!! could not open {z}: {e}")
    if taska_dir:
        print("!!! TASKA_DIR was set to", repr(taska_dir), "-> exists:", os.path.isdir(taska_dir))
    print("!!! Needed: a folder holding one sub-folder named like 'queries' and one named like 'corpus' / 'corpus_elements'.")
    print("!!! Fix: unzip it, then set TASKA_DIR = '<that folder>' in SETTINGS and run again.\n" + "!" * 78 + "\n", flush=True)


def load_texts(taska_dir=None):
    pairs = pd.read_csv("data/resume_job_pairs_v3.csv")
    src = {"v3_resume": pairs.resume_text.drop_duplicates().astype(str).tolist(),
           "v3_job": pairs.job_text.drop_duplicates().astype(str).tolist()}
    found = find_taskA(taska_dir)
    if found is None:
        _diagnose(taska_dir)
    else:
        qd, cd = found
        src["taskA_query"], src["taskA_resume"] = _read_dir(qd), _read_dir(cd)
        print(f"TaskA found: queries={qd} ({len(src['taskA_query'])} files) | corpus={cd} ({len(src['taskA_resume'])} files)", flush=True)
    return src


def tokenizer_and_limit(key):
    if key == "stub":
        return (lambda texts: [len(t.split()) for t in texts]), 256, "stub_whitespace"
    mid = ENCODERS[key][0]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(mid); tok = m.tokenizer
        return (lambda texts: [len(x) for x in tok(texts, add_special_tokens=True, truncation=False,
                                                    verbose=False)["input_ids"]]), int(m.max_seq_length), "model.max_seq_length"
    except Exception as e:
        print(f"[warn] {key}: model load failed ({type(e).__name__}); trying tokenizer only", flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mid)
        return (lambda texts: [len(x) for x in tok(texts, add_special_tokens=True, truncation=False,
                                                    verbose=False)["input_ids"]]), FALLBACK_LIMITS[key], "fallback_unverified"


def main(encoders=ENCODERS_TO_RUN, out=OUT_DIR, taska_dir=TASKA_DIR):
    a = types.SimpleNamespace(encoders=encoders, out=out, taska_dir=taska_dir)
    os.makedirs(a.out, exist_ok=True)
    keys = ["stub"] if a.encoders == "stub" else (list(ENCODERS) if a.encoders == "all" else [k.strip() for k in a.encoders.split(",")])
    texts = load_texts(a.taska_dir); summ, per = [], []
    for key in keys:
        try:
            count, limit, limit_src = tokenizer_and_limit(key)
            count(["probe text"])           # fails for encoders without a word-piece tokenizer (e.g. GloVe averaging model)
        except Exception as e:
            print(f"[SKIP] {key}: {type(e).__name__}: {e}", flush=True); continue
        for source, tl in texts.items():
            n_tok = np.array(count(tl)); n_ws = np.array([len(t.split()) for t in tl])
            summ.append(dict(encoder=key, source=source, n_texts=len(tl), max_seq_length=limit, limit_source=limit_src,
                             median_tokens=float(np.median(n_tok)), p90_tokens=float(np.percentile(n_tok, 90)),
                             max_tokens=int(n_tok.max()),
                             pct_over_limit_wordpiece=round(100 * float((n_tok > limit).mean()), 1),
                             pct_over_limit_whitespace_estimate=round(100 * float((n_ws > limit).mean()), 1),
                             mean_wordpieces_per_whitespace_word=round(float(n_tok.sum() / max(n_ws.sum(), 1)), 2)))
            per.append(pd.DataFrame(dict(encoder=key, source=source, n_wordpieces=n_tok, n_whitespace=n_ws)))
        print(f"{key}: limit {limit} ({limit_src})", flush=True)
    s = pd.DataFrame(summ); s.to_csv(f"{a.out}/token_counts_summary.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(f"{a.out}/token_counts_per_text.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(s[["encoder", "source", "n_texts", "max_seq_length", "pct_over_limit_wordpiece",
             "pct_over_limit_whitespace_estimate", "mean_wordpieces_per_whitespace_word"]].to_string(index=False))
    print("\nsources in output:", sorted(s.source.unique()))
    if not any(x.startswith("taskA") for x in s.source.unique()):
        print("WARNING: no TaskA rows in this output (see the message at the top).")


if __name__ == "__main__":
    main()
