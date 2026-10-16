import os, glob, zipfile
import numpy as np
import pandas as pd
from scipy import sparse

# key: (hf model id, prefix for JOB text, prefix for RESUME text)
ENCODERS = {
    "SBERT-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "SBERT-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "e5-base-v2": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", "", ""),
    "gte-base": ("thenlper/gte-base", "", ""),
    "JobBERT-v2": ("TechWolf/JobBERT-v2", "", ""),
    # static averaged word vectors (historical baseline named in the review, section 4.5); no chunking needed
    "GloVe-avg-300d": ("sentence-transformers/average_word_embeddings_glove.6B.300d", "", ""),
}
STUB_CHUNK_WORDS = 100          # chunk size used only by the stub encoder


class Encoder:
    def __init__(self, key, model_id=None, job_prefix="", resume_prefix="", chunk_long=False, model=None):
        self.key, self.model_id = key, model_id
        self.name = key                      # label used in result tables (scripts may overwrite it, e.g. 'FT[...]')
        self.prefix = {"job": job_prefix, "resume": resume_prefix}
        self.chunk_long = chunk_long
        self.cache = {}
        self.n_texts = self.n_chunked = self.n_pieces = 0
        self.model = model
        if self.model is None and key != "stub":
            from sentence_transformers import SentenceTransformer   # lazy: not needed for stub
            self.model = SentenceTransformer(model_id)

    # ---- splitting -----------------------------------------------------------------------------------------
    def _split(self, text, pref):
        """-> list of (piece_text_with_prefix, weight)."""
        if not self.chunk_long or self.key.startswith("GloVe"):
            return [(pref + text, 1.0)]
        if self.key == "stub":
            w = text.split()
            if len(w) <= STUB_CHUNK_WORDS:
                return [(pref + text, 1.0)]
            return [(pref + " ".join(w[i:i + STUB_CHUNK_WORDS]), float(len(w[i:i + STUB_CHUNK_WORDS])))
                    for i in range(0, len(w), STUB_CHUNK_WORDS)]
        tok, limit = self.model.tokenizer, int(self.model.max_seq_length)
        n_pref = len(tok(pref, add_special_tokens=False)["input_ids"]) if pref else 0
        budget = max(limit - 2 - n_pref - 4, 16)                  # 2 special tokens + prefix + small safety margin
        ids = tok(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        if len(ids) <= budget:
            return [(pref + text, 1.0)]
        return [(pref + tok.decode(ids[i:i + budget], skip_special_tokens=True), float(len(ids[i:i + budget])))
                for i in range(0, len(ids), budget)]

    # ---- embedding -----------------------------------------------------------------------------------------
    def _embed(self, pieces):
        if self.key == "stub":
            from sklearn.feature_extraction.text import HashingVectorizer
            return HashingVectorizer(n_features=512, alternate_sign=False, norm="l2").transform(pieces).toarray()
        return self.model.encode(pieces, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)

    def _raw(self, texts, kind):
        pref = self.prefix[kind]
        pieces, owner, weight = [], [], []
        for i, t in enumerate(texts):
            parts = self._split(t, pref)
            self.n_texts += 1; self.n_pieces += len(parts); self.n_chunked += int(len(parts) > 1)
            for p, w in parts:
                pieces.append(p); owner.append(i); weight.append(w)
        E = self._embed(pieces)
        out = np.zeros((len(texts), E.shape[1]))
        np.add.at(out, np.array(owner), E * np.array(weight)[:, None])
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def encode(self, texts, kind):
        texts = [str(t) for t in texts]
        missing = [t for t in dict.fromkeys(texts) if (kind, t) not in self.cache]
        if missing:
            for t, v in zip(missing, self._raw(missing, kind)):
                self.cache[(kind, t)] = v
        return np.vstack([self.cache[(kind, t)] for t in texts]) if texts else np.zeros((0, 1))

    def report(self):
        if not self.chunk_long:
            return f"{self.key}: chunking OFF (texts longer than the model limit are truncated)"
        m = self.n_pieces / max(self.n_texts, 1)
        return (f"{self.key}: chunking ON -- {self.n_chunked}/{self.n_texts} distinct texts were split "
                f"({m:.2f} pieces per text on average)")


class DenseIndex:
    """Drop-in for the TF-IDF `hard_vectorizer` used by eval_protocol_v7 (dense-mined negatives)."""
    def __init__(self, enc):
        self.enc = enc

    def transform(self, texts):
        return sparse.csr_matrix(self.enc.encode(list(texts), "resume"))


def load_encoders(spec, chunk_long=False):
    if spec == "stub":
        return [Encoder("stub", chunk_long=chunk_long)]
    keys = list(ENCODERS) if spec == "all" else [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ENCODERS]
    if unknown:
        raise SystemExit(f"Unknown encoder(s) {unknown}. Choose from {list(ENCODERS)}, 'all' or 'stub'.")
    out = []
    for k in keys:
        mid, jp, rp = ENCODERS[k]
        try:
            out.append(Encoder(k, mid, jp, rp, chunk_long=chunk_long))
            print(f"[ok] loaded {k} ({mid}) | chunk_long={chunk_long}", flush=True)
        except Exception as e:                       # gated / renamed / offline models must not kill the run
            print(f"[SKIP] could not load {k} ({mid}): {type(e).__name__}: {e}", flush=True)
    if not out:
        raise SystemExit("No encoder could be loaded.")
    return out


# =====================================================================================================================
# Added for the RQ1/RQ2 Colab scripts 4-6
# =====================================================================================================================
class TfidfBackend:
    """TF-IDF with the same encode(texts, kind) interface as Encoder, so a script can treat it like a dense model.
    Fit on the TRAIN texts only. Rows are L2-normalised by TfidfVectorizer (so cosine = dot product)."""
    key = name = "TF-IDF"

    def __init__(self, fit_texts, max_features=3000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english").fit(
            [str(t) for t in fit_texts])
        self.cache = {}

    def encode(self, texts, kind=None):
        texts = [str(t) for t in texts]
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, r in zip(missing, self.v.transform(missing).toarray()):
                self.cache[t] = r
        return np.vstack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1))

    def report(self):
        return "TF-IDF (fit on train texts)"


def _read_dir_dict(path):
    out = {}
    for fname in sorted(os.listdir(path)):
        fp = os.path.join(path, fname)
        if os.path.isfile(fp):
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                out[fname] = f.read()
    return out


def _taskA_dirs(d):
    """(queries_dir, corpus_dir, qrels_file) if folder d has sub-folders named like 'queries*' and 'corpus*'
    (case-insensitive; corpus or corpus_elements) and a file named like 'qrels*', else None."""
    try:
        names = os.listdir(d)
    except OSError:
        return None
    subs = [n for n in names if os.path.isdir(os.path.join(d, n))]
    q = next((n for n in subs if n.lower().startswith("quer")), None)
    c = next((n for n in subs if n.lower().startswith("corpus")), None)
    qr = next((n for n in names if n.lower().startswith("qrels") and os.path.isfile(os.path.join(d, n))), None)
    return (os.path.join(d, q), os.path.join(d, c), os.path.join(d, qr)) if (q and c and qr) else None


def find_taskA(taska_dir=None):
    """Finds the TalentCLEF TaskA dev folder (queries/, corpus/ or corpus_elements/, qrels.tsv); if it only exists as a
    *taskA*.zip, extracts it to data/taskA_en_unzipped. Returns (queries_dir, corpus_dir, qrels_file) or None."""
    def search(roots):
        for r in roots:
            if r and os.path.isdir(r):
                for root, _, _ in os.walk(r):
                    if _taskA_dirs(root):
                        return _taskA_dirs(root)
        return None
    found = search([taska_dir, "data/taskA_en", "taskA_en", "."])
    if found:
        return found
    zs = {z for z in glob.glob("**/*.zip", recursive=True) + glob.glob("/content/*.zip")
          if "taska" in os.path.basename(z).lower().replace("_", "").replace("-", "")}
    for z in sorted(zs):
        print(f"TaskA looks like a zip ({z}) -> extracting to data/taskA_en_unzipped ...", flush=True)
        zipfile.ZipFile(z).extractall("data/taskA_en_unzipped")
        found = search(["data/taskA_en_unzipped"])
        if found:
            return found
    return None


def load_taskA(taska_dir=None):
    """Same output as taskA_loader.load_taskA_pairs (full 10 x 472 grid; label = in qrels) but tolerant to folder names / zip."""
    found = find_taskA(taska_dir)
    if found is None:
        raise FileNotFoundError(
            "TaskA not found. Needed: a folder with sub-folders 'queries*' and 'corpus*' plus a file 'qrels*'. "
            f"Working folder: {os.getcwd()}; contents: {sorted(os.listdir('.'))[:40]}. "
            "Unzip it and set TASKA_DIR = '<that folder>' in SETTINGS.")
    qd, cd, qr = found
    queries, corpus = _read_dir_dict(qd), _read_dir_dict(cd)
    qrels = pd.read_csv(qr, sep="\t", header=None, names=["query_id", "iteration", "doc_id", "relevance"], dtype=str)
    pos = set(zip(qrels["query_id"], qrels["doc_id"]))
    rows = [{"query_id": q, "resume_id": d, "job_text": jt, "resume_text": rt, "label": int((q, d) in pos)}
            for q, jt in queries.items() for d, rt in corpus.items()]
    pairs = pd.DataFrame(rows)
    print(f"TaskA found: {qd} | {cd} | {qr}\n  {len(queries)} queries x {len(corpus)} resumes = {len(pairs)} pairs, "
          f"{int(pairs.label.sum())} positive", flush=True)
    return pairs


def find_seed_dir(explicit=None):
    """Folder holding pairs_seed0.csv: SEED_DIR, data/eval_v7/frozen_seeds, anywhere below '.', or inside ANY zip."""
    for r in (explicit, "data/eval_v7/frozen_seeds", "frozen_seeds", "."):
        if r and os.path.isdir(r):
            for root, dirs, files in os.walk(r):
                dirs[:] = [d for d in dirs if not d.startswith("_archive")]       # never pick up archived (old/unlicensed) seeds
                if "pairs_seed0.csv" in files:
                    return root
    for z in sorted(set(glob.glob("**/*.zip", recursive=True) + glob.glob("/content/*.zip"))):
        try:
            names = zipfile.ZipFile(z).namelist()
        except Exception:
            continue
        if any(n.endswith("pairs_seed0.csv") for n in names):
            print(f"seed files are inside a zip ({z}) -> extracting to data/eval_v7_unzipped ...", flush=True)
            zipfile.ZipFile(z).extractall("data/eval_v7_unzipped")
            for root, _, files in os.walk("data/eval_v7_unzipped"):
                if "pairs_seed0.csv" in files:
                    return root
    raise FileNotFoundError(
        "pairs_seed0.csv ... pairs_seed4.csv NOT FOUND. Unzip them and set SEED_DIR = '<folder>' in SETTINGS. "
        f"Working folder: {os.getcwd()}; contents: {sorted(os.listdir('.'))[:40]}")


def load_seed_pairs(seed_dir, seed):
    df = pd.read_csv(os.path.join(seed_dir, f"pairs_seed{seed}.csv"))
    return df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
