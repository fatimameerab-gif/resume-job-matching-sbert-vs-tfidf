import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, issparse


# ---------------------------------------------------------------------------
# Feature-construction ablation (review S4.5, S3.3)
# ---------------------------------------------------------------------------
FEATURE_MODES = ["cos_only", "absdiff_cos", "concat_absdiff_hadamard"]


def _to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def build_pair_features(r, j, mode="absdiff_cos"):
    """r, j: (n_samples, dim) dense or sparse vectors, row-aligned.
    Returns (X, cos) for the requested feature-construction mode.
      cos_only                 : [cos(r,j)]                      (1 dim)
      absdiff_cos (ORIGINAL)   : [|r-j|, cos(r,j)]                (d+1 dims)
      concat_absdiff_hadamard  : [r, j, |r-j|, r*j]  (standard SBERT/
                                  InferSent head)                 (4d dims)
    """
    r_d, j_d = _to_dense(r), _to_dense(j)
    # Vectorized row-wise cosine similarity (was a Python loop calling
    # sklearn's cosine_similarity() once per row -- ~4,000x per-call
    # overhead, the dominant cost on the TaskA pipeline).
    r_norm = np.linalg.norm(r_d, axis=1)
    j_norm = np.linalg.norm(j_d, axis=1)
    denom = r_norm * j_norm
    denom[denom == 0] = 1e-10
    cos = (np.sum(r_d * j_d, axis=1) / denom).reshape(-1, 1)

    if mode == "cos_only":
        return cos, cos
    elif mode == "absdiff_cos":
        absdiff = np.abs(r_d - j_d)
        return np.hstack([absdiff, cos]), cos
    elif mode == "concat_absdiff_hadamard":
        absdiff = np.abs(r_d - j_d)
        hadamard = r_d * j_d
        return np.hstack([r_d, j_d, absdiff, hadamard]), cos
    else:
        raise ValueError(f"Unknown feature mode: {mode}")


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def tfidf_features(train_resume, train_job, other_resume=None, other_job=None,
                    max_features=3000, ngram_range=(1, 2), feature_mode="absdiff_cos"):
    """Returns (X_train, X_other, cos_train, cos_other) using TF-IDF.
    feature_mode selects among FEATURE_MODES (review S4.5 ablation)."""
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    combined = pd.concat([train_resume, train_job], ignore_index=True)
    vectorizer.fit(combined)
    fit_time = time.perf_counter() - t0

    def _build(res, job):
        t1 = time.perf_counter()
        r = _to_dense(vectorizer.transform(res))
        j = _to_dense(vectorizer.transform(job))
        encode_time = time.perf_counter() - t1
        X, cos = build_pair_features(r, j, mode=feature_mode)
        return X, cos, encode_time, r.nbytes + j.nbytes

    X_train, cos_train, enc_t, nbytes = _build(train_resume, train_job)
    result = {"fit_time_s": fit_time, "encode_time_s": enc_t, "index_bytes": nbytes,
               "dim": vectorizer.transform(train_resume[:1]).shape[1]}
    if other_resume is not None:
        X_other, cos_other, enc_t2, _ = _build(other_resume, other_job)
        result["other_encode_time_s"] = enc_t2
        return X_train, X_other, cos_train, cos_other, result
    return X_train, cos_train, result


# ---------------------------------------------------------------------------
# LSA (TruncatedSVD on TF-IDF)
# ---------------------------------------------------------------------------
def lsa_features(train_resume, train_job, other_resume=None, other_job=None,
                  max_features=3000, n_components=100, seed=42, feature_mode="absdiff_cos"):
    """Returns dense LSA (TruncatedSVD-on-TF-IDF) features."""
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    combined = pd.concat([train_resume, train_job], ignore_index=True)
    tfidf_combined = vectorizer.fit_transform(combined)
    svd.fit(tfidf_combined)
    fit_time = time.perf_counter() - t0

    def _build(res, job):
        t1 = time.perf_counter()
        r = svd.transform(vectorizer.transform(res))
        j = svd.transform(vectorizer.transform(job))
        encode_time = time.perf_counter() - t1
        X, cos = build_pair_features(r, j, mode=feature_mode)
        return X, cos, encode_time, r.nbytes + j.nbytes

    X_train, cos_train, enc_t, nbytes = _build(train_resume, train_job)
    result = {"fit_time_s": fit_time, "encode_time_s": enc_t, "index_bytes": nbytes, "dim": n_components}
    if other_resume is not None:
        X_other, cos_other, enc_t2, _ = _build(other_resume, other_job)
        result["other_encode_time_s"] = enc_t2
        return X_train, X_other, cos_train, cos_other, result
    return X_train, cos_train, result


# ---------------------------------------------------------------------------
# BM25 (review S4.5: "BM25... needed" as a lexical baseline stronger/
# different from TF-IDF+cosine). BM25 has no fixed-length vector per text --
# it scores a (query, document) pair directly against a corpus index, so it
# is exposed as a single score column, not run through FEATURE_MODES.
# ---------------------------------------------------------------------------
def _tokenize(text):
    return text.split()


def bm25_features(train_resume, train_job, other_resume=None, other_job=None):
    """BM25Okapi indexed over the TRAIN job texts only (avoids test-set
    leakage into the IDF statistics). Score = BM25(resume_tokens, matching
    job_text's index position) for the SAME-ROW job text (this dataset is
    pairwise, not corpus retrieval, so we score row-aligned pairs directly
    rather than a full ranking over all jobs)."""
    t0 = time.perf_counter()
    train_job_list = train_job.tolist()
    corpus_tokens = [_tokenize(t) for t in train_job_list]
    from rank_bm25 import BM25Okapi          # lazy import: only needed for the BM25 baseline (pip install rank_bm25)
    bm25 = BM25Okapi(corpus_tokens)
    fit_time = time.perf_counter() - t0

    # Build a lookup so we can score a resume against ANY job text (train or
    # test) using the train-fitted BM25 statistics, by scoring against a
    # length-1 "document" via get_scores on a throwaway single-doc index is
    # not exact BM25 semantics for unseen docs, so instead: score resume
    # query against the train corpus to get IDF-consistent term weights and
    # take a manual per-pair BM25 score using the fitted idf.
    def _pair_score(resume_text, job_text):
        q_tokens = _tokenize(resume_text)
        # temporarily score against a corpus that includes this exact job
        # text at position 0, reusing the already-fitted idf/avgdl from train
        doc_tokens = _tokenize(job_text)
        score = 0.0
        doc_len = len(doc_tokens)
        freqs = {}
        for w in doc_tokens:
            freqs[w] = freqs.get(w, 0) + 1
        k1, b = bm25.k1, bm25.b
        for w in q_tokens:
            if w not in bm25.idf:
                continue
            f = freqs.get(w, 0)
            if f == 0:
                continue
            idf = bm25.idf[w]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / bm25.avgdl))
        return score

    def _build(res, job):
        t1 = time.perf_counter()
        scores = np.array([_pair_score(r, j) for r, j in zip(res, job)]).reshape(-1, 1)
        encode_time = time.perf_counter() - t1
        return scores, scores, encode_time

    X_train, cos_train, enc_t = _build(train_resume, train_job)
    result = {"fit_time_s": fit_time, "encode_time_s": enc_t,
              "index_bytes": len(bm25.idf) * 16, "dim": 1}
    if other_resume is not None:
        X_other, cos_other, enc_t2 = _build(other_resume, other_job)
        result["other_encode_time_s"] = enc_t2
        return X_train, X_other, cos_train, cos_other, result
    return X_train, cos_train, result


# ---------------------------------------------------------------------------
# Locally-trained static embedding (word2vec CBOW, averaged over tokens).
# This is NOT pretrained GloVe/fastText (no internet in this sandbox) -- it
# is trained from scratch on the project's own resume+job corpus, so treat
# it as a lower bound / sanity check for "what a static embedding can do
# here", not as the paper-final static-embedding number. Swap in pretrained
# vectors (gensim-data or fastText.cc, reachable on Colab) for that.
# ---------------------------------------------------------------------------
def _train_static_embeddings(all_texts, dim=100, seed=42):
    from gensim.models import Word2Vec
    sentences = [t.split() for t in all_texts]
    model = Word2Vec(sentences, vector_size=dim, window=5, min_count=1,
                      workers=1, seed=seed, sg=0, epochs=10)
    return model


def _average_embed(texts, model, dim):
    vecs = []
    for t in texts:
        toks = [w for w in t.split() if w in model.wv]
        if toks:
            vecs.append(np.mean(model.wv[toks], axis=0))
        else:
            vecs.append(np.zeros(dim))
    return np.array(vecs)


def static_embedding_features(train_resume, train_job, other_resume=None, other_job=None,
                               dim=100, seed=42, feature_mode="absdiff_cos"):
    t0 = time.perf_counter()
    combined = pd.concat([train_resume, train_job], ignore_index=True)
    model = _train_static_embeddings(combined, dim=dim, seed=seed)
    fit_time = time.perf_counter() - t0

    def _build(res, job):
        t1 = time.perf_counter()
        r = _average_embed(res, model, dim)
        j = _average_embed(job, model, dim)
        encode_time = time.perf_counter() - t1
        X, cos = build_pair_features(r, j, mode=feature_mode)
        return X, cos, encode_time, r.nbytes + j.nbytes

    X_train, cos_train, enc_t, nbytes = _build(train_resume, train_job)
    result = {"fit_time_s": fit_time, "encode_time_s": enc_t, "index_bytes": nbytes, "dim": dim}
    if other_resume is not None:
        X_other, cos_other, enc_t2, _ = _build(other_resume, other_job)
        result["other_encode_time_s"] = enc_t2
        return X_train, X_other, cos_train, cos_other, result
    return X_train, cos_train, result


REPRESENTATIONS = {
    "TF-IDF": tfidf_features,
    "LSA (TruncatedSVD-100d)": lsa_features,
    "BM25": bm25_features,
    "Static-word2vec-local-100d": static_embedding_features,
}
