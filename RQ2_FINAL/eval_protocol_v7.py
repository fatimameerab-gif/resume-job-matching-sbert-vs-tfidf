import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

from data_protocol_v2 import (
    N_NEGATIVES_PER_RESUME,
    MAX_POS_REUSE_PER_JOB,
    build_deduped_resumes,
    build_deduped_job_pool,
    grouped_split,
)

# Weak-label families used ONLY for same-family negative sampling.
# Membership is by resume-dataset category names, not by a new gold taxonomy.
CATEGORY_FAMILIES = {
    "software": [
        "Java Developer", "Python Developer", "DotNet Developer", "DevOps Engineer",
        "Web Designing", "SAP Developer", "Testing", "Automation Testing",
        "Network Security Engineer", "ETL Developer", "Hadoop", "Database", "Blockchain",
    ],
    "data": ["Data Science", "Business Analyst", "ETL Developer", "Hadoop", "Database"],
    "engineering": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineering"],
    "business_ops": ["Sales", "HR", "Operations Manager", "PMO", "Advocate"],
    "other": ["Arts", "Health and fitness"],
}

NEGATIVE_TIERS = ("random_other_category", "same_family", "hard_mined")


def family_of(category: str) -> str:
    hits = [fam for fam, cats in CATEGORY_FAMILIES.items() if category in cats]
    return hits[0] if hits else "ungrouped"


def _job_text_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def grouped_split_resume_and_job_text(resumes, job_pool, test_size=0.30, val_size=0.50, seed=42):
    """Split resumes by resume_id AND jobs by job text so a job posting
    cannot appear in more than one split. Categories with no jobs in a
    split will yield fewer pairs (logged, not silently filled from other splits)."""
    train_res, val_res, test_res = grouped_split(resumes, test_size=test_size, val_size=val_size, seed=seed)

    job_pool = job_pool.copy()
    job_pool["job_text_key"] = job_pool["text_clean"].map(_job_text_key)
    keys = job_pool["job_text_key"].drop_duplicates().sort_values().reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy = pd.DataFrame({"k": keys})
    tr_idx, tmp_idx = next(gss.split(dummy, groups=dummy["k"]))
    train_keys = set(dummy.iloc[tr_idx]["k"])
    tmp_keys = dummy.iloc[tmp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    va_idx, te_idx = next(gss2.split(tmp_keys, groups=tmp_keys["k"]))
    val_keys = set(tmp_keys.iloc[va_idx]["k"])
    test_keys = set(tmp_keys.iloc[te_idx]["k"])

    def _jobs(keys):
        return job_pool[job_pool["job_text_key"].isin(keys)].reset_index(drop=True)

    return (train_res, val_res, test_res,
            _jobs(train_keys), _jobs(val_keys), _jobs(test_keys))


def _sample_row(df, rng):
    idx = int(rng.randint(0, len(df)))
    return df.iloc[idx]


def build_pairs_v7(resumes_split, job_pool, split_name, rng, negative_tier="random_other_category",
                   hard_vectorizer=None, hard_job_matrix=None, pos_rng=None, neg_reuse_cap=None,
                   near_dup_threshold=0.9, length_match_ratio=None, drop_contradictory=False):
    """1 positive + N negatives per resume. RNG is passed in so the same
    seed produces the same pairs regardless of how many models are scored.

    v7b options (defaults keep the original behaviour, so frozen seed files still reproduce):
      pos_rng        separate RNG for positives -> identical positives across negative tiers
                     (paired tier comparison: ONLY the negatives differ).
      neg_reuse_cap  max times one job text may be used as a NEGATIVE within this split.
                     Without it hard-mined negatives collapse onto a handful of big Data-Science
                     jobs, which a classifier memorises (job-prior shortcut, AUC 0.87).
      length_match_ratio  if set (e.g. 2.0), negatives must have a word count within
                     [pos_len/r, pos_len*r] of the POSITIVE job (relaxed x2 until >=6 candidates).
                     Needed because TF-IDF mining prefers long documents: unmatched, hard_mined
                     negatives had median 444 words vs 42 for positives (length-only AUC 0.86).
      drop_contradictory  drop negative candidates whose text is ALSO mapped to the resume's own
                     category (e.g. ESCO 'software developer' mapped to Java+Python+DotNet).
      near_dup_threshold  hard_mined only: skip candidates whose TF-IDF cosine to the positive
                     job is >= this value (near-duplicate of the positive = likely false negative).
    """
    if negative_tier not in NEGATIVE_TIERS:
        raise ValueError(negative_tier)
    pairs = []
    job_use_count = {}
    neg_use_count = {}
    pos_rng = pos_rng if pos_rng is not None else rng
    categories = job_pool["mapped_category"].unique().tolist()
    pool = job_pool.reset_index(drop=True)
    pool["_n_words"] = pool["text_clean"].fillna("").str.split().str.len()
    pool["_key"] = pool["text_clean"].map(_job_text_key)

    for _, resume_row in resumes_split.iterrows():
        res_cat = resume_row["category"]
        same_cat_jobs = pool[pool["mapped_category"] == res_cat]
        if len(same_cat_jobs) == 0:
            continue
        same_cat_jobs = same_cat_jobs.copy()
        same_cat_jobs["use_count"] = same_cat_jobs["job_id"].map(lambda j: job_use_count.get(j, 0))
        eligible = same_cat_jobs[same_cat_jobs["use_count"] < MAX_POS_REUSE_PER_JOB]
        if len(eligible) == 0:
            eligible = same_cat_jobs
        pos_job = _sample_row(eligible, pos_rng)
        job_use_count[pos_job["job_id"]] = job_use_count.get(pos_job["job_id"], 0) + 1
        pairs.append(_pair_row(resume_row, pos_job, res_cat, pos_job["mapped_category"], 1, split_name, "positive"))

        other = pool[pool["mapped_category"] != res_cat]
        if drop_contradictory:
            own_keys = set(pool.loc[pool["mapped_category"] == res_cat, "_key"])
            other = other[~other["_key"].isin(own_keys)]
        if length_match_ratio:
            pl, r = max(int(pos_job["_n_words"]), 1), float(length_match_ratio)
            while True:
                lm = other[(other["_n_words"] >= pl / r) & (other["_n_words"] <= pl * r)]
                if (len(lm) >= 6 and lm["mapped_category"].nunique() >= 2) or r > 64:
                    break
                r *= 2
            other = lm if len(lm) else other
        if len(other) == 0:
            continue
        negs = _select_negatives(
            resume_row, other, res_cat, categories, rng, negative_tier,
            hard_vectorizer, hard_job_matrix, pool,
            neg_use_count=neg_use_count if neg_reuse_cap else None, cap=neg_reuse_cap,
            pos_job=pos_job, near_dup_threshold=near_dup_threshold,
            allowed_ids=set(other["job_id"]) if (length_match_ratio or drop_contradictory) else None,
        )
        for neg_job, reason in negs:
            neg_use_count[neg_job["job_id"]] = neg_use_count.get(neg_job["job_id"], 0) + 1
            pairs.append(_pair_row(resume_row, neg_job, res_cat, neg_job["mapped_category"], 0, split_name, reason))
    return pd.DataFrame(pairs)


def _pair_row(resume_row, job_row, res_cat, job_cat, label, split_name, neg_reason):
    return {
        "resume_id": resume_row["resume_id"],
        "resume_text": resume_row["text_clean"],
        "job_id": job_row["job_id"],
        "job_text": job_row["text_clean"],
        "job_text_key": _job_text_key(job_row["text_clean"]),
        "resume_category": res_cat,
        "job_category": job_cat,
        "label": label,
        "split": split_name,
        "negative_reason": neg_reason,
        "resume_family": family_of(res_cat),
        "job_family": family_of(job_cat),
    }


def _under_cap(df, neg_use_count, cap):
    """Rows whose job has been used as a negative fewer than `cap` times (falls back to df)."""
    if not cap or neg_use_count is None or len(df) == 0:
        return df
    ok = df[df["job_id"].map(lambda j: neg_use_count.get(j, 0)) < cap]
    return ok if len(ok) else df


def _select_negatives(resume_row, other, res_cat, categories, rng, tier,
                      hard_vectorizer, hard_job_matrix, pool,
                      neg_use_count=None, cap=None, pos_job=None, near_dup_threshold=0.9,
                      allowed_ids=None):
    chosen = []
    if tier == "random_other_category":
        # only categories that still have candidates after length/contradiction filtering
        avail = set(other["mapped_category"]) if allowed_ids is not None else None
        other_cats = [c for c in categories if c != res_cat and (avail is None or c in avail)]
        rng.shuffle(other_cats)
        for neg_cat in other_cats[:N_NEGATIVES_PER_RESUME]:
            neg_pool = _under_cap(other[other["mapped_category"] == neg_cat], neg_use_count, cap)
            if len(neg_pool) == 0:
                continue
            chosen.append((_sample_row(neg_pool, rng), "random_other_category"))
        return chosen

    if tier == "same_family":
        fam = family_of(res_cat)
        fam_pool = other[other["mapped_category"].map(family_of) == fam]
        src = fam_pool if len(fam_pool) >= N_NEGATIVES_PER_RESUME else other
        reason = "same_family" if len(fam_pool) >= N_NEGATIVES_PER_RESUME else "same_family_fallback_random"
        src = _under_cap(src, neg_use_count, cap)
        idx = rng.choice(len(src), size=min(N_NEGATIVES_PER_RESUME, len(src)), replace=False)
        for i in idx:
            chosen.append((src.iloc[int(i)], reason))
        return chosen

    # hard_mined: highest TF-IDF cosine among other-category jobs, subject to
    #   (a) not a near-duplicate of the positive job (false-negative guard),
    #   (b) per-job negative reuse cap (diversity; anti job-identity shortcut).
    # NOTE: mining with TF-IDF is adversarial to TF-IDF *by construction*; a dense-encoder
    # mined tier (Colab) is the fairer test for TF-IDF-vs-SBERT.
    if hard_vectorizer is None:
        raise ValueError("hard_mined requires a vectorizer fitted on the job pool")
    r = hard_vectorizer.transform([resume_row["text_clean"]])
    sims = cosine_similarity(r, hard_job_matrix).ravel()
    mask = (pool["mapped_category"].values != res_cat)
    if pos_job is not None:
        pos_pos = int(pool.index[pool["job_id"] == pos_job["job_id"]][0])
        pos_sims = cosine_similarity(hard_job_matrix[pos_pos], hard_job_matrix).ravel()
        mask &= (pos_sims < near_dup_threshold) & (pool["text_clean"].map(_job_text_key).values
                                                   != _job_text_key(pos_job["text_clean"]))
    if allowed_ids is not None:
        mask &= pool["job_id"].isin(allowed_ids).values
    sims = np.where(mask, sims, -1.0)
    for i in np.argsort(-sims):
        if sims[i] < 0:
            break
        row = pool.iloc[int(i)]
        if cap and neg_use_count is not None and neg_use_count.get(row["job_id"], 0) >= cap:
            continue
        chosen.append((row, "hard_mined"))
        if len(chosen) == N_NEGATIVES_PER_RESUME:
            break
    return chosen


def fit_hard_index(job_pool):
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(job_pool["text_clean"].fillna(""))
    return vec, mat


def build_frozen_seed_pairs(out_dir="data/eval_v7/frozen_seeds", n_seeds=5, seed_offset=0):
    """One (train,test) pair file per seed, shared by all models."""
    os.makedirs(out_dir, exist_ok=True)
    resumes = build_deduped_resumes()
    job_pool = build_deduped_job_pool()
    paths = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + seed_offset)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, test_idx = next(gss.split(resumes, groups=resumes["resume_id"]))
        train_res = resumes.iloc[train_idx].reset_index(drop=True)
        test_res = resumes.iloc[test_idx].reset_index(drop=True)
        train_pairs = build_pairs_v7(train_res, job_pool, "train", rng, "random_other_category")
        test_pairs = build_pairs_v7(test_res, job_pool, "test", np.random.RandomState(10_000 + seed),
                                    "random_other_category")
        all_pairs = pd.concat([train_pairs, test_pairs], ignore_index=True)
        path = os.path.join(out_dir, f"pairs_seed{seed}.csv")
        all_pairs.to_csv(path, index=False)
        paths.append(path)
    return paths


def build_tiered_pairs_on_v2_resumes(out_dir="data/eval_v7/negative_tiers"):
    """Reuse the v2 resume split (from saved pairs) but rebuild negatives
    under each tier. Positives are re-sampled with a fixed RNG (not copied
    from v2) so tiers remain internally consistent. Original v2 file is not
    modified."""
    os.makedirs(out_dir, exist_ok=True)
    v2 = pd.read_csv("data/resume_job_pairs_v3.csv")
    resumes = build_deduped_resumes()
    job_pool = build_deduped_job_pool()
    hard_vec, hard_mat = fit_hard_index(job_pool)

    split_ids = {s: set(v2.loc[v2["split"] == s, "resume_id"]) for s in ("train", "val", "test")}
    written = []
    for tier in NEGATIVE_TIERS:
        chunks = []
        for split_name, ids in split_ids.items():
            res_split = resumes[resumes["resume_id"].isin(ids)].reset_index(drop=True)
            rng = np.random.RandomState({"train": 1, "val": 2, "test": 3}[split_name])
            kwargs = {}
            if tier == "hard_mined":
                kwargs = {"hard_vectorizer": hard_vec, "hard_job_matrix": hard_mat}
            chunks.append(build_pairs_v7(res_split, job_pool, split_name, rng, tier, **kwargs))
        df = pd.concat(chunks, ignore_index=True)
        path = os.path.join(out_dir, f"pairs_{tier}.csv")
        df.to_csv(path, index=False)
        written.append(path)
    return written


def build_resume_job_grouped_pairs(out_path="data/eval_v7/pairs_resume_and_job_grouped.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resumes = build_deduped_resumes()
    job_pool = build_deduped_job_pool()
    train_res, val_res, test_res, train_jobs, val_jobs, test_jobs = grouped_split_resume_and_job_text(
        resumes, job_pool, seed=42
    )
    parts = [
        build_pairs_v7(train_res, train_jobs, "train", np.random.RandomState(11), "random_other_category"),
        build_pairs_v7(val_res, val_jobs, "val", np.random.RandomState(12), "random_other_category"),
        build_pairs_v7(test_res, test_jobs, "test", np.random.RandomState(13), "random_other_category"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_path, index=False)
    overlap = (
        set(df.loc[df.split == "train", "job_text_key"])
        & set(df.loc[df.split == "test", "job_text_key"])
    )
    if overlap:
        raise AssertionError(f"job text leaked across train/test: {len(overlap)}")
    return out_path, {
        "n_train_jobs": train_jobs["job_text_key"].nunique(),
        "n_val_jobs": val_jobs["job_text_key"].nunique(),
        "n_test_jobs": test_jobs["job_text_key"].nunique(),
        "n_pairs": len(df),
        "n_test_pairs": int((df["split"] == "test").sum()),
    }


if __name__ == "__main__":
    print("Building frozen seed pairs...")
    print(build_frozen_seed_pairs())
    print("Building negative-tier pairs...")
    print(build_tiered_pairs_on_v2_resumes())
    print("Building resume+job grouped pairs...")
    print(build_resume_job_grouped_pairs())
