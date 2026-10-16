import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

from data_loading import (
    load_resumes, load_job_postings, load_esco_occupations,
    map_title_to_category, map_esco_title_to_category, clean_text
)

random.seed(42)
np.random.seed(42)
N_NEGATIVES_PER_RESUME = 2
MAX_POS_REUSE_PER_JOB = 8  # cap how many times one job text can be reused as a positive


def build_deduped_resumes():
    resumes = load_resumes()
    before = len(resumes)
    resumes = resumes.drop_duplicates(subset="text_clean").reset_index(drop=True)
    resumes["resume_id"] = resumes.index
    print(f"Resumes: {before} rows -> {len(resumes)} unique after dedup")
    return resumes


def build_deduped_job_pool():
    jobs = load_job_postings()
    esco = load_esco_occupations()

    jobs["mapped_category"] = jobs["seek_category"].where(jobs["seek_category"].notna(), jobs["title"].apply(map_title_to_category))
    jobs = jobs.dropna(subset=["mapped_category"])
    jobs_expanded = jobs[["title", "text_clean", "mapped_category", "job_source"]]

    esco["mapped_categories"] = esco["title"].apply(map_esco_title_to_category)
    esco = esco[esco["mapped_categories"].map(len) > 0]
    esco_expanded = esco.explode("mapped_categories").rename(
        columns={"mapped_categories": "mapped_category"}
    )[["title", "text_clean", "mapped_category"]]
    esco_expanded["job_source"] = "esco"

    pool = pd.concat([jobs_expanded, esco_expanded], ignore_index=True)
    before = len(pool)
    pool = pool.drop_duplicates(subset=["text_clean", "mapped_category"]).reset_index(drop=True)
    pool["job_id"] = pool.index
    print(f"Job pool: {before} rows -> {len(pool)} unique after dedup")

    print("\nUnique job texts per category (sparsity check):")
    print(pool.groupby("mapped_category")["text_clean"].nunique().sort_values())

    return pool


def grouped_split(resumes: pd.DataFrame, test_size=0.30, val_size=0.50, seed=42):
    """Splits by resume_id groups so duplicate resumes never span splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(resumes, groups=resumes["resume_id"]))
    train_res = resumes.iloc[train_idx].reset_index(drop=True)
    temp_res = resumes.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_res, groups=temp_res["resume_id"]))
    val_res = temp_res.iloc[val_idx].reset_index(drop=True)
    test_res = temp_res.iloc[test_idx].reset_index(drop=True)

    return train_res, val_res, test_res


def build_pairs_for_split(resumes_split: pd.DataFrame, job_pool: pd.DataFrame, split_name: str):
    """Builds pairs with: 1 positive + N stratified negatives per resume.
    Negatives are drawn from a RANDOMISED SET of other categories (one
    job per category sampled), not from the raw pooled rows -- this is
    what fixes the Data-Science-dominated negative sampling."""
    pairs = []
    job_use_count = {}  # job_id -> times used as positive, to cap reuse

    categories = job_pool["mapped_category"].unique().tolist()

    for _, resume_row in resumes_split.iterrows():
        res_cat = resume_row["category"]

        same_cat_jobs = job_pool[job_pool["mapped_category"] == res_cat]
        if len(same_cat_jobs) == 0:
            continue

        # prefer less-reused job texts for the positive pair
        same_cat_jobs = same_cat_jobs.copy()
        same_cat_jobs["use_count"] = same_cat_jobs["job_id"].map(lambda j: job_use_count.get(j, 0))
        eligible = same_cat_jobs[same_cat_jobs["use_count"] < MAX_POS_REUSE_PER_JOB]
        if len(eligible) == 0:
            eligible = same_cat_jobs  # fall back if all capped (sparse categories)
        pos_job = eligible.sample(1, random_state=random.randint(0, 99999)).iloc[0]
        job_use_count[pos_job["job_id"]] = job_use_count.get(pos_job["job_id"], 0) + 1

        pairs.append({
            "resume_id": resume_row["resume_id"],
            "resume_text": resume_row["text_clean"],
            "job_id": pos_job["job_id"],
            "job_text": pos_job["text_clean"],
            "resume_category": res_cat,
            "job_category": pos_job["mapped_category"],
            "job_source": pos_job["job_source"],
            "label": 1,
            "split": split_name,
        })

        # STRATIFIED negatives: sample from N_NEGATIVES_PER_RESUME distinct
        # OTHER categories (not from raw pooled rows), one job each
        other_cats = [c for c in categories if c != res_cat]
        chosen_cats = random.sample(other_cats, min(N_NEGATIVES_PER_RESUME, len(other_cats)))
        for neg_cat in chosen_cats:
            neg_pool = job_pool[job_pool["mapped_category"] == neg_cat]
            neg_job = neg_pool.sample(1, random_state=random.randint(0, 99999)).iloc[0]
            pairs.append({
                "resume_id": resume_row["resume_id"],
                "resume_text": resume_row["text_clean"],
                "job_id": neg_job["job_id"],
                "job_text": neg_job["text_clean"],
                "resume_category": res_cat,
                "job_category": neg_cat,
                "job_source": neg_job["job_source"],
                "label": 0,
                "split": split_name,
            })

    return pd.DataFrame(pairs)


def build_features(resume_texts, job_texts, vectorizer, fit=False):
    combined = pd.concat([resume_texts, job_texts], ignore_index=True)
    if fit:
        vectorizer.fit(combined)
    r = vectorizer.transform(resume_texts)
    j = vectorizer.transform(job_texts)
    cos = np.array([cosine_similarity(r[i], j[i])[0, 0] for i in range(r.shape[0])]).reshape(-1, 1)
    return hstack([np.abs(r - j), cos]), cos


def run_diagnostics(pairs: pd.DataFrame):
    """Trains and reports the 3 diagnostic baselines on the fixed,
    grouped-split pairs dataset."""
    train = pairs[pairs["split"] == "train"]
    val = pairs[pairs["split"] == "val"]
    test = pairs[pairs["split"] == "test"]

    print(f"\nPairs -> train: {len(train)}, val: {len(pairs[pairs['split']=='val'])}, test: {len(test)}")
    print(f"Test label balance: {test['label'].value_counts().to_dict()}")

    # 1. Majority-class baseline
    majority_label = train["label"].mode()[0]
    majority_acc = (test["label"] == majority_label).mean()

    # 2. Job-only baseline (classifier sees ONLY job text, resume text replaced with empty)
    vec_job_only = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    vec_job_only.fit(train["job_text"])
    Xtr_job = vec_job_only.transform(train["job_text"])
    Xte_job = vec_job_only.transform(test["job_text"])
    clf_job_only = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf_job_only.fit(Xtr_job, train["label"])
    job_only_acc = accuracy_score(test["label"], clf_job_only.predict(Xte_job))

    # 3. Cosine-similarity-only baseline + full pair model (for comparison)
    # FIX (review S3.3 / S4.9): "The cosine-only baseline uses a hand-picked
    # threshold (0.15). It should be tuned on training/validation data."
    # The threshold is now swept on the VALIDATION split (never on test) and
    # the best-F1 threshold is applied to test.
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    Xtr, cos_tr = build_features(train["resume_text"], train["job_text"], vectorizer, fit=True)
    Xte, cos_te = build_features(test["resume_text"], test["job_text"], vectorizer, fit=False)

    if len(val) > 0:
        _, cos_val = build_features(val["resume_text"], val["job_text"], vectorizer, fit=False)
        thresholds = np.linspace(0.0, 1.0, 101)
        best_t, best_f1 = 0.15, -1
        for t in thresholds:
            f1 = f1_score(val["label"], (cos_val.flatten() >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        tuned_threshold = best_t
        print(f"Cosine threshold tuned on validation split: {tuned_threshold:.2f} (val F1={best_f1:.4f})")
    else:
        tuned_threshold = 0.15
        print("WARNING: no validation split available -- falling back to untuned threshold 0.15")

    cos_pred = (cos_te.flatten() >= tuned_threshold).astype(int)
    cos_acc = accuracy_score(test["label"], cos_pred)

    clf_pair = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf_pair.fit(Xtr, train["label"])
    pair_pred = clf_pair.predict(Xte)
    pair_acc = accuracy_score(test["label"], pair_pred)
    pair_f1 = f1_score(test["label"], pair_pred)

    print("\n=== Diagnostic Baselines (fixed grouped-split data) ===")
    print(f"Majority-class baseline accuracy:     {majority_acc:.4f}")
    print(f"Job-only baseline accuracy:            {job_only_acc:.4f}  <- should be close to majority; if much higher, label leakage remains")
    print(f"Cosine-similarity-only accuracy:       {cos_acc:.4f}")
    print(f"Full TF-IDF pair model accuracy:       {pair_acc:.4f}  (F1={pair_f1:.4f})")

    diag = pd.DataFrame([
        {"baseline": "majority_class", "accuracy": majority_acc},
        {"baseline": "job_only", "accuracy": job_only_acc},
        {"baseline": "cosine_only", "accuracy": cos_acc, "tuned_threshold": tuned_threshold},
        {"baseline": "full_pair_tfidf_lr", "accuracy": pair_acc, "f1_score": pair_f1},
    ])
    diag.to_csv("results/diagnostic_baselines.csv", index=False)
    print("\nSaved to results/diagnostic_baselines.csv")
    return diag


if __name__ == "__main__":
    resumes = build_deduped_resumes()
    job_pool = build_deduped_job_pool()

    train_res, val_res, test_res = grouped_split(resumes)
    print(f"\nResume-level grouped split -> train: {len(train_res)}, val: {len(val_res)}, test: {len(test_res)}")
    # sanity check: no overlap
    assert set(train_res["resume_id"]) & set(test_res["resume_id"]) == set()
    assert set(train_res["resume_id"]) & set(val_res["resume_id"]) == set()
    print("Verified: zero resume_id overlap between splits.")

    all_pairs = pd.concat([
        build_pairs_for_split(train_res, job_pool, "train"),
        build_pairs_for_split(val_res, job_pool, "val"),
        build_pairs_for_split(test_res, job_pool, "test"),
    ], ignore_index=True)

    print(f"\nTotal pairs: {len(all_pairs)}")
    print(all_pairs.groupby("split")["label"].value_counts())

    print("\nNegative-pair job-category distribution (should now be spread across categories, not Data-Science-dominated):")
    print(all_pairs[all_pairs["label"] == 0]["job_category"].value_counts(normalize=True).head(8))

    out_path = "data/resume_job_pairs_v3.csv"
    if os.path.exists(out_path):
        print(f"\nRefusing to overwrite {out_path} (frozen evaluation pairs). "
              "New protocols write to data/eval_v7/ via eval_protocol_v7.py.")
    else:
        all_pairs.to_csv(out_path, index=False)
        print("\nSaved fixed pairs dataset to data/resume_job_pairs_v3.csv")

    run_diagnostics(pd.read_csv(out_path) if os.path.exists(out_path) else all_pairs)
