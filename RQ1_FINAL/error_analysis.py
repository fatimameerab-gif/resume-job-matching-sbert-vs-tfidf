import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool, grouped_split, build_pairs_for_split
from representations import tfidf_features, REPRESENTATIONS

RANDOM_STATE = 42
PAIRS_PATH = "data/resume_job_pairs_v3.csv"
SBERT_TEST_FEATURES = "results/sbert_test_features.npy"
SBERT_TEST_LABELS = "results/sbert_test_labels.csv"
SBERT_TRAIN_FEATURES = "results/sbert_train_features.npy"
SBERT_TRAIN_LABELS = "results/sbert_train_labels.csv"


def load_pairs():
    # FIX (reviewer issue #9): load the SAVED pairs CSV instead of regenerating.
    # The old version called grouped_split()/build_pairs_for_split() fresh here,
    # so this analysis ran on a DIFFERENT test set than the one behind every
    # reported RQ1/RQ3 number. That made the "error analysis" uninterpretable --
    # its misclassifications weren't the same misclassifications being reported.
    if not os.path.exists(PAIRS_PATH):
        raise SystemExit(f"{PAIRS_PATH} not found -- run data_protocol_v2.py / full_comparison.py first.")
    print(f"Loading saved pairs from {PAIRS_PATH} (same split used everywhere else)...")
    saved = pd.read_csv(PAIRS_PATH)
    return saved[saved["split"] == "train"].reset_index(drop=True), saved[saved["split"] == "test"].reset_index(drop=True)


def attach_saved_predictions(test_pairs):
    """Prefer keyed v7 predictions; else row-align the historical file."""
    keyed_paths = [
        "results/v7/rq1_synthetic_test_predictions_keyed.csv",
        "results/v7/synthetic_v2_local_predictions.csv",
    ]
    hist = "results/rq1_synthetic_test_predictions.csv"
    test_pairs = test_pairs.copy()
    for keyed in keyed_paths:
        if not os.path.exists(keyed):
            continue
        pred = pd.read_csv(keyed)
        merge_cols = ["resume_id", "job_id", "label"]
        attached = test_pairs.merge(pred, on=merge_cols, how="left", suffixes=("", "_predfile"))
        if "TF-IDF" not in attached.columns or attached["TF-IDF"].isna().any():
            continue
        print(f"Loaded saved predictions from {keyed}")
        attached["predicted_tfidf"] = attached["TF-IDF"].astype(int)
        return attached
    if os.path.exists(hist):
        pred = pd.read_csv(hist)
        if len(pred) != len(test_pairs) or not (pred["label"].values == test_pairs["label"].values).all():
            print("WARNING: historical prediction file is not aligned with saved test pairs.")
            return None
        print(f"Loaded row-aligned saved predictions from {hist}")
        out = test_pairs.copy()
        out["predicted_tfidf"] = pred["TF-IDF"].astype(int)
        for col in pred.columns:
            if col != "label":
                out[col] = pred[col].values
        return out
    return None


def main():
    train_pairs, test_pairs = load_pairs()
    os.makedirs("results/v7", exist_ok=True)

    attached = attach_saved_predictions(test_pairs)
    if attached is not None:
        test_pairs = attached
        y_pred = test_pairs["predicted_tfidf"].values
    else:
        X_train, X_test, _, _, _timing = tfidf_features(
            train_pairs["resume_text"], train_pairs["job_text"],
            test_pairs["resume_text"], test_pairs["job_text"],
        )
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(X_train, train_pairs["label"])
        y_pred = clf.predict(X_test)
        test_pairs = test_pairs.copy()
        test_pairs["predicted_tfidf"] = y_pred
    test_pairs["correct_tfidf"] = test_pairs["label"] == test_pairs["predicted_tfidf"]

    print("=== TF-IDF: overall confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(test_pairs["label"], y_pred))
    print("\n" + classification_report(test_pairs["label"], y_pred, target_names=["no-match", "match"]))

    print("\n=== TF-IDF: per resume-category F1 (categories with >=3 test pairs) ===")
    per_cat = []
    for cat, group in test_pairs.groupby("resume_category"):
        if len(group) < 3:
            continue
        f1 = f1_score(group["label"], group["predicted_tfidf"], zero_division=0)
        acc = (group["label"] == group["predicted_tfidf"]).mean()
        per_cat.append({"resume_category": cat, "n_pairs": len(group), "accuracy": acc, "f1_score": f1})
    per_cat_df = pd.DataFrame(per_cat).sort_values("f1_score")
    print(per_cat_df.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    per_cat_df.to_csv("results/v7/error_analysis_per_category.csv", index=False)

    print("\n=== All-representation predictions + pairwise disagreement counts ===")
    saved_name_map = {
        "TF-IDF": "tfidf",
        "LSA (TruncatedSVD-100d)": "lsa",
        "BM25": "bm25",
        "Static-word2vec-local-100d": "static-word2vec-local-100d",
        "SBERT all-MiniLM-L6-v2": "sbert",
        "e5-base-v2": "e5-base-v2",
        "bge-base-en-v1.5": "bge-base-en-v1.5",
        "gte-base": "gte-base",
    }
    pred_cols = {"tfidf": test_pairs["predicted_tfidf"].values}
    used_saved = False
    for src_col, dest in saved_name_map.items():
        if src_col in test_pairs.columns and dest != "tfidf":
            pred_cols[dest] = test_pairs[src_col].astype(int).values
            used_saved = True
    if not used_saved:
        for rep_name, rep_fn in REPRESENTATIONS.items():
            if rep_name == "TF-IDF":
                continue
            out = rep_fn(train_pairs["resume_text"], train_pairs["job_text"],
                         test_pairs["resume_text"], test_pairs["job_text"])
            Xtr_r, Xte_r, _, _, _ = out
            clf_r = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
            clf_r.fit(Xtr_r, train_pairs["label"])
            pred_cols[rep_name.split()[0].lower()] = clf_r.predict(Xte_r)
    else:
        print("Using saved per-pair predictions (no local re-fit).")

    # ---- All-representations discordant-pairs matrix (review S4.10: "the
    # most informative evidence for RQ1 is where representations disagree",
    # extended here beyond just TF-IDF vs SBERT to every locally-computable
    # representation) ----
    print("\n=== All-representation predictions + pairwise disagreement counts ===")
    pred_cols = {"tfidf": test_pairs["predicted_tfidf"].values}
    for rep_name, rep_fn in REPRESENTATIONS.items():
        if rep_name == "TF-IDF":
            continue
        out = rep_fn(train_pairs["resume_text"], train_pairs["job_text"],
                     test_pairs["resume_text"], test_pairs["job_text"])
        Xtr_r, Xte_r, _, _, _ = out
        clf_r = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf_r.fit(Xtr_r, train_pairs["label"])
        pred_cols[rep_name.split()[0].lower()] = clf_r.predict(Xte_r)

    pred_df = pd.DataFrame(pred_cols)
    pred_df["label"] = test_pairs["label"].values
    disagreement_rows = []
    names = list(pred_cols.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_disagree = (pred_df[a] != pred_df[b]).sum()
            disagreement_rows.append({"rep_a": a, "rep_b": b, "n_disagree": int(n_disagree),
                                        "pct_disagree": round(100 * n_disagree / len(pred_df), 1)})
    disagree_df = pd.DataFrame(disagreement_rows)
    print(disagree_df.to_string(index=False))
    pred_df.to_csv("results/v7/error_analysis_all_representation_predictions.csv", index=False)
    disagree_df.to_csv("results/v7/error_analysis_representation_disagreement_matrix.csv", index=False)
    print("Saved results/error_analysis_all_representation_predictions.csv and "
          "results/error_analysis_representation_disagreement_matrix.csv")

    # ---- SBERT error analysis + TF-IDF/SBERT discordant pairs ----
    # (reviewer: "Report where SBERT and TF-IDF disagree -- this is the most
    # informative evidence for RQ1.")
    if "SBERT all-MiniLM-L6-v2" in test_pairs.columns:
        test_pairs["predicted_sbert"] = test_pairs["SBERT all-MiniLM-L6-v2"].astype(int)
        test_pairs["correct_sbert"] = test_pairs["label"] == test_pairs["predicted_sbert"]
        discordant = test_pairs[test_pairs["predicted_tfidf"] != test_pairs["predicted_sbert"]].copy()
        print(f"\n=== TF-IDF vs SBERT discordant pairs (saved preds): {len(discordant)} of {len(test_pairs)} ===")
        discordant[["resume_category", "job_category", "label", "predicted_tfidf", "predicted_sbert",
                    "resume_text", "job_text"]].to_csv("results/v7/error_analysis_discordant_tfidf_sbert.csv", index=False)
    elif all(os.path.exists(p) for p in [SBERT_TEST_FEATURES, SBERT_TEST_LABELS, SBERT_TRAIN_FEATURES, SBERT_TRAIN_LABELS]):
        Xtr_s = np.load(SBERT_TRAIN_FEATURES)
        Xte_s = np.load(SBERT_TEST_FEATURES)
        ytr_s = pd.read_csv(SBERT_TRAIN_LABELS)["label"].values
        clf_s = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf_s.fit(Xtr_s, ytr_s)
        pred_sbert = clf_s.predict(Xte_s)

        if len(pred_sbert) == len(test_pairs):
            test_pairs["predicted_sbert"] = pred_sbert
            test_pairs["correct_sbert"] = test_pairs["label"] == test_pairs["predicted_sbert"]

            discordant = test_pairs[test_pairs["predicted_tfidf"] != test_pairs["predicted_sbert"]].copy()
            print(f"\n=== TF-IDF vs SBERT discordant pairs: {len(discordant)} of {len(test_pairs)} ===")
            tfidf_only_right = discordant[discordant["correct_tfidf"] & ~discordant["correct_sbert"]]
            sbert_only_right = discordant[discordant["correct_sbert"] & ~discordant["correct_tfidf"]]
            print(f"  TF-IDF right, SBERT wrong: {len(tfidf_only_right)}")
            print(f"  SBERT right, TF-IDF wrong: {len(sbert_only_right)}")
            discordant[["resume_category", "job_category", "label", "predicted_tfidf", "predicted_sbert",
                        "resume_text", "job_text"]].to_csv("results/v7/error_analysis_discordant_tfidf_sbert.csv", index=False)
            print("  Saved results/error_analysis_discordant_tfidf_sbert.csv -- read these rows for the "
                  "manuscript's qualitative RQ1 discussion (this is the strongest evidence for *why* one "
                  "representation wins, not just that it wins).")
        else:
            print("\n(Skipping SBERT discordant-pairs analysis: saved SBERT feature count doesn't match "
                  "the current test_pairs count -- re-export SBERT features from the current split first.)")
    else:
        print("\n(Skipping SBERT error analysis: results/sbert_*_features.npy not found. "
              "Copy them from the Colab run if you want TF-IDF/SBERT discordant-pair analysis.)")

    print("\n=== Sample TF-IDF misclassified pairs (for qualitative discussion in manuscript) ===")
    errors = test_pairs[~test_pairs["correct_tfidf"]].copy()
    sample_errors = errors.sample(min(15, len(errors)), random_state=RANDOM_STATE)
    for _, row in sample_errors.iterrows():
        print(f"\n[true={row['label']}, pred={row['predicted_tfidf']}] "
              f"resume_cat={row['resume_category']}, job_cat={row['job_category']}")
        print(f"  resume snippet: {row['resume_text'][:120]}...")
        print(f"  job snippet:    {row['job_text'][:120]}...")

    sample_errors[["resume_category", "job_category", "label", "predicted_tfidf", "resume_text", "job_text"]].to_csv(
        "results/v7/error_analysis_sample_misclassifications.csv", index=False
    )
    print("\nSaved error-analysis tables under results/v7/")


if __name__ == "__main__":
    main()
