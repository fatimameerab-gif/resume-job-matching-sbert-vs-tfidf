import os
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon

from data_protocol_v2 import (
    build_deduped_resumes, build_deduped_job_pool, grouped_split,
    build_pairs_for_split, run_diagnostics
)
from representations import REPRESENTATIONS
from stats_utils import (
    bootstrap_ci, repeated_grouped_eval, resume_clustered_bootstrap_ci,
    holm_bonferroni, cliffs_delta, friedman_nemenyi
)

RANDOM_STATE = 42
PAIRS_PATH = "data/resume_job_pairs_v3.csv"


def fit_predict_with_representation(train_pairs, test_pairs, rep_fn):
    out = rep_fn(
        train_pairs["resume_text"], train_pairs["job_text"],
        test_pairs["resume_text"], test_pairs["job_text"],
    )
    X_train, X_test, _, _, timing = out
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train_pairs["label"])
    y_pred = clf.predict(X_test)
    return test_pairs["label"].values, y_pred, timing


def main():
    resumes = build_deduped_resumes()
    job_pool = build_deduped_job_pool()

    if os.path.exists(PAIRS_PATH):
        print(f"Loading saved pairs from {PAIRS_PATH} (same split SBERT used)...")
        all_pairs_saved = pd.read_csv(PAIRS_PATH)
        train_pairs = all_pairs_saved[all_pairs_saved["split"] == "train"].reset_index(drop=True)
        test_pairs = all_pairs_saved[all_pairs_saved["split"] == "test"].reset_index(drop=True)
    else:
        print(f"WARNING: {PAIRS_PATH} not found -- generating fresh pairs. "
              "Run data_protocol_v2.py first if you need this to match a Colab SBERT run.")
        train_res, val_res, test_res = grouped_split(resumes, seed=RANDOM_STATE)
        train_pairs = build_pairs_for_split(train_res, job_pool, "train")
        test_pairs = build_pairs_for_split(test_res, job_pool, "test")

    all_results = []
    efficiency_rows = []
    per_rep_preds = {}  # for pairwise McNemar/effect-size comparisons

    for rep_name, rep_fn in REPRESENTATIONS.items():
        print(f"\n=== {rep_name} (seed={RANDOM_STATE}) ===")
        y_true, y_pred, timing = fit_predict_with_representation(train_pairs, test_pairs, rep_fn)
        per_rep_preds[rep_name] = y_pred

        # Resume-clustered bootstrap CI (review S4.9) -- replaces the naive
        # pair-level bootstrap that was here before.
        acc, acc_lo, acc_hi, n_valid = resume_clustered_bootstrap_ci(test_pairs, y_pred, accuracy_score)
        f1, f1_lo, f1_hi, _ = resume_clustered_bootstrap_ci(test_pairs, y_pred, f1_score)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        print(f"Accuracy: {acc:.4f}  (resume-clustered 95% CI: {acc_lo:.4f}-{acc_hi:.4f}, n_boot_valid={n_valid})")
        print(f"F1-score: {f1:.4f}  (resume-clustered 95% CI: {f1_lo:.4f}-{f1_hi:.4f})")
        print(f"Precision: {prec:.4f}  Recall: {rec:.4f}")
        print(f"Efficiency: fit={timing.get('fit_time_s', float('nan')):.3f}s  "
              f"encode(test)={timing.get('other_encode_time_s', float('nan')):.3f}s  "
              f"dim={timing.get('dim')}  index_bytes~={timing.get('index_bytes')}")

        all_results.append({
            "representation": rep_name, "seed": RANDOM_STATE,
            "accuracy": acc, "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
            "precision": prec, "recall": rec,
            "f1_score": f1, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
            "bootstrap_method": "resume_clustered",
        })
        efficiency_rows.append({"representation": rep_name, **timing})

    # ------------------------------------------------------------------
    # 5-seed repeated evaluation (mean +/- std), for robustness.
    # NOTE: these 5-seed CSVs are NOT paired across representations --
    # each call to repeated_grouped_eval() re-samples both the resume
    # split and the negative pairs, so seed k for TF-IDF and seed k for
    # SBERT land on different test sets (see stats_utils.repeated_grouped_eval
    # docstring). The Wilcoxon test below is therefore run on UNPAIRED
    # per-seed scores and is labelled accordingly; use repeated_eval_frozen_pairs
    # for a proper paired comparison (see results/v7/frozen_seed_local_*.csv).
    # ------------------------------------------------------------------
    print("\n\n=== 5-seed repeated evaluation (mean +/- std, NOT paired across reps) ===")
    seed_scores = {}  # rep_name -> [acc_seed0, ..., acc_seed4] for Friedman
    for rep_name, rep_fn in REPRESENTATIONS.items():
        def fp(train_pairs, test_pairs, rep_fn=rep_fn):
            y_true, y_pred, _ = fit_predict_with_representation(train_pairs, test_pairs, rep_fn)
            return y_true, y_pred
        per_seed_df, summary = repeated_grouped_eval(
            resumes, build_pairs_for_split, job_pool, fp, n_seeds=5
        )
        print(f"\n{rep_name}:")
        print(per_seed_df.to_string(index=False))
        print(f"  -> {summary}")
        safe_name = rep_name.split()[0].replace("-", "").replace("(", "").replace(")", "")
        per_seed_df.to_csv(f"results/rq1_{safe_name}_5seed.csv", index=False)
        seed_scores[rep_name] = per_seed_df["accuracy"].tolist()

    pd.DataFrame(all_results).to_csv("results/full_comparison.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv("results/rq1_efficiency.csv", index=False)
    print("\nSaved results/full_comparison.csv, results/rq1_efficiency.csv")

    # ------------------------------------------------------------------
    # Pairwise comparisons with Holm-Bonferroni correction + effect sizes.
    # LIMITATION: seed_scores are from repeated_grouped_eval() which
    # re-samples pairs each seed, so these are NOT paired scores.
    # Wilcoxon here tests whether the two distributions of per-seed
    # accuracies differ, not whether one representation beats the other on
    # the SAME test instances. With n=5 the minimum two-sided p is 0.0625,
    # so no result will reach alpha=0.05 regardless of the data. Use the
    # McNemar tests in rq1_synthetic_final_table.py (on the FIXED 75-pair
    # test set) for the definitive paired comparison.
    # ------------------------------------------------------------------
    print("\n=== Pairwise representation comparisons "
          "(5-seed accuracy, Wilcoxon -- UNPAIRED seeds, min p=0.0625) ===")
    pair_rows = []
    reps = list(seed_scores.keys())
    for a, b in itertools.combinations(reps, 2):
        sa, sb = np.array(seed_scores[a]), np.array(seed_scores[b])
        try:
            stat, p = wilcoxon(sa, sb)
        except ValueError:
            stat, p = np.nan, 1.0  # identical arrays / too few non-zero diffs
        delta = cliffs_delta(sa, sb)
        pair_rows.append({"rep_a": a, "rep_b": b, "wilcoxon_stat": stat,
                            "p_raw": p, "mean_diff": sb.mean() - sa.mean(),
                            "cliffs_delta": delta,
                            "paired": False,
                            "note": "seeds use different test pairs; Wilcoxon is descriptive only; min p=0.0625 with n=5"})
    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df) > 0:
        adj, reject = holm_bonferroni(pair_df["p_raw"].values, alpha=0.05)
        pair_df["p_holm_adjusted"] = adj
        pair_df["significant_at_0.05_holm"] = reject
    pair_df.to_csv("results/rq1_pairwise_representation_tests.csv", index=False)
    print(pair_df.to_string(index=False))
    print("\nSaved results/rq1_pairwise_representation_tests.csv")

    # ------------------------------------------------------------------
    # Friedman + Nemenyi across all representations, same 5 folds (Demsar
    # 2006 protocol, review S4.9/S7). Include SBERT if its Colab 5-seed CSV
    # has been copied in.
    # ------------------------------------------------------------------
    sbert_5seed_path = "results/rq1_SBERT_5seed.csv"
    if os.path.exists(sbert_5seed_path):
        sbert_seed_df = pd.read_csv(sbert_5seed_path)
        if len(sbert_seed_df) == 5:
            seed_scores["SBERT (from Colab)"] = sbert_seed_df["accuracy"].tolist()
            print("\nIncluded results/rq1_SBERT_5seed.csv (Colab) in the Friedman test.")
    else:
        print(f"\n(NOTE: {sbert_5seed_path} not found -- Friedman test below covers only "
              "locally-computable representations. Copy the Colab SBERT 5-seed CSV in "
              "to include SBERT / new encoders.)")

    if len(seed_scores) >= 3:
        try:
            stat, p, nemenyi = friedman_nemenyi(seed_scores)
            print(f"\n=== Friedman test across {len(seed_scores)} representations (5 seeds) ===")
            print(f"Friedman chi2={stat:.4f}  p={p:.4f}")
            nemenyi.to_csv("results/rq1_nemenyi_posthoc.csv")
            print("Nemenyi post-hoc p-value matrix saved to results/rq1_nemenyi_posthoc.csv")
        except Exception as e:
            print(f"\n(Friedman/Nemenyi failed: {e} -- likely too few seeds or degenerate scores)")
    else:
        print("\n(Skipping Friedman test: need >=3 representations with matched seed scores.)")

    # Diagnostic baselines on the same fixed pairs (now with tuned cosine threshold)
    # Include the VAL split so the cosine threshold is really tuned on val
    # (previously val was dropped here and the threshold silently fell back to 0.15).
    if os.path.exists(PAIRS_PATH):
        all_pairs = pd.read_csv(PAIRS_PATH)
    else:
        all_pairs = pd.concat([train_pairs, test_pairs], ignore_index=True)
    run_diagnostics(all_pairs)

    print("\n" + "=" * 70)
    print("Done. Remaining RQ1 items that need Google Colab (GPU/internet):")
    print(" - New SBERT-family / general encoders (e5, bge, gte) + a domain encoder")
    print(" - Pretrained static embeddings (GloVe/fastText) to replace the local word2vec stand-in")
    print(" - Fine-tuned bi-encoder (contrastive, hard negatives, learning curve)")
    print("See COLAB_NOTEBOOK.md for ready-to-run cells for all of the above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
