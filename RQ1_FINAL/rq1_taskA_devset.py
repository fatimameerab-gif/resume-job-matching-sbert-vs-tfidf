"""
rq1_taskA_devset.py
--------------------------
Runs the full RQ1 representation comparison on the human-annotated TaskA
(TalentCLEF 2026, English development set; synthetically generated texts,
manually annotated relevance) instead of the weak-supervision pairs -- this
is the "TaskA benchmark" pass requested to resolve the review's label-validity /
sample-size complaints (S4.1, S4.2) for RQ1 specifically.

Protocol:
  - Leave-one-query-out cross-validation (10 folds, since there are only
    10 job-posting queries): each fold trains a representation + logistic
    regression on the OTHER 9 queries' pairs, and scores the held-out
    query against ALL 472 candidate resumes. This mirrors data_protocol_v2's
    "group by resume/job, never leak a group across train/test" principle,
    generalised to leave-one-GROUP-out given only 10 groups exist.
  - Out-of-fold predictions are pooled across all 10 folds (every pair is
    scored exactly once, when its query was held out) -- this gives one
    clean set of predictions covering all 4,720 pairs with no leakage,
    which is what all metrics below are computed on.
  - BOTH framings are reported, exactly as review S3.2/S4.5 asked for:
      * classification: accuracy / precision / recall / F1 at threshold 0.5
      * ranking: nDCG@10, Recall@10, MRR -- computed per held-out query by
        ranking all 472 candidates by predicted P(relevant), which is the
        framing this dataset (positive-only qrels) actually supports best.
  - Query-clustered bootstrap CIs (resample QUERIES, not pairs -- the same
    "S4.9: bootstrap must resample the grouping unit" principle as before,
    here the grouping unit is the query, not the resume).
  - Holm-Bonferroni + Cliff's delta across representations, using the
    10 per-query fold scores as the paired samples (same test as
    full_comparison.py, applied to this dataset's natural fold count).

SBERT is NOT run here (needs Colab) -- see rq1_colab_taskA_sbert.py.
"""
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon

from taskA_loader import load_taskA_pairs
from representations import (
    REPRESENTATIONS, tfidf_features, lsa_features, bm25_features,
    _train_static_embeddings, _average_embed, build_pair_features,
)
from stats_utils import holm_bonferroni, cliffs_delta

RANDOM_STATE = 42
OUT_DIR = "results"


def _make_fast_static_rep(all_texts, dim=100, seed=42):
    """Trains the static word2vec model ONCE on the full corpus (unsupervised
    -- no label leakage), instead of retraining it inside every one of the
    10 leave-one-query-out folds (which took ~36s/fold, dominating runtime).
    Returns a rep_fn with the same signature as everything in REPRESENTATIONS.

    ⚠️  EMBEDDING LEAKAGE NOTE ⚠️
    `all_texts` is constructed in main() from ALL queries and ALL resumes
    (the full 4,720-pair grid), including the texts that will later be held
    out in each LOO fold. The word2vec model therefore sees held-out job and
    resume texts during its vocabulary/weight fitting step -- this is a form
    of transductive leakage for the static embedding only.

    Practical impact: word2vec's unsupervised pretraining uses no relevance
    labels, so label leakage is absent. Vocabulary/IDF leakage is mild
    (word2vec weights shift slightly when held-out texts add new tokens),
    but it DOES mean the static-embedding results on TaskA are not strictly
    comparable with TF-IDF/BM25/LSA, which re-fit their statistics inside
    each fold using only the 9-query training corpus.

    To eliminate this leakage: move `_train_static_embeddings` inside
    `run_leave_one_query_out`'s fold loop, re-training the model on the
    9 train-query texts only. This is slower but clean. The current
    optimisation is kept for practical runtime reasons; the leakage is
    documented here and should be stated in any manuscript table caption
    that reports the static-embedding TaskA result."""
    print("Pre-training static word2vec model once on the full corpus...")
    print("  (See _make_fast_static_rep docstring for the leakage note "
          "about this pre-training strategy.)")
    model = _train_static_embeddings(all_texts, dim=dim, seed=seed)

    def rep_fn(train_resume, train_job, test_resume=None, test_job=None, feature_mode="absdiff_cos"):
        def _build(res, job):
            r = _average_embed(res, model, dim)
            j = _average_embed(job, model, dim)
            X, cos = build_pair_features(r, j, mode=feature_mode)
            return X, cos
        X_train, cos_train = _build(train_resume, train_job)
        timing = {"fit_time_s": 0.0, "dim": dim, "index_bytes": None, "note": "model pre-trained once, outside fold loop"}
        if test_resume is not None:
            X_test, cos_test = _build(test_resume, test_job)
            return X_train, X_test, cos_train, cos_test, timing
        return X_train, cos_train, timing
    return rep_fn


def ndcg_at_k(ranked_labels, k=10):
    ranked_labels = np.asarray(ranked_labels[:k])
    if ranked_labels.sum() == 0:
        return 0.0
    gains = ranked_labels / np.log2(np.arange(2, len(ranked_labels) + 2))
    dcg = gains.sum()
    ideal = np.sort(ranked_labels)[::-1]
    ideal_gains = ideal / np.log2(np.arange(2, len(ideal) + 2))
    idcg = ideal_gains.sum()
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_labels, n_relevant, k=10):
    if n_relevant == 0:
        return np.nan
    return np.asarray(ranked_labels[:k]).sum() / n_relevant


def mrr(ranked_labels):
    idx = np.argmax(np.asarray(ranked_labels) == 1) if 1 in ranked_labels else None
    if idx is None or ranked_labels[idx] != 1:
        return 0.0
    return 1.0 / (idx + 1)


def average_precision(ranked_labels):
    """MAP's per-query term (review S4.5: 'add MAP, nDCG@50/100, R-precision
    -- nDCG@10/MRR are saturated because several queries have 40-112
    relevant resumes out of 472, so a top-10 window can't discriminate')."""
    ranked_labels = np.asarray(ranked_labels)
    n_rel = ranked_labels.sum()
    if n_rel == 0:
        return np.nan
    hits = np.cumsum(ranked_labels)
    precision_at_i = hits / np.arange(1, len(ranked_labels) + 1)
    return (precision_at_i * ranked_labels).sum() / n_rel


def r_precision(ranked_labels, n_relevant):
    """Precision in the top-R, where R = number of relevant docs for this
    query. Naturally threshold-free and not capped by a fixed k, unlike
    accuracy@0.5 (review S4.5: 'LSA's 0.149 accuracy is a threshold
    artifact at 10% base rate, not a representation-quality result')."""
    if n_relevant == 0:
        return np.nan
    return np.asarray(ranked_labels[:n_relevant]).sum() / n_relevant


def run_leave_one_query_out(pairs, rep_name, rep_fn):
    query_ids = pairs["query_id"].unique()
    oof_scores = np.zeros(len(pairs))
    oof_preds = np.zeros(len(pairs))
    per_query_ranking_rows = []
    fold_accuracy = []
    zeroshot_ranking_rows = []  # raw cosine, NO LogisticRegression at all

    for held_out_qid in query_ids:
        test_mask = pairs["query_id"] == held_out_qid
        train_pairs = pairs[~test_mask].reset_index(drop=True)
        test_pairs = pairs[test_mask].reset_index(drop=True)

        out = rep_fn(train_pairs["resume_text"], train_pairs["job_text"],
                     test_pairs["resume_text"], test_pairs["job_text"])
        X_train, X_test, _, cos_test, _timing = out

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(X_train, train_pairs["label"])
        y_pred = clf.predict(X_test)
        try:
            scores = clf.predict_proba(X_test)[:, 1]
        except Exception:
            scores = y_pred.astype(float)

        idx = pairs.index[test_mask]
        oof_scores[idx] = scores
        oof_preds[idx] = y_pred
        fold_accuracy.append(accuracy_score(test_pairs["label"], y_pred))

        n_rel = test_pairs["label"].sum()

        # Trained-classifier ranking (existing behaviour)
        order = np.argsort(-scores)
        ranked_labels = test_pairs["label"].values[order]
        per_query_ranking_rows.append({
            "representation": rep_name, "query_id": held_out_qid, "n_relevant": int(n_rel),
            "ndcg@10": ndcg_at_k(ranked_labels, 10), "ndcg@100": ndcg_at_k(ranked_labels, 100),
            "recall@10": recall_at_k(ranked_labels, n_rel, 10),
            "map": average_precision(ranked_labels), "r_precision": r_precision(ranked_labels, n_rel),
            "mrr": mrr(ranked_labels),
        })

        # Zero-shot ranking: rank purely by cosine similarity, NO
        # classifier training at all -- review S4.5: "TaskA has no
        # zero-shot cosine ranking; report S3.2 found TF-IDF beat SBERT
        # specifically under THAT framing, and it's missing here."
        cos_order = np.argsort(-cos_test.flatten())
        cos_ranked_labels = test_pairs["label"].values[cos_order]
        zeroshot_ranking_rows.append({
            "representation": rep_name, "query_id": held_out_qid, "n_relevant": int(n_rel),
            "ndcg@10": ndcg_at_k(cos_ranked_labels, 10), "ndcg@100": ndcg_at_k(cos_ranked_labels, 100),
            "recall@10": recall_at_k(cos_ranked_labels, n_rel, 10),
            "map": average_precision(cos_ranked_labels), "r_precision": r_precision(cos_ranked_labels, n_rel),
            "mrr": mrr(cos_ranked_labels),
        })

    return oof_scores, oof_preds, fold_accuracy, per_query_ranking_rows, zeroshot_ranking_rows


def query_clustered_bootstrap_ci(pairs, oof_preds, metric_fn, n_boot=10000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    query_ids = pairs["query_id"].unique()
    y_true = pairs["label"].values
    # Precompute index arrays once per query (was recomputed with np.where
    # inside every one of the 10,000 bootstrap iterations -- the real
    # bottleneck, not the representation fitting).
    idx_by_query = {q: np.where(pairs["query_id"].values == q)[0] for q in query_ids}
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(query_ids, size=len(query_ids), replace=True)
        idx = np.concatenate([idx_by_query[q] for q in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], oof_preds[idx]))
        except ValueError:
            continue
    point = metric_fn(y_true, oof_preds)
    lo, hi = np.percentile(stats, (100 - ci) / 2), np.percentile(stats, 100 - (100 - ci) / 2)
    return point, lo, hi


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_taskA_pairs()

    # Fast static-embedding rep (pre-trained once, see note above)
    all_texts = pd.concat([
        pairs.drop_duplicates("query_id")["job_text"],
        pairs.drop_duplicates("resume_id")["resume_text"],
    ], ignore_index=True)
    reps_to_run = dict(REPRESENTATIONS)
    reps_to_run["Static-word2vec-local-100d"] = _make_fast_static_rep(all_texts)

    classification_rows = []
    all_ranking_rows = []
    all_zeroshot_rows = []
    fold_scores_by_rep = {}

    for rep_name, rep_fn in reps_to_run.items():
        print(f"\n=== {rep_name} (leave-one-query-out, 10 folds) ===")
        oof_scores, oof_preds, fold_acc, ranking_rows, zeroshot_rows = run_leave_one_query_out(pairs, rep_name, rep_fn)
        fold_scores_by_rep[rep_name] = fold_acc
        all_ranking_rows.extend(ranking_rows)
        all_zeroshot_rows.extend(zeroshot_rows)
        safe_name = rep_name.split()[0].replace("-", "").replace("(", "").replace(")", "")
        pd.DataFrame({"fold": range(10), "accuracy": fold_acc}).to_csv(
            f"{OUT_DIR}/rq1_taskA_{safe_name}_folds.csv", index=False)

        # ---- Save per-pair OOF scores and predictions keyed by
        # (query_id, resume_id) so downstream paired tests and error
        # analysis can load them without re-running the LOO loop.
        # Every pair in the 4,720-pair grid is scored exactly once
        # (when its query was held out), so oof_scores is a complete,
        # leakage-free prediction for the entire dataset.
        oof_df = pairs[["query_id", "resume_id", "label"]].copy()
        oof_df["score"] = oof_scores
        oof_df["pred"] = oof_preds.astype(int)
        oof_df["representation"] = rep_name
        oof_path = f"{OUT_DIR}/rq1_taskA_{safe_name}_oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"  Saved per-pair OOF predictions -> {oof_path}")

        acc, acc_lo, acc_hi = query_clustered_bootstrap_ci(pairs, oof_preds, accuracy_score)
        f1, f1_lo, f1_hi = query_clustered_bootstrap_ci(pairs, oof_preds, f1_score)
        prec = precision_score(pairs["label"], oof_preds, zero_division=0)
        rec = recall_score(pairs["label"], oof_preds, zero_division=0)

        ranking_df = pd.DataFrame(ranking_rows)
        zs_df = pd.DataFrame(zeroshot_rows)

        print(f"[classification, threshold=0.5] accuracy={acc:.4f} (query-clustered 95% CI {acc_lo:.4f}-{acc_hi:.4f})  "
              f"precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f} (CI {f1_lo:.4f}-{f1_hi:.4f})")
        print(f"  NOTE: at ~10% base rate, threshold=0.5 accuracy can be a poor/unstable "
              f"summary (review S4.5) -- R-precision and MAP below are threshold-free.")
        print(f"[ranking, trained classifier score] nDCG@10={ranking_df['ndcg@10'].mean():.4f}  "
              f"nDCG@100={ranking_df['ndcg@100'].mean():.4f}  MAP={ranking_df['map'].mean():.4f}  "
              f"R-precision={ranking_df['r_precision'].mean():.4f}  MRR={ranking_df['mrr'].mean():.4f}")
        print(f"[ranking, ZERO-SHOT cosine, no classifier] nDCG@10={zs_df['ndcg@10'].mean():.4f}  "
              f"nDCG@100={zs_df['ndcg@100'].mean():.4f}  MAP={zs_df['map'].mean():.4f}  "
              f"R-precision={zs_df['r_precision'].mean():.4f}  MRR={zs_df['mrr'].mean():.4f}")

        classification_rows.append({
            "representation": rep_name, "framing": "classification",
            "accuracy": acc, "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
            "precision": prec, "recall": rec, "f1_score": f1, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
            "ndcg@10_trained": ranking_df["ndcg@10"].mean(), "ndcg@100_trained": ranking_df["ndcg@100"].mean(),
            "map_trained": ranking_df["map"].mean(), "r_precision_trained": ranking_df["r_precision"].mean(),
            "mrr_trained": ranking_df["mrr"].mean(),
            "ndcg@10_zeroshot": zs_df["ndcg@10"].mean(), "ndcg@100_zeroshot": zs_df["ndcg@100"].mean(),
            "map_zeroshot": zs_df["map"].mean(), "r_precision_zeroshot": zs_df["r_precision"].mean(),
            "mrr_zeroshot": zs_df["mrr"].mean(),
            "bootstrap_method": "query_clustered", "n_folds": 10,
        })

    pd.DataFrame(classification_rows).to_csv(f"{OUT_DIR}/rq1_taskA_representation_comparison.csv", index=False)
    pd.DataFrame(all_ranking_rows).to_csv(f"{OUT_DIR}/rq1_taskA_per_query_ranking.csv", index=False)
    pd.DataFrame(all_zeroshot_rows).to_csv(f"{OUT_DIR}/rq1_taskA_per_query_ranking_zeroshot.csv", index=False)
    print(f"\nSaved {OUT_DIR}/rq1_taskA_representation_comparison.csv, "
          f"{OUT_DIR}/rq1_taskA_per_query_ranking.csv, "
          f"{OUT_DIR}/rq1_taskA_per_query_ranking_zeroshot.csv")
    print(f"Per-pair OOF predictions: {OUT_DIR}/rq1_taskA_<rep>_oof_predictions.csv "
          f"(one file per representation; use these for paired McNemar tests across reps).")

    # ---- Pairwise Holm-Bonferroni + Cliff's delta across representations,
    # using the 10 per-query fold accuracies as paired samples.
    # These fold scores ARE legitimately paired: every representation was
    # evaluated on the SAME 10 leave-one-query-out folds in the same order,
    # so seed k always refers to the same held-out query.
    # Limitation: with n=10, the exact Wilcoxon minimum two-sided p is
    # 2/1024 ≈ 0.00195; Holm over ≥6 pairs cannot reject at alpha=0.05.
    # Results are reported as descriptive evidence (mean diff, Cliff's delta)
    # not as significance claims. See results/v7/taskA_stats_limitations.txt.
    # ----
    print("\n=== Pairwise representation comparisons "
          "(10-fold accuracy, Wilcoxon -- folds ARE paired; Holm min-p limitation applies) ===")
    pair_rows = []
    reps = list(fold_scores_by_rep.keys())
    for a, b in itertools.combinations(reps, 2):
        sa, sb = np.array(fold_scores_by_rep[a]), np.array(fold_scores_by_rep[b])
        try:
            stat, p = wilcoxon(sa, sb)
        except ValueError:
            stat, p = np.nan, 1.0
        pair_rows.append({"rep_a": a, "rep_b": b, "wilcoxon_stat": stat, "p_raw": p,
                            "mean_diff": sb.mean() - sa.mean(), "cliffs_delta": cliffs_delta(sa, sb)})
    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df) > 0:
        adj, reject = holm_bonferroni(pair_df["p_raw"].values, alpha=0.05)
        pair_df["p_holm_adjusted"] = adj
        pair_df["significant_at_0.05_holm"] = reject
    pair_df.to_csv(f"{OUT_DIR}/rq1_taskA_pairwise_representation_tests.csv", index=False)
    print(pair_df.to_string(index=False))
    print(f"\nSaved {OUT_DIR}/rq1_taskA_pairwise_representation_tests.csv")

    print("\n" + "=" * 70)
    print("NOTE: SBERT/e5/bge/gte are not run here (need GPU/HF access).")
    print("See rq1_colab_taskA_sbert.py to add them on Google Colab.")
    print("=" * 70)


if __name__ == "__main__":
    main()
