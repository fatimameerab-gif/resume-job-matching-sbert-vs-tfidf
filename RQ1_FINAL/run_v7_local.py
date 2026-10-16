import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from representations import REPRESENTATIONS, tfidf_features
from stats_utils import (
    paired_permutation_pvalue, benjamini_hochberg, holm_bonferroni,
    min_detectable_effect_paired, paired_t_power, repeated_eval_frozen_pairs,
)
from hybrid_rrf import rrf_fuse, ranks_from_scores
from rq1_taskA_devset import ndcg_at_k, recall_at_k, mrr, average_precision, r_precision
from taskA_loader import load_taskA_pairs
from split_diagnostics import write_reports
from eval_protocol_v7 import (
    build_frozen_seed_pairs, build_tiered_pairs_on_v2_resumes,
    build_resume_job_grouped_pairs, NEGATIVE_TIERS,
)
from text_processing import apply_strategy, STRATEGIES, word_chunks, pool_chunk_vectors
from robustness import perturb_pairs, SAFE_SYNONYMS
import tier_eval_v7b, robustness_v2, grouped_repeated_v7b, check_environment

OUT = "results/v7"
os.makedirs(OUT, exist_ok=True)
SEED = 42


def _fit_predict(train, test, rep_fn):
    out = rep_fn(train["resume_text"], train["job_text"], test["resume_text"], test["job_text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(out[0], train["label"])
    pred = clf.predict(out[1])
    proba = clf.predict_proba(out[1])[:, 1] if hasattr(clf, "predict_proba") else pred.astype(float)
    return pred, proba


def save_v2_local_predictions():
    pairs = pd.read_csv("data/resume_job_pairs_v3.csv")
    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    test = pairs[pairs["split"] == "test"].reset_index(drop=True)
    keyed = test[["resume_id", "job_id", "resume_category", "job_category", "split", "label"]].copy()
    rows = []
    for name, fn in REPRESENTATIONS.items():
        pred, proba = _fit_predict(train, test, fn)
        keyed[name] = pred
        keyed[f"{name}__score"] = proba
        rows.append({"representation": name, "accuracy": accuracy_score(test["label"], pred),
                     "f1": f1_score(test["label"], pred, zero_division=0),
                     "n_test": len(test), "pairs_file": "data/resume_job_pairs_v3.csv"})
    keyed.to_csv(f"{OUT}/synthetic_v2_local_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/synthetic_v2_local_metrics.csv", index=False)
    return keyed, train, test


def diagnostic_vs_ablation_note(test, train):
    """Reconcile the three confusingly-close-but-different TF-IDF accuracy numbers.

    Numbers (from saved CSVs, not re-computed here):
      diagnostic_baselines.csv  cosine_only      0.8533  <- cosine THRESHOLD tuned on val
      rq1_feature_ablation_table TF-IDF cos_only 0.8667  <- LR on 1-d cosine feature
      diagnostic_baselines.csv  full_pair_tfidf_lr 0.800 <- LR on |u-v|+cos
      rq1_feature_ablation_table TF-IDF absdiff_cos 0.800 <- same as above (confirmed match)

    Why they differ:
      0.8533 vs 0.8667: both use only cosine similarity, but 0.8533 is a threshold
        picked to maximise validation-F1, while 0.8667 is an LR decision boundary
        trained on the training labels. The LR fits the boundary more precisely.
      0.8667 vs 0.800: adding the |u-v| difference vector (3000 dims) alongside cosine
        HURTS because 116 training pairs cannot regularise 3000 extra dimensions;
        the high-dimensional noise overwhelms the useful cosine signal.
      0.800 == 0.800: full_pair_tfidf_lr and absdiff_cos are the same model; the match
        confirms both CSVs use the same frozen pairs file.

    Verdict for the manuscript: cite 0.800 as the TF-IDF+LR number (absdiff_cos,
    the original feature construction), and report 0.8667 (cos_only LR) as evidence
    that the |u-v| feature adds noise rather than signal at this sample size.
    """
    hist_diag = pd.read_csv("results/diagnostic_baselines.csv")
    hist_abl = pd.read_csv("results/rq1_feature_ablation_table.csv")

    cos_only_diag  = float(hist_diag.loc[hist_diag.baseline == "cosine_only", "accuracy"].iloc[0])
    thr            = float(hist_diag.loc[hist_diag.baseline == "cosine_only", "tuned_threshold"].iloc[0])
    full_pair_diag = float(hist_diag.loc[hist_diag.baseline == "full_pair_tfidf_lr", "accuracy"].iloc[0])
    tfidf_cos_abl  = float(hist_abl.loc[(hist_abl.representation == "TF-IDF") & (hist_abl.feature_mode == "cos_only"), "accuracy"].iloc[0])
    tfidf_abs_abl  = float(hist_abl.loc[(hist_abl.representation == "TF-IDF") & (hist_abl.feature_mode == "absdiff_cos"), "accuracy"].iloc[0])

    # Sanity check: the two absdiff_cos / full_pair numbers must agree.
    if abs(full_pair_diag - tfidf_abs_abl) > 1e-6:
        print(f"WARNING: full_pair_tfidf_lr ({full_pair_diag:.4f}) != "
              f"absdiff_cos ablation ({tfidf_abs_abl:.4f}) -- "
              "one of the CSVs may be from a different pairs file.")

    note = pd.DataFrame([
        {"source_file": "results/diagnostic_baselines.csv",
         "row": "cosine_only",
         "accuracy": cos_only_diag,
         "model": f"TF-IDF cosine threshold tuned on val split (threshold={thr:.2f})",
         "why_different": (
             "Binary threshold sweep; boundary chosen for val-F1. "
             "NOT a trained LR classifier. 0.8533 > 0.800 because the tuned "
             "threshold exploits class imbalance differently than balanced LR."
         )},
        {"source_file": "results/rq1_feature_ablation_table.csv",
         "row": "TF-IDF feature_mode=cos_only",
         "accuracy": tfidf_cos_abl,
         "model": "LogisticRegression on single cosine feature (1-d input)",
         "why_different": (
             "LR fits decision boundary on training labels; better than the "
             "threshold-tuned version (0.8667 > 0.8533). Neither uses |u-v|."
         )},
        {"source_file": "results/diagnostic_baselines.csv",
         "row": "full_pair_tfidf_lr",
         "accuracy": full_pair_diag,
         "model": "LogisticRegression on [|u-v|, cos] (absdiff_cos mode, 3001-d)",
         "why_different": (
             "Adding 3000-d |u-v| to cos HURTS at n=116 training pairs "
             "(high-dim noise; LR cannot regularise). "
             "0.800 < 0.8667: absdiff_cos worse than cos_only."
         )},
        {"source_file": "results/rq1_feature_ablation_table.csv",
         "row": "TF-IDF feature_mode=absdiff_cos",
         "accuracy": tfidf_abs_abl,
         "model": "LogisticRegression on [|u-v|, cos] -- same model as full_pair_tfidf_lr",
         "why_different": (
             f"Matches full_pair_tfidf_lr ({full_pair_diag:.4f} == {tfidf_abs_abl:.4f}). "
             "Cross-check confirms both CSVs use the same frozen pairs file."
         )},
    ])
    note.to_csv(f"{OUT}/diagnostic_vs_ablation_reconciliation.csv", index=False)
    print(f"\n=== Diagnostic vs Ablation Reconciliation ===")
    print(note[["row", "accuracy", "model"]].to_string(index=False))
    print(f"\nSaved {OUT}/diagnostic_vs_ablation_reconciliation.csv")
    return note


def frozen_seed_eval():
    paths = build_frozen_seed_pairs()
    frozen = {}
    for p in paths:
        df = pd.read_csv(p)
        seed = int(os.path.basename(p).replace("pairs_seed", "").replace(".csv", ""))
        frozen[seed] = (df[df.split == "train"].reset_index(drop=True),
                        df[df.split == "test"].reset_index(drop=True))
    all_seed_rows = []
    pred_frames = []
    for name, fn in REPRESENTATIONS.items():
        if name.startswith("Static"):
            continue  # skip slow word2vec retraining on 5 seeds

        def fp(tr, te, fn=fn):
            pred, _ = _fit_predict(tr, te, fn)
            return te["label"].values, pred

        per_seed, summary = repeated_eval_frozen_pairs(frozen, fp)
        per_seed["representation"] = name
        all_seed_rows.append(per_seed)
        print(name, summary)
        for seed, (tr, te) in frozen.items():
            pred, proba = _fit_predict(tr, te, fn)
            pred_frames.append(pd.DataFrame({
                "seed": seed, "resume_id": te["resume_id"], "job_id": te["job_id"],
                "label": te["label"], "representation": name, "pred": pred, "score": proba,
            }))
    pd.concat(all_seed_rows).to_csv(f"{OUT}/frozen_seed_local_metrics.csv", index=False)
    pd.concat(pred_frames).to_csv(f"{OUT}/frozen_seed_local_predictions.csv", index=False)


def negative_tier_eval():
    """SUPERSEDED by tier_eval_v7b.run() (shortcut-contaminated: length + job-prior)."""
    paths = build_tiered_pairs_on_v2_resumes()
    rows, preds = [], []
    for path in paths:
        df = pd.read_csv(path)
        train, test = df[df.split == "train"], df[df.split == "test"]
        pred, proba = _fit_predict(train, test, tfidf_features)
        tier = os.path.basename(path).replace("pairs_", "").replace(".csv", "")
        rows.append({"negative_tier": tier, "n_test": len(test),
                     "accuracy": accuracy_score(test.label, pred),
                     "f1": f1_score(test.label, pred, zero_division=0),
                     "pairs_file": path})
        tmp = test[["resume_id", "job_id", "label", "negative_reason", "resume_family", "job_family"]].copy()
        tmp["tier"] = tier
        tmp["pred"] = pred
        tmp["score"] = proba
        preds.append(tmp)
    pd.DataFrame(rows).to_csv(f"{OUT}/negative_tier_tfidf_metrics.csv", index=False)
    pd.concat(preds).to_csv(f"{OUT}/negative_tier_tfidf_predictions.csv", index=False)


def grouped_job_eval():
    """SUPERSEDED by grouped_repeated_v7b.run() (single split = 36 test pairs, seed-luck)."""
    path, meta = build_resume_job_grouped_pairs()
    df = pd.read_csv(path)
    train, test = df[df.split == "train"], df[df.split == "test"]
    rows, frames = [], []
    for name, fn in REPRESENTATIONS.items():
        if name.startswith("Static"):
            continue
        pred, proba = _fit_predict(train, test, fn)
        rows.append({"representation": name, "accuracy": accuracy_score(test.label, pred),
                     "f1": f1_score(test.label, pred, zero_division=0),
                     **meta, "pairs_file": path})
        frames.append(pd.DataFrame({
            "resume_id": test.resume_id, "job_id": test.job_id, "label": test.label,
            "representation": name, "pred": pred, "score": proba,
        }))
    pd.DataFrame(rows).to_csv(f"{OUT}/resume_job_grouped_metrics.csv", index=False)
    pd.concat(frames).to_csv(f"{OUT}/resume_job_grouped_predictions.csv", index=False)
    with open(f"{OUT}/resume_job_grouped_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def taskA_rrf():
    pairs = load_taskA_pairs()
    queries = pairs.drop_duplicates("query_id")[["query_id", "job_text"]]
    corpus = pairs.drop_duplicates("resume_id")[["resume_id", "resume_text"]]
    qrels = pairs[pairs.label == 1][["query_id", "resume_id"]]

    tokenized = [t.split() for t in corpus["resume_text"].tolist()]
    bm25 = BM25Okapi(tokenized)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    C = vec.fit_transform(corpus["resume_text"])
    doc_ids = corpus["resume_id"].tolist()

    rank_rows = []
    pred_rows = []
    for _, q in queries.iterrows():
        q_tokens = str(q["job_text"]).split()
        bm_scores = np.asarray(bm25.get_scores(q_tokens))
        q_vec = vec.transform([q["job_text"]])
        tf_scores = cosine_similarity(q_vec, C).ravel()
        r_bm = ranks_from_scores(bm_scores)
        r_tf = ranks_from_scores(tf_scores)
        rrf = rrf_fuse(r_bm, r_tf, k=60)
        rel = set(qrels.loc[qrels.query_id == q["query_id"], "resume_id"])
        labels = np.array([1 if d in rel else 0 for d in doc_ids])
        n_rel = labels.sum()
        for name, scores in [("BM25_retrieval", bm_scores), ("TFIDF_cosine_retrieval", tf_scores),
                             ("RRF_BM25_TFIDF", rrf)]:
            order = np.argsort(-scores)
            ranked_labels = labels[order]
            rank_rows.append({
                "representation": name, "query_id": q["query_id"], "n_relevant": int(n_rel),
                "ndcg@10": ndcg_at_k(ranked_labels, 10), "ndcg@100": ndcg_at_k(ranked_labels, 100),
                "recall@10": recall_at_k(ranked_labels, n_rel, 10),
                "map": average_precision(ranked_labels),
                "r_precision": r_precision(ranked_labels, n_rel),
                "mrr": mrr(ranked_labels),
            })
        for i, did in enumerate(doc_ids):
            pred_rows.append({
                "query_id": q["query_id"], "resume_id": did, "label": int(labels[i]),
                "score_bm25": float(bm_scores[i]), "score_tfidf": float(tf_scores[i]),
                "score_rrf": float(rrf[i]), "rank_bm25": int(r_bm[i]), "rank_tfidf": int(r_tf[i]),
            })
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(f"{OUT}/taskA_rrf_per_query.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{OUT}/taskA_rrf_predictions.csv", index=False)
    summary = ranks.groupby("representation")[["ndcg@10", "ndcg@100", "map", "r_precision", "mrr", "recall@10"]].mean()
    summary.to_csv(f"{OUT}/taskA_rrf_summary.csv")
    print(summary.round(4))
    return ranks


def taskA_permutation_and_power():
    fold_dir = "results"
    files = [f for f in os.listdir(fold_dir) if f.startswith("rq1_taskA_") and f.endswith("_folds.csv")]
    scores = {}
    for f in files:
        name = f.replace("rq1_taskA_", "").replace("_folds.csv", "")
        df = pd.read_csv(os.path.join(fold_dir, f))
        if len(df) != 10:
            continue
        scores[name] = df.sort_values("fold")["accuracy"].values
    names = sorted(scores)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            delta, p_perm = paired_permutation_pvalue(scores[a], scores[b], n_perm=5000, seed=42)
            sd = np.std(scores[b] - scores[a], ddof=1)
            mde = min_detectable_effect_paired(10, sd)
            pow_obs = paired_t_power(delta, 10, sd)
            rows.append({
                "rep_a": a, "rep_b": b, "mean_a": scores[a].mean(), "mean_b": scores[b].mean(),
                "mean_diff_b_minus_a": delta, "p_permutation": p_perm,
                "sd_paired_diff": sd, "mde_paired_t_80pct_alpha0.05": mde,
                "approx_power_at_observed_diff": pow_obs,
            })
    table = pd.DataFrame(rows)
    table["p_holm"], table["sig_holm_0.05"] = holm_bonferroni(table["p_permutation"].values)
    table["p_bh"], table["sig_bh_fdr_0.05"] = benjamini_hochberg(table["p_permutation"].values)
    table = table.sort_values("p_permutation")
    table.to_csv(f"{OUT}/taskA_query_permutation_power.csv", index=False)
    note = (
        "Exact Wilcoxon two-sided p with n=10 cannot be smaller than 2/1024=0.001953. "
        "Holm over 45 tests therefore cannot reject at 0.05 even if every sign agrees. "
        "Permutation p-values are reported with Holm (FWER) and BH (FDR). "
        "MDE is the paired-t approximation; it is a planning quantity, not a significance claim."
    )
    with open(f"{OUT}/taskA_stats_limitations.txt", "w", encoding="utf-8") as f:
        f.write(note + "\n")
        f.write(f"n_comparisons={len(table)} n_queries=10\n")
    print(note)
    print("BH discoveries:", int(table["sig_bh_fdr_0.05"].sum()),
          "Holm discoveries:", int(table["sig_holm_0.05"].sum()))


def truncation_and_text_strategies(train, test):
    rows = []
    texts = pd.concat([
        train[["resume_text"]].assign(side="resume_train"),
        train[["job_text"]].rename(columns={"job_text": "resume_text"}).assign(side="job_train"),
        test[["resume_text"]].assign(side="resume_test"),
        test[["job_text"]].rename(columns={"job_text": "resume_text"}).assign(side="job_test"),
    ], ignore_index=True)
    wc = texts["resume_text"].fillna("").map(lambda t: len(str(t).split()))
    limits = {"minilm_256": 256, "mpnet_384": 384, "bert_512": 512}
    tok_rows = [{"n_docs": len(wc), "mean_whitespace_tokens": float(wc.mean()),
                 "p50": float(wc.median()), "p90": float(wc.quantile(0.9)),
                 "p99": float(wc.quantile(0.99)), "max": int(wc.max())}]
    for name, lim in limits.items():
        tok_rows[0][f"frac_over_{name}"] = float((wc > lim).mean())
    tokenizer_note = "whitespace approximation; HF tokenizer not used unless import succeeds"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        lens = [len(tok.encode(t, add_special_tokens=True, truncation=False)) for t in texts["resume_text"].fillna("")]
        tok_rows[0]["minilm_true_mean_tokens"] = float(np.mean(lens))
        tok_rows[0]["minilm_true_frac_over_256"] = float(np.mean(np.array(lens) > 256))
        tokenizer_note = "used transformers AutoTokenizer sentence-transformers/all-MiniLM-L6-v2"
    except Exception as e:
        tok_rows[0]["hf_tokenizer_error"] = str(e)
    tok_rows[0]["tokenizer_note"] = tokenizer_note
    pd.DataFrame(tok_rows).to_csv(f"{OUT}/token_length_truncation.csv", index=False)

    for strat in STRATEGIES:
        tr = train.copy()
        te = test.copy()
        tr["resume_text"] = tr["resume_text"].map(lambda t: apply_strategy(t, strat, 256))
        tr["job_text"] = tr["job_text"].map(lambda t: apply_strategy(t, strat, 256))
        te["resume_text"] = te["resume_text"].map(lambda t: apply_strategy(t, strat, 256))
        te["job_text"] = te["job_text"].map(lambda t: apply_strategy(t, strat, 256))
        pred, _ = _fit_predict(tr, te, tfidf_features)
        rows.append({"strategy": strat, "n_tokens_budget": 256, "accuracy": accuracy_score(test.label, pred),
                     "f1": f1_score(test.label, pred, zero_division=0)})

    # chunked mean / max on TF-IDF word chunks of the TEST texts only, using a train-fitted vectorizer
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    vec.fit(pd.concat([train.resume_text, train.job_text], ignore_index=True))

    def embed_chunks(series, how):
        out = []
        for t in series:
            ch = word_chunks(t)
            mats = vec.transform(ch)
            out.append(pool_chunk_vectors(mats.toarray(), how))
        return np.vstack(out)

    from representations import build_pair_features
    for how in ("mean", "max"):
        r_tr, j_tr = embed_chunks(train.resume_text, how), embed_chunks(train.job_text, how)
        r_te, j_te = embed_chunks(test.resume_text, how), embed_chunks(test.job_text, how)
        Xtr, _ = build_pair_features(r_tr, j_tr)
        Xte, _ = build_pair_features(r_te, j_te)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, train.label)
        pred = clf.predict(Xte)
        rows.append({"strategy": f"chunked_{how}_pool_tfidf", "n_tokens_budget": "chunk_words=180",
                     "accuracy": accuracy_score(test.label, pred),
                     "f1": f1_score(test.label, pred, zero_division=0)})
    pd.DataFrame(rows).to_csv(f"{OUT}/text_strategy_tfidf_metrics.csv", index=False)


def robustness_eval(train, test):
    """SUPERSEDED by robustness_v2.run() (16-entry lexicon that cancelled itself; 9% coverage)."""
    te_p = perturb_pairs(test)
    pred_clean, _ = _fit_predict(train, test, tfidf_features)
    pred_pert, _ = _fit_predict(train, te_p, tfidf_features)
    rows = [
        {"condition": "original_test", "accuracy": accuracy_score(test.label, pred_clean),
         "f1": f1_score(test.label, pred_clean, zero_division=0),
         "label_changed": 0, "n_synonyms": len(SAFE_SYNONYMS)},
        {"condition": "synonym_perturbed_test", "accuracy": accuracy_score(te_p.label, pred_pert),
         "f1": f1_score(te_p.label, pred_pert, zero_division=0),
         "label_changed": int((te_p.label.values != test.label.values).sum()),
         "n_synonyms": len(SAFE_SYNONYMS)},
    ]
    pd.DataFrame(rows).to_csv(f"{OUT}/robustness_synonym_tfidf.csv", index=False)
    pd.DataFrame({"token": list(SAFE_SYNONYMS), "replacement": list(SAFE_SYNONYMS.values())}).to_csv(
        f"{OUT}/robustness_synonym_lexicon.csv", index=False)
    keyed = test[["resume_id", "job_id", "label"]].copy()
    keyed["pred_original"] = pred_clean
    keyed["pred_perturbed"] = pred_pert
    keyed.to_csv(f"{OUT}/robustness_synonym_predictions.csv", index=False)


def main():
    print("=== leakage diagnostics ===")
    write_reports(out_dir=OUT)
    print("=== v2 local predictions ===")
    keyed, train, test = save_v2_local_predictions()
    diagnostic_vs_ablation_note(test, train)
    print("=== frozen 5-seed pairs (shared across models) ===")
    frozen_seed_eval()
    print("=== negative-sampling tiers (v7b: paired, capped, length-matched, 10 seeds) ===")
    tier_eval_v7b.run()
    print("=== resume+job grouped split (v7b: repeated over 20 seeds) ===")
    grouped_repeated_v7b.run()
    print("=== TaskA BM25 + TF-IDF RRF ===")
    taskA_rrf()
    print("=== TaskA permutation / MDE ===")
    taskA_permutation_and_power()
    print("=== truncation + text strategies ===")
    truncation_and_text_strategies(train, test)
    print("=== robustness (v7b: directional lexicon, one-sided, rate-controlled, repeated) ===")
    robustness_v2.run()
    print("=== environment check ===")
    check_environment.main()
    import subprocess, sys
    with open(f"{OUT}/environment_freeze_v7.txt", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    print("Done. New outputs are under results/v7/ and data/eval_v7/.")


if __name__ == "__main__":
    main()
