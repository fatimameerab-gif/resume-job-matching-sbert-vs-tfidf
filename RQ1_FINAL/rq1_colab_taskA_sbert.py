import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from taskA_loader import load_taskA_pairs

RANDOM_STATE = 42
OUT_DIR = "results"

ENCODERS = {
    "SBERT-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "SBERT-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "e5-base-v2": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", "", ""),
    "gte-base": ("thenlper/gte-base", "", ""),
}
DOMAIN_ENCODER_CANDIDATES = ["TechWolf/JobBERT-v2", "jjzha/jobbert-base-cased"]


def build_pair_features(r, j):
    r_norm, j_norm = np.linalg.norm(r, axis=1), np.linalg.norm(j, axis=1)
    denom = r_norm * j_norm
    denom[denom == 0] = 1e-10
    cos = (np.sum(r * j, axis=1) / denom).reshape(-1, 1)
    return np.hstack([np.abs(r - j), cos]), cos


def ndcg_at_k(ranked_labels, k=10):
    ranked_labels = np.asarray(ranked_labels[:k])
    if ranked_labels.sum() == 0:
        return 0.0
    gains = ranked_labels / np.log2(np.arange(2, len(ranked_labels) + 2))
    ideal = np.sort(ranked_labels)[::-1]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return gains.sum() / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_labels, n_relevant, k=10):
    return np.nan if n_relevant == 0 else np.asarray(ranked_labels[:k]).sum() / n_relevant


def mrr(ranked_labels):
    if 1 not in ranked_labels:
        return 0.0
    idx = np.argmax(np.asarray(ranked_labels) == 1)
    return 1.0 / (idx + 1)


def average_precision(ranked_labels):
    ranked_labels = np.asarray(ranked_labels)
    n_rel = ranked_labels.sum()
    if n_rel == 0:
        return np.nan
    hits = np.cumsum(ranked_labels)
    precision_at_i = hits / np.arange(1, len(ranked_labels) + 1)
    return (precision_at_i * ranked_labels).sum() / n_rel


def r_precision(ranked_labels, n_relevant):
    if n_relevant == 0:
        return np.nan
    return np.asarray(ranked_labels[:n_relevant]).sum() / n_relevant


def _ranking_row(rep_name, qid, n_rel, ranked_labels):
    return {
        "representation": rep_name, "query_id": qid, "n_relevant": int(n_rel),
        "ndcg@10": ndcg_at_k(ranked_labels, 10), "ndcg@100": ndcg_at_k(ranked_labels, 100),
        "recall@10": recall_at_k(ranked_labels, n_rel, 10),
        "map": average_precision(ranked_labels), "r_precision": r_precision(ranked_labels, n_rel),
        "mrr": mrr(ranked_labels),
    }


def run_leave_one_query_out_encoder(pairs, model, prefix_r, prefix_j, name):
    """Encodes every unique query/resume text ONCE (unsupervised, so
    precomputing outside the fold loop is valid and ~10x faster)."""
    unique_jobs = pairs.drop_duplicates("query_id")[["query_id", "job_text"]]
    unique_resumes = pairs.drop_duplicates("resume_id")[["resume_id", "resume_text"]]
    t0 = time.perf_counter()
    job_emb = model.encode([prefix_r + t for t in unique_jobs["job_text"]], show_progress_bar=True)
    resume_emb = model.encode([prefix_j + t for t in unique_resumes["resume_text"]], show_progress_bar=True)
    encode_time = time.perf_counter() - t0
    job_emb_map = dict(zip(unique_jobs["query_id"], job_emb))
    resume_emb_map = dict(zip(unique_resumes["resume_id"], resume_emb))

    r_all = np.array([resume_emb_map[rid] for rid in pairs["resume_id"]])
    j_all = np.array([job_emb_map[qid] for qid in pairs["query_id"]])
    X_all, cos_all = build_pair_features(r_all, j_all)

    query_ids = pairs["query_id"].unique()
    oof_preds = np.zeros(len(pairs))
    fold_acc = []
    ranking_rows, zeroshot_rows = [], []
    for held_out_qid in query_ids:
        test_mask = (pairs["query_id"] == held_out_qid).values
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(X_all[~test_mask], pairs["label"][~test_mask])
        y_pred = clf.predict(X_all[test_mask])
        scores = clf.predict_proba(X_all[test_mask])[:, 1]
        oof_preds[test_mask] = y_pred
        y_true_fold = pairs["label"][test_mask].values
        fold_acc.append(accuracy_score(y_true_fold, y_pred))
        n_rel = y_true_fold.sum()

        order = np.argsort(-scores)
        ranking_rows.append(_ranking_row(name, held_out_qid, n_rel, y_true_fold[order]))

        cos_fold = cos_all[test_mask].flatten()
        cos_order = np.argsort(-cos_fold)
        zeroshot_rows.append(_ranking_row(name, held_out_qid, n_rel, y_true_fold[cos_order]))

    return oof_preds, fold_acc, ranking_rows, zeroshot_rows, encode_time, X_all.shape[1] - 1


def _summarize_and_save(name, pairs, oof_preds, fold_acc, ranking_rows, zeroshot_rows, enc_time, dim,
                          all_rows, all_ranking, all_zeroshot, efficiency_rows):
    acc = accuracy_score(pairs["label"], oof_preds)
    prec = precision_score(pairs["label"], oof_preds, zero_division=0)
    rec = recall_score(pairs["label"], oof_preds, zero_division=0)
    f1 = f1_score(pairs["label"], oof_preds, zero_division=0)
    rdf = pd.DataFrame(ranking_rows)
    zdf = pd.DataFrame(zeroshot_rows)
    print(f"{name}: accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print(f"  [trained]   nDCG@10={rdf['ndcg@10'].mean():.4f}  nDCG@100={rdf['ndcg@100'].mean():.4f}  "
          f"MAP={rdf['map'].mean():.4f}  R-prec={rdf['r_precision'].mean():.4f}  MRR={rdf['mrr'].mean():.4f}")
    print(f"  [zero-shot] nDCG@10={zdf['ndcg@10'].mean():.4f}  nDCG@100={zdf['ndcg@100'].mean():.4f}  "
          f"MAP={zdf['map'].mean():.4f}  R-prec={zdf['r_precision'].mean():.4f}  MRR={zdf['mrr'].mean():.4f}")
    print(f"  encode_time={enc_time:.1f}s  dim={dim}")

    all_rows.append({
        "representation": name, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "ndcg@10_trained": rdf["ndcg@10"].mean(), "ndcg@100_trained": rdf["ndcg@100"].mean(),
        "map_trained": rdf["map"].mean(), "r_precision_trained": rdf["r_precision"].mean(), "mrr_trained": rdf["mrr"].mean(),
        "ndcg@10_zeroshot": zdf["ndcg@10"].mean(), "ndcg@100_zeroshot": zdf["ndcg@100"].mean(),
        "map_zeroshot": zdf["map"].mean(), "r_precision_zeroshot": zdf["r_precision"].mean(), "mrr_zeroshot": zdf["mrr"].mean(),
        "encode_time_s": enc_time, "dim": dim,
    })
    all_ranking.extend(ranking_rows)
    all_zeroshot.extend(zeroshot_rows)
    efficiency_rows.append({"representation": name, "dim": dim, "encode_time_s_total": enc_time})

    safe = name.replace("/", "_").replace("-", "").replace(":", "_")
    pd.DataFrame({"fold": range(len(fold_acc)), "accuracy": fold_acc}).to_csv(
        f"{OUT_DIR}/rq1_taskA_{safe}_folds.csv", index=False)


def main():
    from sentence_transformers import SentenceTransformer
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_taskA_pairs()

    all_rows, all_ranking, all_zeroshot, efficiency_rows = [], [], [], []

    for name, (path, pr, pj) in ENCODERS.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        model = SentenceTransformer(path)
        oof_preds, fold_acc, ranking_rows, zeroshot_rows, enc_time, dim = \
            run_leave_one_query_out_encoder(pairs, model, pr, pj, name)
        _summarize_and_save(name, pairs, oof_preds, fold_acc, ranking_rows, zeroshot_rows, enc_time, dim,
                             all_rows, all_ranking, all_zeroshot, efficiency_rows)
        pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/rq1_taskA_encoder_comparison.csv", index=False)

    # Domain encoder -- now gets the SAME full treatment as every other encoder
    print(f"\n{'='*60}\nDomain encoder\n{'='*60}")
    for cand in DOMAIN_ENCODER_CANDIDATES:
        try:
            model = SentenceTransformer(cand)
            print(f"Loaded domain encoder: {cand}")
            name = f"domain:{cand}"
            oof_preds, fold_acc, ranking_rows, zeroshot_rows, enc_time, dim = \
                run_leave_one_query_out_encoder(pairs, model, "", "", name)
            _summarize_and_save(name, pairs, oof_preds, fold_acc, ranking_rows, zeroshot_rows, enc_time, dim,
                                 all_rows, all_ranking, all_zeroshot, efficiency_rows)
            break
        except Exception as e:
            print(f"Could not load {cand}: {e}")

    pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/rq1_taskA_encoder_comparison.csv", index=False)
    pd.DataFrame(all_ranking).to_csv(f"{OUT_DIR}/rq1_taskA_encoder_ranking.csv", index=False)
    pd.DataFrame(all_zeroshot).to_csv(f"{OUT_DIR}/rq1_taskA_encoder_ranking_zeroshot.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv(f"{OUT_DIR}/rq1_taskA_encoder_efficiency.csv", index=False)
    print(f"\nSaved rq1_taskA_encoder_comparison.csv, rq1_taskA_encoder_ranking.csv, "
          f"rq1_taskA_encoder_ranking_zeroshot.csv, rq1_taskA_encoder_efficiency.csv, "
          f"and one rq1_taskA_<name>_folds.csv per encoder (incl. the domain encoder).")
    print("\nCopy the whole results/ folder back and merge with the local TaskA CSVs "
          "(rq1_taskA_representation_comparison.csv).")


if __name__ == "__main__":
    main()
