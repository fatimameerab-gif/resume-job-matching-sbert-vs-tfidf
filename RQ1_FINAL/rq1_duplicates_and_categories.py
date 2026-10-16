import os, hashlib
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csgraph import connected_components
from scipy import sparse
OUT = "results/duplicates_categories"; SEED_DIR = "data/eval_v7/frozen_seeds"
from emb_paths import EMB_DIR
EMB = f"{EMB_DIR}/SBERT-mpnet-base-v2.npz"
PRED = ["results/colab_v7b_finetune/pair_predictions.csv", "results/colab_v7b_newenc/finetune/pair_predictions.csv", "results/category_baseline/pair_predictions.csv"]
KEY = ["TF-IDF head", "TF-IDF cosine (zero-shot)", "SBERT-mpnet-base-v2 frozen head", "SBERT-mpnet-base-v2 fine-tuned head", "SBERT-MiniLM-L6-v2 fine-tuned head", "JobBERT-v2 frozen head",
       "JobBERT-v2 fine-tuned head", "Category classifiers (TF-IDF)", "Category classifiers (SBERT-mpnet-base-v2)", "Category classifiers (JobBERT-v2)"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()
THR = 0.9


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(EMB, allow_pickle=True); E = dict(zip(z["keys"], z["vectors"]))
    P = pd.concat([pd.read_csv(p) for p in PRED]).drop_duplicates(["seed", "model", "resume_id", "job_id"]); P = P[P.model.isin(KEY)]
    seeds = [pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv").assign(seed=s) for s in range(5)]; allp = pd.concat(seeds)
    res = allp.drop_duplicates("resume_id")[["resume_id", "resume_text", "resume_category"]].reset_index(drop=True)
    job = allp.drop_duplicates("job_id")[["job_id", "job_text", "job_category"]].reset_index(drop=True)
    def sims(texts, emb=True):
        tv = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(texts); X = tv.transform(texts); S1 = (X @ X.T).toarray()
        V = np.vstack([E[sha(t)] for t in texts]); S2 = V @ V.T; return S1, S2
    Sr_t, Sr_e = sims(res.resume_text); Sj_t, Sj_e = sims(job.job_text)
    ridx = {r: i for i, r in enumerate(res.resume_id)}; jidx = {j: i for i, j in enumerate(job.job_id)}
    # (2) clusters
    rows = []
    for nm, S in (("resume TF-IDF", Sr_t), ("resume mpnet", Sr_e)):
        A = S.copy(); np.fill_diagonal(A, 0); n, lab = connected_components(sparse.csr_matrix(A > THR))
        sizes = pd.Series(lab).value_counts(); rows.append(dict(kind=nm, threshold=THR, n_texts=len(res), n_clusters=n, texts_in_multi_member_clusters=int((sizes[sizes > 1]).sum()), largest_cluster=int(sizes.max())))
    for nm, S in (("job TF-IDF", Sj_t), ("job mpnet", Sj_e)):
        A = S.copy(); np.fill_diagonal(A, 0); n, lab = connected_components(sparse.csr_matrix(A > THR))
        sizes = pd.Series(lab).value_counts(); rows.append(dict(kind=nm, threshold=THR, n_texts=len(job), n_clusters=n, texts_in_multi_member_clusters=int((sizes[sizes > 1]).sum()), largest_cluster=int(sizes.max())))
    pd.DataFrame(rows).to_csv(f"{OUT}/near_duplicate_clusters.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
    # (1) flag test resumes / jobs near a train item
    flag = []
    for s, d in enumerate(seeds):
        tr, te = d[d.split == "train"], d[d.split == "test"]
        tr_r = [ridx[r] for r in tr.resume_id.unique()]; tr_j = [jidx[j] for j in tr.job_id.unique()]
        for kind, ids, trn, S_t, S_e, idx, col in (("resume", te.resume_id.unique(), tr_r, Sr_t, Sr_e, ridx, "resume_id"), ("job", te.job_id.unique(), tr_j, Sj_t, Sj_e, jidx, "job_id")):
            for i in ids:
                a = idx[i]; flag.append(dict(seed=s, kind=kind, id=i, max_cos_tfidf=S_t[a, trn].max(), max_cos_mpnet=S_e[a, trn].max()))
    F = pd.DataFrame(flag); F["near_dup"] = (F.max_cos_tfidf > THR) | (F.max_cos_mpnet > THR); F.to_csv(f"{OUT}/test_items_near_train.csv", index=False)
    summ = F.groupby(["seed", "kind"]).agg(n_test_items=("id", "size"), n_flagged=("near_dup", "sum")).reset_index(); summ.to_csv(f"{OUT}/near_duplicate_flag_summary.csv", index=False); print(summ.to_string(index=False))
    fr = F[F.kind == "resume"][["seed", "id", "near_dup"]].rename(columns={"id": "resume_id", "near_dup": "res_dup"}); fj = F[F.kind == "job"][["seed", "id", "near_dup"]].rename(columns={"id": "job_id", "near_dup": "job_dup"})
    P = P.merge(fr, on=["seed", "resume_id"], how="left").merge(fj, on=["seed", "job_id"], how="left")
    P["any_dup"] = P.res_dup.fillna(False) | P.job_dup.fillna(False)
    acc = P.groupby(["model", "any_dup"]).correct.agg(["mean", "size"]).unstack().round(3); acc.columns = [f"{a}_{'flagged' if b else 'unflagged'}" for a, b in acc.columns]; acc.to_csv(f"{OUT}/accuracy_flagged_vs_unflagged.csv"); print(acc.to_string())
    # (3) majority rate, per-category, macro, Data Science
    maj = allp[allp.split == "test"].groupby("seed").label.mean(); pd.DataFrame({"seed": maj.index, "positive_rate": maj.values, "majority_class_accuracy": np.maximum(maj.values, 1 - maj.values)}).to_csv(f"{OUT}/majority_rate.csv", index=False)
    pc = P.groupby(["model", "resume_category"]).correct.agg(["mean", "size"]).reset_index(); pc.round(3).to_csv(f"{OUT}/per_resume_category_accuracy.csv", index=False)
    macro = pc.groupby("model")["mean"].mean().rename("macro_accuracy_over_resume_categories")
    nods = P[(P.job_category != "Data Science")].groupby("model").correct.mean().rename("acc_without_DataScience_jobs")
    w = P.assign(w=1.0 / P.groupby(["model", "job_category"]).correct.transform("size")); eq = (w.assign(c=w.correct * w.w).groupby("model").c.sum() / w.groupby("model").w.sum()).rename("acc_equal_weight_per_job_category")
    all_acc = P.groupby("model").correct.mean().rename("acc_all_pairs")
    T = pd.concat([all_acc, macro, nods, eq], axis=1).round(3).sort_values("acc_all_pairs", ascending=False); T.to_csv(f"{OUT}/macro_and_datascience.csv"); print(T.to_string())
    print("share of job texts that are Data Science:", round((job.job_category == "Data Science").mean(), 3), "| majority-class accuracy per seed:", maj.round(3).tolist())
    open(f"{OUT}/NOTE.txt", "w").write("Near-duplicate threshold 0.9 (TF-IDF cosine or frozen mpnet cosine) to any TRAIN item of the same seed. Flags describe leakage risk; they are not removed from the main tables. Test positive rate = 1/3, so majority-class accuracy is 0.667.\n")


if __name__ == "__main__":
    main()
