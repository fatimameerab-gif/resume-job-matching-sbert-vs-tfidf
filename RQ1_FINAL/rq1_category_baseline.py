import os, glob, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
warnings.filterwarnings("ignore")
SEED_DIR = "data/eval_v7/frozen_seeds"
from emb_paths import EMB_DIR
OUT = "results/category_baseline"
ENC = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "e5-base-v2", "bge-base-en-v1.5", "gte-base", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def load_emb(name):
    z = np.load(f"{EMB_DIR}/{name}.npz", allow_pickle=True)
    return dict(zip(z["keys"], z["vectors"]))


def clf(C=10.0):
    return LogisticRegression(C=C, max_iter=2000)


def fit_side(X_tr, y_tr, classes):
    m = clf().fit(X_tr, y_tr)
    return m


def proba_full(m, X, classes):
    P = np.zeros((X.shape[0], len(classes))); idx = {c: i for i, c in enumerate(classes)}
    pr = m.predict_proba(X)
    for j, c in enumerate(m.classes_): P[:, idx[c]] = pr[:, j]
    return P


def rank_metrics(df, score):
    d = df.assign(s=score); hit, rr = [], []
    for r, g in d.groupby("resume_id"):
        if (g.label == 1).sum() != 1: continue
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values
        k = 1 + (o > p).sum() + 0.5 * (o == p).sum(); hit.append(float(k == 1)); rr.append(1 / k)
    return float(np.mean(hit)), float(np.mean(rr))


def main():
    os.makedirs(OUT, exist_ok=True)
    classes = None; rows, preds, unseen = [], [], []
    embs = {e: load_emb(e) for e in ENC if os.path.exists(f"{EMB_DIR}/{e}.npz")}
    for seed in range(5):
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{seed}.csv"); tr, te = d[d.split == "train"], d[d.split == "test"]
        classes = sorted(set(d.resume_category) | set(d.job_category))
        rtr = tr.drop_duplicates("resume_id"); jtr = tr.drop_duplicates("job_id")
        n_seen_res = set(rtr.resume_category); n_seen_job = set(jtr.job_category)
        unseen.append(dict(seed=seed, n_train_resumes=len(rtr), n_train_jobs=len(jtr), cats_in_train_resumes=len(n_seen_res), cats_in_train_jobs=len(n_seen_job),
                           test_pairs=len(te), test_pairs_with_unseen_resume_cat=int((~te.resume_category.isin(n_seen_res)).sum()),
                           test_pairs_with_unseen_job_cat=int((~te.job_category.isin(n_seen_job)).sum())))
        variants = {"TF-IDF": None, **{e: e for e in embs}}
        for name, e in variants.items():
            if e is None:
                tv_r = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(rtr.resume_text)
                tv_j = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(jtr.job_text)
                Xr_tr, Xr_te = tv_r.transform(rtr.resume_text), tv_r.transform(te.resume_text)
                Xj_tr, Xj_te = tv_j.transform(jtr.job_text), tv_j.transform(te.job_text)
            else:
                g = lambda col, df: np.vstack([embs[e][sha(t)] for t in df[col]])
                Xr_tr, Xr_te, Xj_tr, Xj_te = g("resume_text", rtr), g("resume_text", te), g("job_text", jtr), g("job_text", te)
            mr, mj = fit_side(Xr_tr, rtr.resume_category, classes), fit_side(Xj_tr, jtr.job_category, classes)
            Pr, Pj = proba_full(mr, Xr_te, classes), proba_full(mj, Xj_te, classes)
            score = (Pr * Pj).sum(1)
            hard = (Pr.argmax(1) == Pj.argmax(1)).astype(int); y = te.label.values
            hit, mrr = rank_metrics(te, score)
            rows.append(dict(seed=seed, model=f"Category classifiers ({name})", acc=accuracy_score(y, hard), bal_acc=balanced_accuracy_score(y, hard),
                             f1=f1_score(y, hard, zero_division=0), auc=roc_auc_score(y, score), hit1=hit, mrr=mrr,
                             resume_cat_acc=float(np.mean(np.array(classes)[Pr.argmax(1)] == te.resume_category.values)),
                             job_cat_acc=float(np.mean(np.array(classes)[Pj.argmax(1)] == te.job_category.values))))
            preds.append(te[["resume_id", "job_id", "resume_category", "job_category", "label"]].assign(seed=seed, model=f"Category classifiers ({name})", score=score, pred=hard, correct=(hard == y).astype(int)))
    per = pd.DataFrame(rows); per.to_csv(f"{OUT}/per_seed.csv", index=False)
    pd.concat(preds).to_csv(f"{OUT}/pair_predictions.csv", index=False)
    pd.DataFrame(unseen).to_csv(f"{OUT}/unseen_category_check.csv", index=False)
    s = per.groupby("model").agg(["mean", "std"]).drop(columns="seed", level=0)
    s.columns = ["_".join(c) for c in s.columns]; s = s.round(3).reset_index(); s.to_csv(f"{OUT}/summary.csv", index=False)
    print(s[["model", "acc_mean", "acc_std", "auc_mean", "hit1_mean", "mrr_mean", "resume_cat_acc_mean", "job_cat_acc_mean"]].to_string(index=False))
    print(pd.DataFrame(unseen).to_string(index=False))
    open(f"{OUT}/NOTE.txt", "w").write(
        "Two category classifiers (logistic regression, C=10, trained on the TRAIN resumes and TRAIN job texts of each frozen seed); pair score = P(same category); "
        "hard decision = argmax categories equal. No test-set tuning. Seeds share resumes, so SDs are descriptive. Compare with the fine-tuned and TF-IDF-head rows of "
        "results/colab_v7b_finetune on the same test pairs (see rq1_category_vs_models.py).\n")


if __name__ == "__main__":
    main()
