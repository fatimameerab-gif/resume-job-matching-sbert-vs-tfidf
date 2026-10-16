import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from representations import build_pair_features

OUT = "results/rq2"; N_SPLITS = 20; SOURCES = ("seek_australia", "esco")

def build_pairs(resumes, pool, cats, rng):
    rows = []
    byc = {c: pool[pool.mapped_category == c].reset_index(drop=True) for c in cats}
    for _, r in resumes.iterrows():
        pj = byc[r.category].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        rows.append((r.resume_id, r.text_clean, pj.job_id, pj.text_clean, r.category, r.category, 1))
        for c in rng.choice([c for c in cats if c != r.category], 2, replace=False):
            nj = byc[c].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            rows.append((r.resume_id, r.text_clean, nj.job_id, nj.text_clean, r.category, c, 0))
    return pd.DataFrame(rows, columns=["resume_id", "resume_text", "job_id", "job_text", "resume_category", "job_category", "label"])

import hashlib
EMB_DIR = "results/embeddings"      # made by colab_export_embeddings.py (frozen dense encoders); optional
def sha(t): return hashlib.sha1(str(t).encode("utf8")).hexdigest()
class Dense:
    def __init__(self, name):
        z = np.load(f"{EMB_DIR}/{name}.npz", allow_pickle=True); self.E = dict(zip(z["keys"], z["vectors"])); self.name = name
    def mats(self, df): return np.vstack([self.E[sha(t)] for t in df.resume_text]), np.vstack([self.E[sha(t)] for t in df.job_text])
class Tfidf:
    name = "tfidf"
    def __init__(self, tr): self.v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
    def mats(self, df): return self.v.transform(df.resume_text), self.v.transform(df.job_text)
def available_dense():
    return [f[:-4] for f in sorted(os.listdir(EMB_DIR)) if f.endswith(".npz")] if os.path.isdir(EMB_DIR) else []
def feats(rep, df, mode):
    r, j = rep.mats(df); return build_pair_features(r, j, mode=mode)[0]

def hit_mrr(df, s):
    d = df[["resume_id", "label"]].copy(); d["s"] = np.asarray(s); h, r = [], []
    for _, g in d.groupby("resume_id"):
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + .5 * (o == p).sum()
        h.append(float(k == 1)); r.append(1 / k)
    return float(np.mean(h)), float(np.mean(r))

def main(n_splits=N_SPLITS, out=OUT):
    os.makedirs(out, exist_ok=True); dense_names = available_dense()
    resumes, pool = build_deduped_resumes(), build_deduped_job_pool()
    pools = {s: pool[pool.job_source == s] for s in SOURCES}
    shared = sorted(set(pools[SOURCES[0]].mapped_category) & set(pools[SOURCES[1]].mapped_category) & set(resumes.category))
    print(f"shared categories: {len(shared)}: {shared}")
    resumes = resumes[resumes.category.isin(shared)].reset_index(drop=True)
    texts_per = {s: pools[s][pools[s].mapped_category.isin(shared)].groupby("mapped_category").text_clean.nunique().to_dict() for s in SOURCES}
    pd.DataFrame(texts_per).rename_axis("category").to_csv(f"{out}/cross_source_job_texts_per_category.csv")
    rows, pred_rows = [], []
    for sp in range(n_splits):
        rng = np.random.default_rng(1000 + sp)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=sp).split(resumes, groups=resumes.resume_id))
        tr_r, te_r = resumes.iloc[tr_i], resumes.iloc[te_i]
        P = {s: (build_pairs(tr_r, pools[s], shared, rng), build_pairs(te_r, pools[s], shared, rng)) for s in SOURCES}
        for a in SOURCES:
            tr = P[a][0]
            for rep in [Tfidf(tr)] + [Dense(n) for n in dense_names]:
              for hname, mode in (("full", "absdiff_cos"), ("cosLR", "cos_only")):
                name, v = f"{rep.name}_{hname}", rep
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(v, tr, mode), tr.label)
                ptr = clf.predict_proba(feats(v, tr, mode))[:, 1]
                # threshold chosen on TRAIN data = value maximising balanced accuracy (not 0.5, not on test)
                cand = np.unique(np.round(ptr, 3)); thr = cand[np.argmax([balanced_accuracy_score(tr.label, ptr >= t) for t in cand])]
                for b in SOURCES:
                    te = P[b][1]; p = clf.predict_proba(feats(v, te, mode))[:, 1]; h, m = hit_mrr(te, p); pr = (p >= thr).astype(int)
                    rows.append(dict(split=sp, model=name, train_src=a, test_src=b, kind="within" if a == b else "cross", n_test_pairs=len(te),
                                     auc=roc_auc_score(te.label, p), hit1=h, mrr=m, acc=accuracy_score(te.label, pr),
                                     bal_acc=balanced_accuracy_score(te.label, pr), recall=recall_score(te.label, pr),
                                     pred_pos_rate=float(pr.mean()), threshold=float(thr)))
    R = pd.DataFrame(rows); R.to_csv(f"{out}/cross_source_per_split.csv", index=False)
    S = R.groupby(["model", "train_src", "test_src", "kind"]).agg(
        auc=("auc", "mean"), auc_sd=("auc", "std"), hit1=("hit1", "mean"), mrr=("mrr", "mean"), acc=("acc", "mean"),
        bal_acc=("bal_acc", "mean"), recall=("recall", "mean"), pred_pos_rate=("pred_pos_rate", "mean"), n_test_pairs=("n_test_pairs", "mean")).reset_index()
    # paired cross-vs-within gap per training source: auc(train A -> test B) - auc(train B -> test B), same splits and test resumes
    gaps = []
    for model, g in R.groupby("model"):
        for tgt in SOURCES:
            w = g[(g.train_src == tgt) & (g.test_src == tgt)].set_index("split").auc
            c = g[(g.train_src != tgt) & (g.test_src == tgt)].set_index("split").auc
            d = (c - w).dropna(); lo, hi = np.percentile([np.random.default_rng(i).choice(d.values, len(d)).mean() for i in range(2000)], [2.5, 97.5])
            gaps.append(dict(model=model, test_src=tgt, mean_auc_cross_minus_within=d.mean(), ci_lo=lo, ci_hi=hi,
                             wilcoxon_p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0))
    G = pd.DataFrame(gaps); S.round(4).to_csv(f"{out}/cross_source_summary.csv", index=False); G.round(4).to_csv(f"{out}/cross_source_gap.csv", index=False)
    open(f"{out}/NOTE_cross_source.txt", "w").write(
        f"Shared categories ({len(shared)}): {shared}\nBoth sources: 1 positive + 2 negatives per resume, negatives only from shared categories, same resume split for train/test, "
        f"{n_splits} repeated grouped splits. Threshold chosen on training data. A cross-source drop is source shift ONLY (label set is identical). "
        "The SEEK and ESCO pools have few job texts per category (see cross_source_job_texts_per_category.csv); with 1 positive text per resume drawn "
        "from a small pool the same text is reused across resumes, so read the numbers as an indication, not a precise estimate.\n")
    pd.set_option("display.width", 220); print(S.round(3).to_string(index=False)); print(G.round(3).to_string(index=False))

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); main()
