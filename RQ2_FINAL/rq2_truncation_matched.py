import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from representations import build_pair_features
OUT = "results/rq2"; LENGTHS = (20, 50, 100, 200, None)

def cut(s, L): return s if L is None else " ".join(str(s).split()[:L])
def hit_mrr(df, s):
    d = df[["resume_id", "label"]].copy(); d["s"] = np.asarray(s); h, r = [], []
    for _, g in d.groupby("resume_id"):
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + .5 * (o == p).sum(); h.append(float(k == 1)); r.append(1 / k)
    return np.mean(h), np.mean(r)

def main(out=OUT, seeds=range(5), seed_dir="data/eval_v7/frozen_seeds"):
    os.makedirs(out, exist_ok=True); rows = []
    for seed in seeds:
        df = pd.read_csv(f"{seed_dir}/pairs_seed{seed}.csv"); tr0, te0 = df[df.split == "train"], df[df.split == "test"]
        for design in ("matched", "shifted"):
            for L in LENGTHS:
                tr = tr0.assign(resume_text=tr0.resume_text.map(lambda s: cut(s, L if design == "matched" else None)), job_text=tr0.job_text.map(lambda s: cut(s, L if design == "matched" else None)))
                te = te0.assign(resume_text=te0.resume_text.map(lambda s: cut(s, L)), job_text=te0.job_text.map(lambda s: cut(s, L)))
                v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
                for name, mode in (("tfidf_full", "absdiff_cos"), ("tfidf_cosLR", "cos_only")):
                    f = lambda d: build_pair_features(v.transform(d.resume_text), v.transform(d.job_text), mode=mode)[0]
                    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(f(tr), tr.label); p = clf.predict_proba(f(te))[:, 1]
                    h, m = hit_mrr(te, p)
                    rows.append(dict(seed=seed, design=design, words=L or "full", model=name, auc=roc_auc_score(te.label, p), acc=accuracy_score(te.label, p >= .5), hit1=h, mrr=m))
    R = pd.DataFrame(rows); R.to_csv(f"{out}/truncation_per_seed.csv", index=False)
    S = R.groupby(["model", "design", "words"], sort=False).agg(auc=("auc", "mean"), auc_sd=("auc", "std"), acc=("acc", "mean"), acc_sd=("acc", "std"), hit1=("hit1", "mean"), mrr=("mrr", "mean")).reset_index()
    S.round(4).to_csv(f"{out}/truncation_summary.csv", index=False)
    pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); main()
