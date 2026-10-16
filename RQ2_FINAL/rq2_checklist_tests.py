import os, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from representations import build_pair_features
from robustness_v2 import rank_change
OUT = "results/rq2"; INV_TOL = 0.02   # an INV test passes when |score change| < 0.02 (probability scale); fixed in advance
FILL = " references available upon request. available for interview at your convenience. thank you for your consideration."

def main(out=OUT, seeds=range(5), seed_dir="data/eval_v7/frozen_seeds"):
    os.makedirs(out, exist_ok=True); rows = []
    for seed in seeds:
        df = pd.read_csv(f"{seed_dir}/pairs_seed{seed}.csv"); tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
        vocab = np.array(v.get_feature_names_out())
        def topwords(job, k=10):
            w = v.transform([job]).toarray()[0]; return " ".join(w_ for w_ in vocab[np.argsort(-w)[:k]] if " " not in w_)
        by_cat = te.groupby("resume_category").resume_text.apply(list).to_dict()
        def foreign(cat):
            others = [c for c in by_cat if c != cat]; c = others[rng.integers(len(others))]; t = by_cat[c][rng.integers(len(by_cat[c]))]
            return " ".join(t.split()[:30])
        variants = {
            "INV_filler": (te.resume_text + FILL, None),
            "INV_punct": (te.resume_text.map(lambda s: re.sub(r"\s+", "  ", re.sub(r"[^\w\s]", "", s.upper()))), None),
            "DIR_up_copy_job": (te.resume_text + " " + te.job_text.map(topwords), "up"),
            "DIR_down_foreign": (te.resume_text + " " + te.resume_category.map(foreign), "down"),
        }
        for name, mode in (("tfidf_full", "absdiff_cos"), ("tfidf_cosLR", "cos_only")):
            f = lambda r, j: build_pair_features(v.transform(r), v.transform(j), mode=mode)[0]
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(f(tr.resume_text, tr.job_text), tr.label)
            base = clf.predict_proba(f(te.resume_text, te.job_text))[:, 1]
            for vn, (newres, direction) in variants.items():
                p = clf.predict_proba(f(newres, te.job_text))[:, 1]; d = p - base
                sel = np.ones(len(te), bool) if vn != "DIR_down_foreign" else (te.label.values == 1)
                if direction is None: ok = np.abs(d[sel]) < INV_TOL
                elif direction == "up": ok = d[sel] > 1e-9
                else: ok = d[sel] < -1e-9
                tau, top1 = rank_change(te, base, p)
                rows.append(dict(seed=seed, model=name, test=vn, expectation={"up": "score up", "down": "score down", None: "score unchanged"}[direction],
                                 n_pairs=int(sel.sum()), pass_rate=float(ok.mean()), mean_dscore=float(d[sel].mean()),
                                 flip_rate=float(((p >= .5) != (base >= .5))[sel].mean()), kendall_tau=tau, top1_change=top1))
    R = pd.DataFrame(rows); R.to_csv(f"{out}/checklist_per_seed.csv", index=False)
    S = R.groupby(["model", "test", "expectation"]).agg(pass_rate=("pass_rate", "mean"), pass_sd=("pass_rate", "std"), mean_dscore=("mean_dscore", "mean"), flip_rate=("flip_rate", "mean"), kendall_tau=("kendall_tau", "mean"), top1_change=("top1_change", "mean")).reset_index()
    S.round(4).to_csv(f"{out}/checklist_summary.csv", index=False); pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); main()
