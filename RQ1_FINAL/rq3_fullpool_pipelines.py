"""
Review point 15 (CPU): decision layers on the FULL job pool, with fine-tuned features.

Question: on the full-pool ranking (every test resume against all unique pool job texts that the seed's training pairs did not use, relevance =
same mapped category), is a tuned classifier on top of a representation better than the plain cosine score, and does a fine-tuned encoder beat
TF-IDF plus a tuned classifier? Same five frozen seeds, same candidate sets as rq1_fullpool_ranking.py.

Representations: TF-IDF (max 3000 features, 1-2 grams, English stop words, as in rq3_classifier_grid_v3.py), and for MiniLM, mpnet and JobBERT-v2 both the frozen
embeddings (results/embeddings) and the fine-tuned embeddings of the same seed (results/ft_embeddings, exported by gpu_finetune_export.py).
Scores per representation: (a) cosine, (b) tuned logistic regression and (c) tuned gradient boosting on [|u-v|, cos(u,v)], trained on the seed's
training pairs, hyper-parameters chosen by the same inner resume-grouped 3-fold CV (mean AUC) and grids as in rq3_classifier_grid_v3.py. The classifier
score (probability or margin) ranks the candidates; no threshold is involved.
Caveat: for the fine-tuned features the classifier is trained on the same training pairs the encoder was fine-tuned on, as in the RQ1 'fine-tuned head'.
Statistics: per-resume MRR (mean over the seeds in which the resume is a test resume), paired resume-level bootstrap (2,000 resamples) of the MRR
difference, bootstrap p-value (two-sided) with Holm correction over the listed comparisons.
Output: results/fullpool_pipelines/{per_resume.csv, summary.csv, paired.csv}.   Run: python rq3_fullpool_pipelines.py   (about 10-20 min on a CPU)
"""
import os, hashlib, warnings, itertools, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import rq3_grid_lib as G
from rq1_fullpool_ranking import metrics, ENC
from emb_paths import EMB_DIR
SEED_DIR = "data/eval_v7/frozen_seeds"; OUT = "results/fullpool_pipelines"; FT_DIR = "results/ft_embeddings"
ENCS = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "JobBERT-v2"]; CLFS = ["LR", "GradientBoosting"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def tune_fit(X, y, groups, name, seed):
    best = None
    for hp in G.GRIDS[name]:
        oof = np.zeros(len(y)); aucs = []
        for a, b in GroupKFold(3).split(X, y, groups):
            m = G.make(name, hp, seed).fit(X[a], y[a]); oof[b] = G.score(m, X[b])
            if len(set(y[b])) == 2: aucs.append(roc_auc_score(y[b], oof[b]))
        mu = float(np.mean(aucs))
        if best is None or mu > best[0]: best = (mu, hp)
    return G.make(name, best[1], seed).fit(X, y), best[1]


def main(seeds=range(5), out=OUT):
    from data_protocol_v2 import build_deduped_job_pool
    os.makedirs(out, exist_ok=True)
    pool = build_deduped_job_pool().reset_index(drop=True); jt = pool.text_clean.astype(str).values; jc = pool.mapped_category.values
    rows, t0 = [], time.time()
    for s in seeds:
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv"); tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"]
        keep = np.array([t not in set(tr.job_text.astype(str)) for t in jt]); cj_t, cj_c = jt[keep], jc[keep]; tres = te.drop_duplicates("resume_id").reset_index(drop=True)
        y, grp = tr.label.values, tr.resume_id.values
        reps = {}
        v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
        reps["TF-IDF"] = (lambda x, v=v: v.transform(x).toarray(), lambda x, v=v: v.transform(x).toarray())
        for e in ENCS:
            for kind, base in (("frozen", EMB_DIR), ("fine-tuned", FT_DIR)):
                f = f"{base}/{e}.npz" if kind == "frozen" else f"{base}/{e}_seed{s}.npz"
                if os.path.exists(f):
                    z = np.load(f, allow_pickle=True); M = dict(zip(z["keys"], z["vectors"]))
                    reps[f"{e} {kind}"] = (lambda x, M=M: np.vstack([M[sha(t)] for t in x]),) * 2
        for rname, (fr, fj) in reps.items():
            Utr, Vtr = fr(tr.resume_text), fj(tr.job_text); Ute, Vc = fr(tres.resume_text), fj(cj_t)
            Xtr = G.head(Utr, Vtr); scorers = {"cosine": lambda i: Vc @ Ute[i]}
            for c in CLFS:
                m, hp = tune_fit(Xtr, y, grp, c, seed=s)
                scorers[c] = lambda i, m=m: G.score(m, np.hstack([np.abs(Ute[i] - Vc), (Vc @ Ute[i])[:, None]]))
            for k, name in ((a, f"{rname} + {a}") for a in scorers):
                for i, r in tres.iterrows():
                    rel = (cj_c == r.resume_category).astype(int); mt = metrics(scorers[k](i), rel)
                    if mt: rows.append(dict(seed=s, model=name, resume_id=r.resume_id, n_relevant=int(rel.sum()), n_candidates=len(rel), **mt))
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{out}/per_resume.csv", index=False)
    S = R.groupby(["model", "seed"])[["mrr", "hit10", "p10", "ndcg10", "map"]].mean().reset_index().groupby("model")[["mrr", "hit10", "p10", "ndcg10", "map"]].agg(["mean", "std"])
    S.columns = ["_".join(c) for c in S.columns]
    pr = R.groupby(["model", "resume_id"]).mrr.mean().unstack(0); S["mrr_per_resume"] = pr.mean()
    S.sort_values("mrr_per_resume", ascending=False).round(3).reset_index().to_csv(f"{out}/summary.csv", index=False)
    # paired comparisons (resume-level bootstrap of per-resume MRR differences)
    comps = []
    for e in ENCS:
        n = {"SBERT-MiniLM-L6-v2": "SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2": "SBERT-mpnet-base-v2", "JobBERT-v2": "JobBERT-v2"}[e]
        comps += [(f"{n} fine-tuned + cosine", "TF-IDF + cosine"), (f"{n} fine-tuned + cosine", "TF-IDF + GradientBoosting"),
                  (f"{n} fine-tuned + cosine", "TF-IDF + LR"), (f"{n} fine-tuned + GradientBoosting", f"{n} fine-tuned + cosine"),
                  (f"{n} fine-tuned + LR", f"{n} fine-tuned + cosine"), (f"{n} fine-tuned + GradientBoosting", "TF-IDF + GradientBoosting")]
    comps += [("TF-IDF + GradientBoosting", "TF-IDF + cosine"), ("TF-IDF + LR", "TF-IDF + cosine")]
    res = []
    for a, b in comps:
        if a not in pr or b not in pr: continue
        dv = (pr[a] - pr[b]).dropna().values; rng = np.random.default_rng(0)
        bs = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(2000)])
        p = min(1.0, 2 * (min((bs <= 0).sum(), (bs >= 0).sum()) + 1) / 2001)
        res.append(dict(a=a, b=b, dMRR=dv.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p, n_resumes=len(dv)))
    P = pd.DataFrame(res); P["p_holm"] = G.holm(P.p.values); P.round(4).to_csv(f"{out}/paired.csv", index=False)
    pd.set_option("display.width", 220); print(S.sort_values("mrr_per_resume", ascending=False).round(3).to_string()); print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
