import os, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
SEED_DIR = "data/eval_v7/frozen_seeds"; OUT = "results/fullpool_ranking"; FT_DIR = "results/ft_embeddings"
from emb_paths import EMB_DIR
ENC = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "e5-base-v2", "bge-base-en-v1.5", "gte-base", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def metrics(score, rel):
    o = np.argsort(-score, kind="stable"); r = rel[o]; n_rel = int(rel.sum())
    if n_rel == 0: return None
    first = np.argmax(r) + 1; ap = float(np.sum(np.cumsum(r) / (np.arange(len(r)) + 1) * r) / n_rel)
    dcg = float(np.sum(r[:10] / np.log2(np.arange(2, 12)))); idcg = float(np.sum(1 / np.log2(np.arange(2, 2 + min(n_rel, 10)))))
    return dict(mrr=1 / first, hit10=float(r[:10].any()), p10=float(r[:10].mean()), ndcg10=dcg / idcg, map=ap)


def main():
    from data_protocol_v2 import build_deduped_job_pool
    os.makedirs(OUT, exist_ok=True); pool = build_deduped_job_pool().reset_index(drop=True); jt = pool.text_clean.astype(str).values; jc = pool.mapped_category.values
    E = {e: dict(zip(*(lambda z: (z["keys"], z["vectors"]))(np.load(f"{EMB_DIR}/{e}.npz", allow_pickle=True)))) for e in ENC if os.path.exists(f"{EMB_DIR}/{e}.npz")}
    rng = np.random.default_rng(0); rows, sizes = [], []
    for s in range(5):
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv"); tr, te = d[d.split == "train"], d[d.split == "test"]
        train_job_texts = set(tr.job_text.astype(str)); keep = np.array([t not in train_job_texts for t in jt]); cj_t, cj_c = jt[keep], jc[keep]
        sizes.append(dict(seed=s, test_resumes=te.resume_id.nunique(), pair_benchmark_candidates_per_resume=3, pool_candidates=int(keep.sum()), pool_unique_texts=len(jt)))
        rtr = tr.drop_duplicates("resume_id"); jtr = tr.drop_duplicates("job_id"); tres = te.drop_duplicates("resume_id")
        # --- scoring functions: each returns a (n_test_resumes x n_candidates) score matrix
        scorers = {}
        tv = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(pd.concat([rtr.resume_text, jtr.job_text])); scorers["TF-IDF cosine"] = (tv.transform(tres.resume_text) @ tv.transform(cj_t).T).toarray()
        for e, M in E.items():
            U = np.vstack([M[sha(t)] for t in tres.resume_text]); V = np.vstack([M[sha(t)] for t in cj_t]); scorers[f"{e} frozen cosine"] = U @ V.T
        classes = sorted(set(d.resume_category) | set(d.job_category) | set(jc))
        def full(m, X):
            P = np.zeros((X.shape[0], len(classes))); idx = {c: i for i, c in enumerate(classes)}; pr = m.predict_proba(X)
            for k, c in enumerate(m.classes_): P[:, idx[c]] = pr[:, k]
            return P
        tvr = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(rtr.resume_text); tvj = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(jtr.job_text)
        mr = LogisticRegression(C=10, max_iter=2000).fit(tvr.transform(rtr.resume_text), rtr.resume_category); mj = LogisticRegression(C=10, max_iter=2000).fit(tvj.transform(jtr.job_text), jtr.job_category)
        scorers["Category classifiers (TF-IDF)"] = full(mr, tvr.transform(tres.resume_text)) @ full(mj, tvj.transform(cj_t)).T
        for e in ("SBERT-mpnet-base-v2", "JobBERT-v2"):
            if e in E:
                g = lambda col, df: np.vstack([E[e][sha(t)] for t in df[col]]); m1 = LogisticRegression(C=10, max_iter=2000).fit(g("resume_text", rtr), rtr.resume_category); m2 = LogisticRegression(C=10, max_iter=2000).fit(g("job_text", jtr), jtr.job_category)
                scorers[f"Category classifiers ({e})"] = full(m1, g("resume_text", tres)) @ full(m2, np.vstack([E[e][sha(t)] for t in cj_t])).T
        for e in ENC:                                     # optional fine-tuned embeddings exported on the GPU
            p = f"{FT_DIR}/{e}_seed{s}.npz"
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True); M = dict(zip(z["keys"], z["vectors"])); U = np.vstack([M[sha(t)] for t in tres.resume_text]); V = np.vstack([M[sha(t)] for t in cj_t]); scorers[f"{e} fine-tuned cosine"] = U @ V.T
        scorers["Random (simulated)"] = None
        for name, S in scorers.items():
            for i, (_, r) in enumerate(tres.iterrows()):
                rel = (cj_c == r.resume_category).astype(int)
                if name.startswith("Random"):
                    ms = [metrics(rng.random(len(rel)), rel) for _ in range(50)]; m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]} if ms[0] else None
                else: m = metrics(S[i], rel)
                if m: rows.append(dict(seed=s, model=name, resume_id=r.resume_id, n_relevant=int(rel.sum()), n_candidates=len(rel), **m))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/per_resume.csv", index=False); pd.DataFrame(sizes).to_csv(f"{OUT}/candidate_sizes.csv", index=False)
    per_seed = R.groupby(["model", "seed"])[["mrr", "hit10", "p10", "ndcg10", "map"]].mean().reset_index(); per_seed.to_csv(f"{OUT}/per_seed.csv", index=False)
    S = per_seed.groupby("model")[["mrr", "hit10", "p10", "ndcg10", "map"]].agg(["mean", "std"]).round(3); S.columns = ["_".join(c) for c in S.columns]; S = S.sort_values("ndcg10_mean", ascending=False).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220); print(pd.DataFrame(sizes).to_string(index=False)); print(S.to_string(index=False))
    open(f"{OUT}/NOTE.txt", "w").write("Candidates = unique pool job texts not used in the seed's training pairs; relevant = same mapped category. About 40% of the pool is Data Science, so P@10 and MAP are easy for Data Science resumes: see per-category columns in per_resume.csv (n_relevant). Seeds share resumes (descriptive SDs). Compare with 3-candidate random: Hit@1 0.333, MRR 0.611.\n")


if __name__ == "__main__":
    main()
