"""Review 3.8: add the TF-IDF row to the efficiency table (CPU timing; hardware printed). Output: results/colab_v7b_extras/efficiency_tfidf.csv"""
import time, platform, numpy as np, pandas as pd, os
from sklearn.feature_extraction.text import TfidfVectorizer
d = pd.read_csv("data/eval_v7/frozen_seeds/pairs_seed0.csv"); tr = d[d.split == "train"]; docs = pd.concat([d.resume_text, d.job_text]).drop_duplicates().tolist()[:176]
t = time.perf_counter(); v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text])); fit_s = time.perf_counter() - t
ts = []
for _ in range(5): t = time.perf_counter(); X = v.transform(docs); ts.append(time.perf_counter() - t)
S = pd.DataFrame([dict(model="TF-IDF (1-2 grams, 3000 features)", n_docs=len(docs), sec_per_1k_docs=round(np.median(ts) / len(docs) * 1000, 3), fit_once_seconds=round(fit_s, 3), dim=X.shape[1],
    index_MB_per_100k_docs=round(X.nnz / len(docs) * 100000 * 12 / 1e6, 1), hardware=f"{platform.processor() or platform.machine()}, {os.cpu_count()} CPU cores, no GPU")])
S.to_csv("results/colab_v7b_extras/efficiency_tfidf.csv", index=False); print(S.to_string(index=False))
