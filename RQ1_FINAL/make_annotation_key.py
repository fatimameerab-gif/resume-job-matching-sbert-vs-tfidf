import numpy as np, pandas as pd
p = pd.read_csv("data/resume_job_pairs_v3.csv")
rng = np.random.RandomState(42)
parts = []
for label, n in ((1, 80), (0, 160)):
    g = p[p.label == label]
    share = g.job_source.value_counts(normalize=True)
    quota = (share * n).round().astype(int).clip(lower=8)
    quota = quota.clip(upper=g.job_source.value_counts())
    extra = n - quota.sum()
    quota[quota.idxmax()] += extra                         # absorb rounding in the largest source
    for src, q in quota.items():
        parts.append(g[g.job_source == src].sample(int(q), random_state=rng))
k = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
assert len(k) == 240 and k.duplicated(["resume_id", "job_id"]).sum() == 0
k.insert(0, "item_id", [f"A{i + 1:03d}" for i in range(len(k))])
k[["item_id", "resume_id", "job_id", "resume_category", "job_category", "label", "split", "job_source"]].to_csv(
    "annotation_kit/annotation-KEY-do-not-share-with-annotators.csv", index=False)
print(k.label.value_counts().to_dict(), k.job_source.value_counts().to_dict())
