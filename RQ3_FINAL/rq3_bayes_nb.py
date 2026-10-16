import pandas as pd, itertools, numpy as np
from rq3_classifier_grid_v3 import holm
from stats_extra import nadeau_bengio_t, bayesian_correlated_t
R = pd.read_csv("results/rq3/grid_per_split_B.csv"); P = pd.read_csv("data/resume_job_pairs_v3.csv")
n = len(P); n_test = int(round(0.3 * n)); n_train = n - n_test      # approximate sizes of a 70/30 split of all pairs
rows = []
for metric in ("auc", "bal_acc"):
    for rep, g in R.groupby("representation"):
        M = g.pivot(index="split", columns="classifier", values=metric)
        for a, b in itertools.combinations(M.columns, 2):
            d = (M[a] - M[b]).values; rows.append(dict(kind="classifier_vs_classifier", representation=rep, metric=metric, a=a, b=b,
                **nadeau_bengio_t(d, n_train, n_test), **{k: v for k, v in bayesian_correlated_t(d, n_train, n_test).items() if k != "mean_diff"}))
    for clf, g in R.groupby("classifier"):
        M = g.pivot(index="split", columns="representation", values=metric)
        if {"TF-IDF", "LSA"} <= set(M.columns):
            d = (M["TF-IDF"] - M["LSA"]).values; rows.append(dict(kind="representation_vs_representation", representation=clf, metric=metric, a="TF-IDF", b="LSA",
                **nadeau_bengio_t(d, n_train, n_test), **{k: v for k, v in bayesian_correlated_t(d, n_train, n_test).items() if k != "mean_diff"}))
out = pd.DataFrame(rows); out["p_holm_nb"] = np.nan
for _, ix in out.groupby(["kind", "metric"]).groups.items(): out.loc[ix, "p_holm_nb"] = holm(out.loc[ix, "p"].values)
out.round(4).to_csv("results/rq3/bayes_nb_designB.csv", index=False)
print(out[(out.metric == "auc")].sort_values("p").head(12).round(3).to_string(index=False))
