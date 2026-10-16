import re, numpy as np, pandas as pd
D = "results/colab_v7b_finetune/"
SEED_DIR = "data/eval_v7/frozen_seeds/"
clean = lambda s: re.sub(r" \(weighted w_tfidf=[0-9.]+\)", " (weighted)", s)

p = pd.read_csv(D + "finetune_per_seed.csv"); p["model"] = p.model.map(clean)
s = (p.groupby("model").agg(n_seeds=("seed", "nunique"), acc_mean=("acc", "mean"), acc_sd=("acc", "std"), bal_acc_mean=("bal_acc", "mean"),
                            f1_mean=("f1", "mean"), auc_mean=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
     .sort_values("acc_mean", ascending=False))
s["accuracy_reported"] = ~s.model.str.contains("RRF")        # RRF accuracy is not reported (threshold does not transfer); use AUC
s.round(4).to_csv(D + "finetune_summary_clean.csv", index=False)

P = pd.read_csv(D + "pair_predictions.csv"); P["model"] = P.model.map(clean)
parts = []
for seed in sorted(P.seed.unique()):
    d = pd.read_csv(f"{SEED_DIR}pairs_seed{seed}.csv"); seen = set(d[d.split == "train"].job_id)
    q = P[P.seed == seed].copy(); q["job_unseen_in_train"] = ~q.job_id.isin(seen); parts.append(q)
Q = pd.concat(parts)
u = Q.groupby(["model", "job_unseen_in_train"]).agg(acc=("correct", "mean"), n_pair_evals=("correct", "size")).reset_index()
u = u.pivot(index="model", columns="job_unseen_in_train", values=["acc", "n_pair_evals"])
u.columns = [f"{a}_{'unseen_job' if b else 'seen_job'}" for a, b in u.columns]
u.reset_index().sort_values("acc_unseen_job", ascending=False).round(4).to_csv(D + "finetune_unseen_job_subset.csv", index=False)

v = pd.read_csv(D + "paired_vs_tfidf.csv"); v["model"] = v.model.map(clean)
g = (v.groupby("model").agg(n_seeds=("seed", "nunique"), mean_diff_pp=("diff_pp", "mean"), min_diff_pp=("diff_pp", "min"), max_diff_pp=("diff_pp", "max"),
                            seeds_mcnemar_p_lt_005=("mcnemar_p", lambda x: int((x < 0.05).sum())),
                            discordant_only_candidate_right=("only_candidate_right", "sum"), discordant_only_tfidf_right=("only_tfidf_right", "sum"))
     .reset_index().sort_values("mean_diff_pp", ascending=False))
g.round(3).to_csv(D + "paired_vs_tfidf_summary.csv", index=False)
print(s.round(3).to_string(index=False)); print(); print(g.round(2).to_string(index=False))
