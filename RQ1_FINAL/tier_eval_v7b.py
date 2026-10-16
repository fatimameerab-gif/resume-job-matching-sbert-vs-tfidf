import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from eval_protocol_v7 import build_pairs_v7, fit_hard_index, NEGATIVE_TIERS
from representations import tfidf_features

OUT = "results/v7"
NEG_REUSE_CAP = 3
LENGTH_RATIO = 2.0
N_SEEDS = 10


def _lr(X, y, seed=42):
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(X, y)


def _job_prior_auc(train, test):
    prior = train.groupby("job_id")["label"].mean()
    score = test["job_id"].map(prior).fillna(train["label"].mean())
    return roc_auc_score(test["label"], score)


def _job_length_auc(test):
    """Shortcut control: can job length alone separate positives from negatives?"""
    jl = test["job_text"].str.split().str.len()
    return roc_auc_score(test["label"], -jl)


def build_tier_pairs(seed, resumes, job_pool, hv, hm, cap=NEG_REUSE_CAP):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_i, te_i = next(gss.split(resumes, groups=resumes["resume_id"]))
    splits = {"train": resumes.iloc[tr_i].reset_index(drop=True),
              "test": resumes.iloc[te_i].reset_index(drop=True)}
    out = {}
    for tier in NEGATIVE_TIERS:
        parts = []
        for name, res in splits.items():
            base = 1000 * (1 if name == "train" else 2) + seed
            kw = dict(hard_vectorizer=hv, hard_job_matrix=hm) if tier == "hard_mined" else {}
            parts.append(build_pairs_v7(
                res, job_pool, name, np.random.RandomState(base + 500),   # negatives RNG
                tier, pos_rng=np.random.RandomState(base),                # positives RNG (shared)
                neg_reuse_cap=cap, length_match_ratio=LENGTH_RATIO,
                drop_contradictory=True, **kw))
        out[tier] = pd.concat(parts, ignore_index=True)
    return out


def run(n_seeds=N_SEEDS, cap=NEG_REUSE_CAP, out_dir=OUT, pairs_dir="data/eval_v7/negative_tiers_v7b"):
    os.makedirs(out_dir, exist_ok=True); os.makedirs(pairs_dir, exist_ok=True)
    resumes, job_pool = build_deduped_resumes(), build_deduped_job_pool()
    hv, hm = fit_hard_index(job_pool)
    rows, diag = [], []
    for seed in range(n_seeds):
        tiers = build_tier_pairs(seed, resumes, job_pool, hv, hm, cap)
        for tier, df in tiers.items():
            if seed == 0:
                df.to_csv(os.path.join(pairs_dir, f"pairs_{tier}_seed0.csv"), index=False)
            tr, te = df[df.split == "train"], df[df.split == "test"]
            # full model (original feature set) -- can exploit job identity
            Xtr, Xte, _, _, _ = tfidf_features(tr.resume_text, tr.job_text, te.resume_text, te.job_text)
            full = _lr(Xtr, tr.label); pf = full.predict(Xte); sf = full.predict_proba(Xte)[:, 1]
            # job-blind model: 1-d cosine only
            Ctr, Cte, ctr, cte, _ = tfidf_features(tr.resume_text, tr.job_text, te.resume_text, te.job_text,
                                                   feature_mode="cos_only")
            co = _lr(Ctr, tr.label); pc = co.predict(Cte); sc = co.predict_proba(Cte)[:, 1]
            neg_te = te[te.label == 0]
            rows.append(dict(
                seed=seed, tier=tier, n_test=len(te), n_train=len(tr),
                full_acc=accuracy_score(te.label, pf), full_bal_acc=balanced_accuracy_score(te.label, pf),
                full_auc=roc_auc_score(te.label, sf),
                cosLR_acc=accuracy_score(te.label, pc), cosLR_bal_acc=balanced_accuracy_score(te.label, pc),
                cosLR_auc=roc_auc_score(te.label, sc),
                zeroshot_cos_auc=roc_auc_score(te.label, cte),
                job_prior_auc=_job_prior_auc(tr, te), job_length_auc=_job_length_auc(te)))
            diag.append(dict(
                seed=seed, tier=tier,
                unique_neg_jobs_test=neg_te.job_id.nunique(), n_neg_test=len(neg_te),
                max_neg_reuse_test=int(neg_te.job_id.value_counts().max()),
                share_data_science_neg_test=float((neg_te.job_category == "Data Science").mean()),
                neg_same_text_as_positive=float(neg_te.job_text_key.isin(te[te.label == 1].job_text_key).mean())))
    per_seed = pd.DataFrame(rows); dg = pd.DataFrame(diag)
    per_seed.to_csv(f"{out_dir}/negative_tier_v7b_per_seed.csv", index=False)
    dg.groupby("tier").mean(numeric_only=True).drop(columns="seed").round(3).to_csv(
        f"{out_dir}/negative_tier_v7b_diagnostics.csv")

    metrics = ["full_acc", "full_bal_acc", "full_auc", "cosLR_acc", "cosLR_bal_acc", "cosLR_auc",
               "zeroshot_cos_auc", "job_prior_auc", "job_length_auc"]
    summ = []
    base = per_seed[per_seed.tier == "random_other_category"].set_index("seed")
    for tier in NEGATIVE_TIERS:
        g = per_seed[per_seed.tier == tier].set_index("seed")
        row = {"tier": tier, "n_seeds": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean(); row[f"{m}_sd"] = g[m].std(ddof=1)
            if tier != "random_other_category":
                d = (g[m] - base[m]).values
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = stats.t.ppf(0.975, len(d) - 1)
                row[f"{m}_diff_vs_random"] = d.mean()
                row[f"{m}_diff_lo"] = d.mean() - t * se; row[f"{m}_diff_hi"] = d.mean() + t * se
        summ.append(row)
    pd.DataFrame(summ).round(4).to_csv(f"{out_dir}/negative_tier_v7b_summary.csv", index=False)
    return per_seed, dg


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    ps, dg = run()
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    s = pd.read_csv(f"{OUT}/negative_tier_v7b_summary.csv")
    print(s[["tier", "full_acc_mean", "full_bal_acc_mean", "full_auc_mean", "cosLR_auc_mean",
             "zeroshot_cos_auc_mean", "job_prior_auc_mean", "job_length_auc_mean"]])
    print(pd.read_csv(f"{OUT}/negative_tier_v7b_diagnostics.csv"))
