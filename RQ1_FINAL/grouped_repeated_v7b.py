import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from eval_protocol_v7 import grouped_split_resume_and_job_text, build_pairs_v7
from representations import REPRESENTATIONS

OUT = "results/v7"
N_SEEDS = 20
REPS = [n for n in REPRESENTATIONS if not n.startswith("Static")]


def _fit_predict(train, test, fn):
    o = fn(train.resume_text, train.job_text, test.resume_text, test.job_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(o[0], train.label)
    return clf.predict(o[1]), clf.predict_proba(o[1])[:, 1]


def _auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def run(n_seeds=N_SEEDS, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    resumes, job_pool = build_deduped_resumes(), build_deduped_job_pool()
    rows, preds, cov = [], [], []
    for seed in range(n_seeds):
        tr_r, va_r, te_r, tr_j, va_j, te_j = grouped_split_resume_and_job_text(resumes, job_pool, seed=seed)
        tr = build_pairs_v7(tr_r, tr_j, "train", np.random.RandomState(100 + seed))
        te = build_pairs_v7(te_r, te_j, "test", np.random.RandomState(200 + seed))
        cov.append(dict(seed=seed, n_test_resumes=len(te_r), n_test_resumes_with_pairs=te.resume_id.nunique(),
                        n_train_pairs=len(tr), n_test_pairs=len(te),
                        test_pos_rate=te.label.mean() if len(te) else np.nan))
        if len(te) < 10 or te.label.nunique() < 2 or tr.label.nunique() < 2:
            continue
        assert not (set(tr.job_text_key) & set(te.job_text_key)), "job text leaked"
        assert not (set(tr.resume_id) & set(te.resume_id)), "resume leaked"
        for name in REPS:
            pred, score = _fit_predict(tr, te, REPRESENTATIONS[name])
            rows.append(dict(seed=seed, representation=name, n_test=len(te),
                             acc=accuracy_score(te.label, pred), bal_acc=balanced_accuracy_score(te.label, pred),
                             auc=_auc(te.label, score)))
            preds.append(pd.DataFrame(dict(seed=seed, representation=name, resume_id=te.resume_id.values,
                                           job_id=te.job_id.values, label=te.label.values, pred=pred, score=score)))
    per_seed = pd.DataFrame(rows); pd.DataFrame(cov).to_csv(f"{out_dir}/grouped_repeated_coverage.csv", index=False)
    per_seed.to_csv(f"{out_dir}/grouped_repeated_per_seed.csv", index=False)
    P = pd.concat(preds, ignore_index=True); P.to_csv(f"{out_dir}/grouped_repeated_predictions.csv", index=False)

    rng = np.random.default_rng(0); out = []
    for name, g in per_seed.groupby("representation"):
        n = len(g); t = stats.t.ppf(0.975, n - 1)
        row = dict(representation=name, n_seeds=n, mean_test_pairs_per_seed=g.n_test.mean(),
                   total_test_pair_evals=int(g.n_test.sum()))
        for m in ("acc", "bal_acc", "auc"):
            row[f"{m}_mean"] = g[m].mean(); row[f"{m}_sd"] = g[m].std(ddof=1)
            row[f"{m}_ci_lo"] = g[m].mean() - t * g[m].std(ddof=1) / np.sqrt(n)
            row[f"{m}_ci_hi"] = g[m].mean() + t * g[m].std(ddof=1) / np.sqrt(n)
        # pooled resume-clustered bootstrap for accuracy (resample resumes, keep all their pairs)
        d = P[P.representation == name]
        by = {r: (d[d.resume_id == r].pred.values == d[d.resume_id == r].label.values) for r in d.resume_id.unique()}
        ids = np.array(list(by)); boots = []
        for _ in range(2000):
            samp = rng.choice(ids, len(ids), replace=True)
            boots.append(np.concatenate([by[r] for r in samp]).mean())
        row["acc_pooled"] = np.concatenate(list(by.values())).mean()
        row["acc_pooled_cluster_ci_lo"], row["acc_pooled_cluster_ci_hi"] = np.percentile(boots, [2.5, 97.5])
        out.append(row)
    summ = pd.DataFrame(out); summ.round(4).to_csv(f"{out_dir}/grouped_repeated_summary.csv", index=False)
    return summ, pd.DataFrame(cov)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    summ, cov = run()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(summ[["representation", "n_seeds", "mean_test_pairs_per_seed", "acc_mean", "acc_sd", "bal_acc_mean",
                "auc_mean", "acc_pooled", "acc_pooled_cluster_ci_lo", "acc_pooled_cluster_ci_hi"]].round(3).to_string(index=False))
    print("\ncoverage: test resumes kept with >=1 pair:",
          f"{cov.n_test_resumes_with_pairs.sum()}/{cov.n_test_resumes.sum()} "
          f"({100*cov.n_test_resumes_with_pairs.sum()/cov.n_test_resumes.sum():.0f}%)",
          "| mean test pairs/seed:", round(cov.n_test_pairs.mean(), 1))
