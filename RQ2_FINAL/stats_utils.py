import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def bootstrap_ci(y_true, y_pred, metric_fn=accuracy_score, n_boot=2000, ci=95, seed=42):
    """NAIVE bootstrap: resamples individual PAIRS, not resumes. Kept for
    backward compatibility only. Per review S4.9 ('the bootstrap resamples
    pairs rather than resumes, which ignores the 3-pairs-per-resume
    clustering and understates variance'), use
    resume_clustered_bootstrap_ci() below for anything reported in the
    paper -- this naive version gives artificially narrow CIs."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    point = metric_fn(y_true, y_pred)
    return point, lower, upper


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test for paired predictions from two classifiers on the
    same test instances. Returns (statistic, p_value).
    Null hypothesis: both classifiers have the same error rate.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct_b_wrong = np.sum((pred_a == y_true) & (pred_b != y_true))
    a_wrong_b_correct = np.sum((pred_a != y_true) & (pred_b == y_true))
    both_correct = np.sum((pred_a == y_true) & (pred_b == y_true))
    both_wrong = np.sum((pred_a != y_true) & (pred_b != y_true))

    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    result = mcnemar(table, exact=(a_correct_b_wrong + a_wrong_b_correct) < 25, correction=True)
    return result.statistic, result.pvalue, table


def resume_clustered_bootstrap_ci(pairs_df, y_pred, metric_fn=accuracy_score,
                                   n_boot=10000, ci=95, seed=42):
    """Resume-level clustered bootstrap CI (review S4.9: 'The bootstrap
    resamples pairs rather than resumes, which ignores the 3-pairs-per-resume
    clustering and understates variance'). Resamples resume_id WITH
    replacement, keeping every pair belonging to a sampled resume together.
    pairs_df must have columns 'resume_id' and 'label'."""
    rng = np.random.RandomState(seed)
    resume_ids = pairs_df["resume_id"].unique()
    y_true = pairs_df["label"].values
    y_pred = np.asarray(y_pred)
    idx_by_resume = {rid: np.where(pairs_df["resume_id"].values == rid)[0] for rid in resume_ids}
    stats = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(resume_ids, size=len(resume_ids), replace=True)
        idx = np.concatenate([idx_by_resume[rid] for rid in sampled_ids])
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    point = metric_fn(y_true, y_pred)
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return point, lower, upper, len(stats)


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction for a family of p-values
    (review S4.9/S7: 'no multiple-comparison correction (RQ1-RQ3 involve
    dozens of comparisons)'). Returns (adjusted_pvalues, reject) in the
    ORIGINAL order of `pvalues`."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * pvalues[idx]
        running_max = max(running_max, adjusted)
        adj[idx] = min(running_max, 1.0)
    reject = adj < alpha
    return adj, reject


def cliffs_delta(a, b):
    """Cliff's delta effect size for two paired/unpaired samples of a
    per-instance metric (review S4.9: 'no effect sizes'). Returns delta in
    [-1, 1]; |delta|<0.147 negligible, <0.33 small, <0.474 medium, else large
    (Romano et al. 2006 thresholds)."""
    a, b = np.asarray(a), np.asarray(b)
    n_gt = sum(1 for x in a for y in b if x > y)
    n_lt = sum(1 for x in a for y in b if x < y)
    delta = (n_gt - n_lt) / (len(a) * len(b))
    return delta


def cohens_d(a, b):
    """Cohen's d for two independent samples (review S4.9: 'no effect
    sizes')."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def friedman_nemenyi(scores_by_representation: dict):
    """Friedman test + Nemenyi post-hoc across >=3 representations, scored
    on the SAME folds/seeds (review S4.9/S7, Demsar 2006 protocol).
    scores_by_representation: {representation_name: [score_per_fold, ...]}
    all lists must be the same length (same folds/seeds), aligned by index.
    Returns (friedman_stat, friedman_p, nemenyi_pvalue_matrix_df)."""
    import pandas as pd
    from scipy.stats import friedmanchisquare
    import scikit_posthocs as sp

    names = list(scores_by_representation.keys())
    arrays = [np.asarray(scores_by_representation[n]) for n in names]
    stat, p = friedmanchisquare(*arrays)

    data = np.array(arrays).T  # rows=folds, cols=representations
    nemenyi = sp.posthoc_nemenyi_friedman(data)
    nemenyi.index = names
    nemenyi.columns = names
    return stat, p, nemenyi


def repeated_grouped_eval(resumes, build_pairs_fn, job_pool, fit_predict_fn,
                           n_seeds=5, metric_fns=None):
    """Repeats grouped train/test splitting + fit/predict across n_seeds
    random seeds, returns a DataFrame of per-seed metrics plus mean±std.
    `fit_predict_fn(train_pairs, test_pairs) -> (y_true, y_pred)`

    ⚠️  PAIRED-TEST WARNING ⚠️
    Each call to this function re-splits resumes AND re-samples negative
    pairs using the global `random` state.  That means two representations
    evaluated via separate calls to this function will get DIFFERENT test
    pairs for the same seed index -- the seed-k score for TF-IDF and the
    seed-k score for SBERT come from different test sets.

    Consequence: the per-seed arrays returned by this function are NOT
    aligned across representations and MUST NOT be fed to a paired test
    (Wilcoxon, paired t-test, etc.).  The summary in the returned dict
    is labelled "mean+-std (NOT paired -- each seed uses different test
    pairs)" to make this explicit.

    To do a legitimate paired comparison across representations, use
    repeated_eval_frozen_pairs() with pre-generated fold files so every
    representation is evaluated on the identical pairs.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    if metric_fns is None:
        metric_fns = {"accuracy": accuracy_score, "f1_score": f1_score}

    rows = []
    for seed in range(n_seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, test_idx = next(gss.split(resumes, groups=resumes["resume_id"]))
        train_res = resumes.iloc[train_idx].reset_index(drop=True)
        test_res = resumes.iloc[test_idx].reset_index(drop=True)

        train_pairs = build_pairs_fn(train_res, job_pool, "train")
        test_pairs = build_pairs_fn(test_res, job_pool, "test")

        y_true, y_pred = fit_predict_fn(train_pairs, test_pairs)
        row = {"seed": seed, "paired": False}
        for name, fn in metric_fns.items():
            row[name] = fn(y_true, y_pred)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {"seed": "mean+-std (NOT paired -- each seed uses different test pairs)"}
    for name in metric_fns:
        summary[name] = f"{df[name].mean():.4f} +/- {df[name].std():.4f}"
    return df, summary


def repeated_eval_frozen_pairs(frozen_pairs_by_seed, fit_predict_fn, metric_fns=None):
    """Score every representation on the SAME saved (train, test) pairs per
    seed. Use this for model comparison. Do not interpret mean+/-std from
    this function as a confidence interval."""
    import pandas as pd

    if metric_fns is None:
        metric_fns = {"accuracy": accuracy_score, "f1_score": f1_score}

    rows = []
    for seed, (train_pairs, test_pairs) in frozen_pairs_by_seed.items():
        y_true, y_pred = fit_predict_fn(train_pairs, test_pairs)
        row = {"seed": seed, "n_test_pairs": len(test_pairs)}
        for name, fn in metric_fns.items():
            row[name] = fn(y_true, y_pred)
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {"seed": "mean+-std (not a CI)"}
    for name in metric_fns:
        summary[name] = f"{df[name].mean():.4f} +/- {df[name].std():.4f}"
    return df, summary


def paired_permutation_pvalue(scores_a, scores_b, n_perm=10000, seed=42):
    """Two-sided paired permutation test on per-unit scores (queries or
    resumes). Null: randomly flip the sign of each unit's difference.
    With n=10 units there are only 1024 distinct permutations; n_perm
    larger than 1024 still samples with replacement of those flips."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_permutation_pvalue requires aligned scores")
    diff = b - a
    obs = np.mean(diff)
    rng = np.random.RandomState(seed)
    extreme = 0
    n = len(diff)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(np.mean(signs * diff)) >= abs(obs) - 1e-15:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return float(obs), float(p)


def benjamini_hochberg(pvalues, alpha=0.05):
    """BH-FDR adjusted p-values and discoveries. Use alongside Holm when
    family-wise error with n=10 is known to be underpowered."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        val = min(prev, pvalues[idx] * n / rank)
        adj[idx] = val
        prev = val
    return adj, adj < alpha


def min_detectable_effect_paired(n, sd_of_paired_diff, alpha=0.05, power=0.80):
    """Approximate two-sided paired t-test MDE: (z_{1-a/2}+z_{power})*sd/sqrt(n).
    Returns nan if n < 2 or sd is not finite. This is a planning quantity,
    not a p-value."""
    from scipy.stats import norm
    if n < 2 or not np.isfinite(sd_of_paired_diff) or sd_of_paired_diff < 0:
        return np.nan
    z_a = norm.ppf(1 - alpha / 2)
    z_pow = norm.ppf(power)
    return (z_a + z_pow) * sd_of_paired_diff / np.sqrt(n)


def paired_t_power(delta, n, sd_of_paired_diff, alpha=0.05):
    """Achieved power of a two-sided paired t-test for a given mean difference."""
    from scipy.stats import norm
    if n < 2 or sd_of_paired_diff <= 0 or not np.isfinite(delta):
        return np.nan
    z_a = norm.ppf(1 - alpha / 2)
    z = abs(delta) / (sd_of_paired_diff / np.sqrt(n))
    return float(norm.cdf(z - z_a) + norm.cdf(-z - z_a))
