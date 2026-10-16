"""Extra paired tests for repeated resampling on ONE dataset (review section 7).
 * nadeau_bengio_t: corrected resampled t-test (Nadeau & Bengio 2003); variance inflated by (1/J + n_test/n_train).
 * bayesian_correlated_t: Bayesian correlated t-test (Corani & Benavoli 2015; Benavoli et al. 2017) with a ROPE;
   returns P(A better), P(practically equal), P(B better). Same correlation heuristic rho = n_test/(n_train+n_test).
NOTE: Benavoli's Bayesian SIGNED-RANK test is for comparisons over several datasets; we have one dataset family,
so the correlated t-test is the right Bayesian tool here (a signed-rank test over 'datasets' would pseudo-replicate)."""
import numpy as np
from scipy import stats

def _rho(n_train, n_test): return n_test / (n_train + n_test)

def nadeau_bengio_t(diffs, n_train, n_test):
    d = np.asarray(diffs, float); J = len(d); m, v = d.mean(), d.var(ddof=1)
    if v == 0: return dict(mean_diff=m, t=np.nan, p=1.0 if m == 0 else 0.0, df=J - 1)
    t = m / np.sqrt((1.0 / J + n_test / n_train) * v)
    return dict(mean_diff=m, t=t, p=2 * stats.t.sf(abs(t), J - 1), df=J - 1)

def bayesian_correlated_t(diffs, n_train, n_test, rope=0.01):
    d = np.asarray(diffs, float); J = len(d); m, v = d.mean(), d.var(ddof=1); rho = _rho(n_train, n_test)
    if v == 0:
        return dict(mean_diff=m, p_a_better=float(m > rope), p_rope=float(abs(m) <= rope), p_b_better=float(m < -rope))
    scale = np.sqrt((1.0 / J + rho / (1 - rho)) * v)
    dist = stats.t(J - 1, loc=m, scale=scale)
    return dict(mean_diff=m, p_b_better=dist.cdf(-rope), p_rope=dist.cdf(rope) - dist.cdf(-rope), p_a_better=dist.sf(rope))
