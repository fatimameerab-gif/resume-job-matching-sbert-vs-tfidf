import numpy as np
from stats_extra import nadeau_bengio_t, bayesian_correlated_t

def test_nb_wider_than_naive_t():
    from scipy import stats
    d = np.array([0.02, 0.03, 0.01, 0.025, 0.015, 0.03, 0.02, 0.01, 0.02, 0.03])
    naive = stats.ttest_1samp(d, 0).pvalue
    assert nadeau_bengio_t(d, 700, 300)["p"] > naive          # correction must make the test more conservative

def test_bayes_probabilities_sum_to_one_and_follow_sign():
    d = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06])
    r = bayesian_correlated_t(d, 700, 300, rope=0.01)
    assert abs(r["p_a_better"] + r["p_rope"] + r["p_b_better"] - 1) < 1e-9 and r["p_a_better"] > 0.9
    z = bayesian_correlated_t(np.array([0.001, -0.002, 0.0015, -0.001, 0.0005, 0.0]), 700, 300)
    assert z["p_rope"] > 0.9
