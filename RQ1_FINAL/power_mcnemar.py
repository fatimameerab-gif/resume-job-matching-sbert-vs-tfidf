import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

ALPHA, POWER = 0.05, 0.80
za, zb = norm.ppf(1 - ALPHA / 2), norm.ppf(POWER)

def n_needed(d, p):
    return (za * np.sqrt(p) + zb * np.sqrt(p - d ** 2)) ** 2 / d ** 2

def mde(n, p):
    return brentq(lambda d: n_needed(d, p) - n, 1e-4, np.sqrt(p) - 1e-6)

rows = []
for p in (0.10, 0.15, 0.20, 0.30):
    for d in (0.03, 0.05, 0.08, 0.10):
        if d ** 2 < p:
            rows.append(dict(kind="n_needed_pairs", discordant_rate=p, accuracy_gap=d, value=round(n_needed(d, p), 1)))
    for n in (75, 150, 300):
        rows.append(dict(kind=f"MDE_gap_at_n={n}", discordant_rate=p, accuracy_gap=np.nan, value=round(mde(n, p), 4)))
out = pd.DataFrame(rows)
out["assumptions"] = f"two-sided McNemar, alpha={ALPHA}, power={POWER}, normal approximation; pairs treated as independent (resume clustering would INCREASE the n needed)"
out.to_csv("results/v7/synthetic_mcnemar_power.csv", index=False)
print(out[(out.discordant_rate == 0.20)].drop(columns="assumptions").to_string(index=False))
