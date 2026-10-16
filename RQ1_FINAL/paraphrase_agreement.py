import itertools, os, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
IN = "data_private_ratings"; OUT = "results/paraphrases/human_check"; os.makedirs(OUT, exist_ok=True)
S = {k: pd.read_excel(f"{IN}/{f}").sort_values("id").reset_index(drop=True)
     for k, f in [("A", "rater_A.xlsx"), ("B", "rater_B.xlsx"), ("C", "rater_C.xlsx"), ("LLM", "judge_LLM.xlsx")]}
for k, d in S.items():
    assert len(d) == 60 and (d.id.values == S["A"].id.values).all() and (d.paraphrase.values == S["A"].paraphrase.values).all(), k
role = S["A"].role.values
long = pd.concat([d.assign(rater=k)[["rater", "id", "role", "meaning_preserved", "info_added", "info_lost", "fluent"]] for k, d in S.items()])
long.to_csv(f"{OUT}/ratings_long.csv", index=False)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - h) / d), min(1, (c + h) / d)
rows = []
for k, d in S.items():
    for sub, m in [("all", np.ones(60, bool)), ("resume", role == "resume"), ("job", role == "job")]:
        n = int(m.sum()); x = d.meaning_preserved.values[m]; lo, hi = wilson(int(x.sum()), n)
        rows.append(dict(rater=k, subset=sub, n=n, meaning_preserved=x.mean(), mp_ci_lo=lo, mp_ci_hi=hi,
                         info_added=d.info_added.values[m].mean(), info_lost=d.info_lost.values[m].mean(), fluent_mean=d.fluent.values[m].mean()))
summ = pd.DataFrame(rows); summ.to_csv(f"{OUT}/summary.csv", index=False)
K = []
for a, b in itertools.combinations(S, 2):
    for col in ["meaning_preserved", "info_added", "info_lost"]:
        x, y = S[a][col].values, S[b][col].values
        K.append(dict(rater_1=a, rater_2=b, item=col, agreement=(x == y).mean(), kappa=cohen_kappa_score(x, y)))
    K.append(dict(rater_1=a, rater_2=b, item="fluent (quadratic-weighted kappa)", agreement=(S[a].fluent.values == S[b].fluent.values).mean(),
                  kappa=cohen_kappa_score(S[a].fluent.values, S[b].fluent.values, weights="quadratic")))
K = pd.DataFrame(K); K.to_csv(f"{OUT}/kappa.csv", index=False)
def fleiss(M):  # M: items x raters (binary)
    n = M.shape[1]; c1 = M.sum(1); P = ((c1 * (c1 - 1)) + ((n - c1) * (n - c1 - 1))) / (n * (n - 1)); pj = M.mean(); Pe = pj ** 2 + (1 - pj) ** 2
    return (P.mean() - Pe) / (1 - Pe)
mp = {k: S[k].meaning_preserved.values for k in S}
extra = dict(fleiss_ABC=fleiss(np.stack([mp["A"], mp["B"], mp["C"]], 1)),
             majority_ABC=float((mp["A"] + mp["B"] + mp["C"] >= 2).mean()), unanimous_ABC=float((mp["A"] + mp["B"] + mp["C"] == 3).mean()),
             majority_BC_all_agree_preserved=float(((mp["B"] + mp["C"]) == 2).mean()), BC_agreement=float((mp["B"] == mp["C"]).mean()),
             BC_kappa=cohen_kappa_score(mp["B"], mp["C"]), BC_both_preserved=int(((mp["B"] + mp["C"]) == 2).sum()),
             BC_either_preserved=float(((mp["B"] + mp["C"]) >= 1).mean()))
pd.Series(extra).to_csv(f"{OUT}/extra.csv", header=["value"])
pd.set_option("display.width", 200); print(summ.round(3).to_string()); print(K[K.item == "meaning_preserved"].round(3).to_string()); print(pd.Series(extra).round(3))
