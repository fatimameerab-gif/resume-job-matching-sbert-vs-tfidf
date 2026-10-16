import re, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
OUT = "results/colab_v7b_finetune/"
RNG = np.random.default_rng(0)
p = pd.read_csv(OUT + "pair_predictions.csv")
p["model"] = p["model"].str.replace(r" \(weighted w_tfidf=[0-9.]+\)", " (weighted)", regex=True)
v3 = pd.read_csv("data/resume_job_pairs_v3.csv", usecols=["job_text", "job_source"]).drop_duplicates("job_text")
fs = []
for sd in range(5):
    x = pd.read_csv(f"data/eval_v7/frozen_seeds/pairs_seed{sd}.csv"); x = x[x.split == "test"]; x["seed"] = sd; fs.append(x)
pairs = pd.concat(fs).merge(v3, on="job_text", how="left")
BASES = ["TF-IDF head", "TF-IDF cosine (zero-shot)"]

# ---- per (seed, model, resume): reciprocal rank + hit@1 of the positive candidate
def rank_rows(g):
    pos = g[g.label == 1]
    if len(pos) != 1 or len(g) != 3: return None
    s = pos.score.iloc[0]; others = g[g.label == 0].score.values
    better = (others > s).sum(); ties = (others == s).sum()
    rank = 1 + better + 0.5 * ties            # ties counted pessimistically-average
    return pd.Series({"hit": float(better == 0 and ties == 0) + 0.5 * float(better == 0 and ties > 0),
                      "rr": 1.0 / rank})
r = (p.groupby(["seed", "model", "resume_id"]).apply(rank_rows, include_groups=False).dropna().reset_index())
r.to_csv(OUT + "ranking_per_resume.csv", index=False)

rows = []
for m, g in r.groupby("model"):
    rows.append({"model": m, "Hit@1": g.hit.mean(), "MRR": g.rr.mean(),
                 "n_resume_seed": len(g), "n_seeds": g.seed.nunique()})
rank_tab = pd.DataFrame(rows).sort_values("MRR", ascending=False)
rank_tab.to_csv(OUT + "ranking_summary.csv", index=False)

# ---- paired vs TF-IDF: cluster (resume) bootstrap on delta MRR, Wilcoxon, then Holm
def boot_delta(a, b, col, B=5000):
    j = a.merge(b, on=["seed", "resume_id"], suffixes=("_a", "_b"))
    j["d"] = j[col + "_a"] - j[col + "_b"]
    per = j.groupby("resume_id").d.mean()          # cluster = resume, averaged over seeds
    idx = RNG.integers(0, len(per), (B, len(per)))
    bs = per.values[idx].mean(1)
    return j, per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def holm(pv):
    pv = np.asarray(pv, float); o = np.argsort(pv); n = len(pv); adj = np.empty(n); run = 0
    for k, i in enumerate(o):
        run = max(run, (n - k) * pv[i]); adj[i] = min(1, run)
    return adj

res = []
for base in BASES:
    b = r[r.model == base]
    for m in sorted(r.model.unique()):
        if m in BASES: continue
        a = r[r.model == m]
        j, dm, lo, hi = boot_delta(a, b, "rr")
        _, dh, hlo, hhi = boot_delta(a, b, "hit")
        d = j.groupby("resume_id").d.mean().values
        try: pw = wilcoxon(d[d != 0]).pvalue if (d != 0).any() else 1.0
        except ValueError: pw = 1.0
        res.append({"baseline": base, "model": m, "dMRR": dm, "dMRR_lo": lo, "dMRR_hi": hi,
                    "dHit1": dh, "dHit1_lo": hlo, "dHit1_hi": hhi, "wilcoxon_p": pw})
res = pd.DataFrame(res)
res["holm_p"] = res.groupby("baseline").wilcoxon_p.transform(lambda s: holm(s.values))
res["sig_after_holm"] = res.holm_p < 0.05
res.to_csv(OUT + "ranking_paired_holm.csv", index=False)

# ---- Holm on the existing per-seed accuracy McNemar tests (pooled over seeds, caveat: seeds share resumes)
pv = pd.read_csv(OUT + "paired_vs_tfidf.csv")
pv["model"] = pv["model"].str.replace(r" \(weighted w_tfidf=[0-9.]+\)", " (weighted)", regex=True)
agg = pv.groupby("model").agg(c=("only_candidate_right", "sum"), t=("only_tfidf_right", "sum"),
                              diff_pp=("diff_pp", "mean")).reset_index()
agg["mcnemar_p_pooled"] = [binomtest(int(c), int(c + t), 0.5).pvalue if c + t else 1.0 for c, t in zip(agg.c, agg.t)]
agg["holm_p"] = holm(agg.mcnemar_p_pooled.values)
agg["sig_after_holm"] = agg.holm_p < 0.05
agg.to_csv(OUT + "paired_accuracy_holm.csv", index=False)

# ---- disagreement analysis (TF-IDF head vs each SBERT head), seed-level
def pivot(m):
    x = p[p.model == m][["seed", "resume_id", "job_id", "label", "negative_reason", "correct"]]
    return x.rename(columns={"correct": "c_" + m})
dis = []
for m in ["SBERT-MiniLM-L6-v2 frozen head", "SBERT-mpnet-base-v2 frozen head",
          "SBERT-MiniLM-L6-v2 fine-tuned head", "SBERT-mpnet-base-v2 fine-tuned head"]:
    j = pivot("TF-IDF head").merge(pivot(m)[["seed", "resume_id", "job_id", "c_" + m]], on=["seed", "resume_id", "job_id"])
    j = j.merge(pairs[["seed", "resume_id", "job_id", "job_source", "job_text", "resume_text", "resume_category", "job_category", "resume_family", "job_family"]].drop_duplicates(["seed","resume_id","job_id"]), on=["seed", "resume_id", "job_id"], how="left")
    j["resume_words"] = j.resume_text.fillna("").str.split().str.len()
    j["job_words"] = j.job_text.fillna("").str.split().str.len()
    d = j[j["c_TF-IDF head"] != j["c_" + m]].copy()
    d["who_right"] = np.where(d["c_" + m] == 1, "SBERT only", "TF-IDF only")
    d["model"] = m
    dis.append(d.drop(columns=["resume_text", "job_text", "c_" + m, "c_TF-IDF head"]))
dis = pd.concat(dis)
dis.to_csv(OUT + "disagreement_pairs.csv", index=False)
tax = (dis.assign(kind=np.where(dis.label == 1, "positive pair", dis.negative_reason.fillna("negative")))
          .groupby(["model", "kind", "who_right"]).size().unstack(fill_value=0).reset_index())
tax.to_csv(OUT + "disagreement_taxonomy.csv", index=False)
src = dis.groupby(["model", "job_source", "who_right"]).size().unstack(fill_value=0).reset_index()
src.to_csv(OUT + "disagreement_by_job_source.csv", index=False)
ln = dis.groupby(["model", "who_right"])[["resume_words", "job_words"]].median().reset_index()
ln.to_csv(OUT + "disagreement_lengths.csv", index=False)

conf = (dis[dis.label == 0].groupby(["model", "resume_category", "job_category", "who_right"]).size().unstack(fill_value=0).reset_index())
conf.to_csv(OUT + "disagreement_category_confusions.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
print("RANKING\n", rank_tab.round(3).to_string(index=False))
print("\nPAIRED RANKING vs TF-IDF head (Holm)\n", res[res.baseline == "TF-IDF head"].round(3).drop(columns="baseline").to_string(index=False))
print("\nACCURACY McNemar pooled (Holm)\n", agg.round(4).to_string(index=False))
print("\nDISAGREEMENT TAXONOMY\n", tax.to_string(index=False))
print("\nBY JOB SOURCE\n", src.to_string(index=False)); print("\nLENGTHS\n", ln.round(0).to_string(index=False))
