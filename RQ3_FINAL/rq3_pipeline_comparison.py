"""
RQ3 pipeline-level comparison (review 4.7: 'most importantly, an end-to-end fine-tuned bi-encoder and a cross-encoder').
Puts, on the SAME 5 frozen test sets (750 pair-seeds), side by side:
  - tuned classifier heads on frozen features (rq3 design A: best classifier per representation by INNER AUC, chosen without looking at test)
  - the RQ1 heads / hybrids / FINE-TUNED bi-encoders (results/colab_v7b_finetune/pair_predictions.csv, default 0.5 threshold, 3 epochs, 256 train tokens)
  - zero-shot cross-encoder (results/colab_v7b_extras/cross_encoder.csv, aggregated only; produced by colab_v7b_extras.py)
Metrics: AUC, Hit@1, MRR (threshold-free), accuracy at the model's own threshold, and a resume-clustered bootstrap CI of delta-MRR vs TF-IDF-LR head.
Run after rq3_classifier_grid_v3.py. Output: results/rq3/pipeline_comparison.csv, pipeline_vs_tfidf_head.csv
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
OUT = "results/rq3"; FT = "results/colab_v7b_finetune/pair_predictions.csv"; CE = "results/colab_v7b_extras/cross_encoder.csv"

def rr_hit(g):
    out = []
    for r, x in g.groupby("resume_id"):
        if x.label.sum() != 1: continue
        p = x.s[x.label == 1].iloc[0]; o = x.s[x.label == 0].values; k = 1 + (o > p).sum() + .5 * (o == p).sum(); out.append((r, float(k == 1), 1 / k))
    return pd.DataFrame(out, columns=["resume_id", "hit", "rr"])

def main():
    rows, per = [], {}
    ft = pd.read_csv(FT); ft["model"] = ft.model.str.replace(r" \(weighted w_tfidf=[0-9.]+\)", " (weighted)", regex=True)
    sets = {(m, "RQ1 run"): g.rename(columns={"score": "s"}) for m, g in ft.groupby("model")}
    thr = {m: 0.5 for m, _ in sets}
    A = pd.read_csv(f"{OUT}/predictions_A.csv") if os.path.exists(f"{OUT}/predictions_A.csv") else None
    S = pd.read_csv(f"{OUT}/grid_summary_A.csv") if A is not None else None
    if A is not None:
        for rep, g in S.groupby("representation"):
            # best classifier chosen by the mean INNER-CV AUC logged per split (never by test metrics)
            R = pd.read_csv(f"{OUT}/grid_per_split_A.csv"); best = R[R.representation == rep].groupby("classifier").inner_auc.mean().idxmax()
            sub = A[(A.representation == rep) & (A.classifier == best)].rename(columns={"split": "seed", "p": "s"})
            sets[(f"{rep} + tuned {best} (RQ3)", "RQ3 tuned head")] = sub
    for (m, fam), g in sets.items():
        g = g.copy(); rh = []
        for sd, x in g.groupby("seed"):
            r = rr_hit(x); r["seed"] = sd; rh.append(r)
        rh = pd.concat(rh); per[m] = rh.set_index(["seed", "resume_id"])
        rows.append(dict(model=m, family=fam, auc=np.mean([roc_auc_score(x.label, x.s) for _, x in g.groupby("seed")]), hit1=rh.hit.mean(), mrr=rh.rr.mean(), mrr_per_resume=rh.groupby('resume_id').rr.mean().mean(), hit1_per_resume=rh.groupby('resume_id').hit.mean().mean(),
                         acc_at_own_threshold=np.mean([accuracy_score(x.label, x.s >= 0.5) for _, x in g.groupby("seed")]) if fam == "RQ1 run" and "cosine" not in m and "RRF" not in m else np.nan))
    if os.path.exists(CE):
        c = pd.read_csv(CE); rows.append(dict(model=c.model.iloc[0] + " (zero-shot cross-encoder)", family="cross-encoder", auc=c.auc.mean(), hit1=c.hit1.mean(), mrr=c.mrr.mean(), mrr_per_resume=np.nan, hit1_per_resume=np.nan, acc_at_own_threshold=np.nan))
    T = pd.DataFrame(rows).sort_values("mrr", ascending=False); T.round(4).to_csv(f"{OUT}/pipeline_comparison.csv", index=False)
    base = per.get("TF-IDF head"); res = []
    if base is not None:
        for m, p in per.items():
            if m == "TF-IDF head": continue
            j = p.join(base, lsuffix="_a", rsuffix="_b", how="inner"); d = (j.rr_a - j.rr_b).groupby(level="resume_id").mean()
            bs = [d.values[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]
            res.append(dict(model=m, mrr_per_resume=p.rr.groupby(level="resume_id").mean().mean(), dMRR=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(res).round(4).to_csv(f"{OUT}/pipeline_vs_tfidf_head.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
