"""Review 3.7: paired, resume-clustered comparison of complete pipelines (MRR over the 3 candidates of each test resume).
The claim 'choosing the pipeline matters more than choosing the classifier' is tested directly: differences of per-resume reciprocal rank,
averaged over the five seeds for each unique test resume, percentile bootstrap over resumes (10,000), sign-flip permutation p, Holm.
Outputs results/pipeline_paired/pipeline_paired.csv.  Run: python rq3_pipeline_paired.py (CPU)"""
import os, numpy as np, pandas as pd
OUT = "results/pipeline_paired"; os.makedirs(OUT, exist_ok=True)
P1 = pd.concat([pd.read_csv(p) for p in ["results/colab_v7b_finetune/pair_predictions.csv", "results/colab_v7b_newenc/finetune/pair_predictions.csv"] if os.path.exists(p)]).drop_duplicates(["seed", "model", "resume_id", "job_id"])
A = pd.read_csv("../RQ3_FINAL/results/rq3/predictions_A.csv")
gb = A[(A.representation == "TF-IDF") & (A.classifier.str.contains("Gradient|GB", case=False))].rename(columns={"split": "seed", "p": "score"}); gb["model"] = "TF-IDF + tuned gradient boosting"
print("classifiers in RQ3 file:", sorted(A.classifier.unique()), "| rows GB:", len(gb))
sc = [c for c in P1.columns if c in ("score", "p", "prob")][0]
cols = ["seed", "model", "resume_id", "job_id", "label", sc]
D = pd.concat([P1[cols].rename(columns={sc: "score"}), gb[["seed", "model", "resume_id", "job_id", "label", "score"]]])
def rr(g):
    pos = g.score[g.label == 1]; neg = g.score[g.label == 0].values
    if len(pos) != 1: return np.nan
    p = pos.iloc[0]; return 1 / (1 + (neg > p).sum() + 0.5 * (neg == p).sum())
R = D.groupby(["model", "seed", "resume_id"]).apply(rr).rename("rr").reset_index().dropna()
M = R.groupby(["model", "resume_id"]).rr.mean().unstack(0)
pairs = [("SBERT-mpnet-base-v2 fine-tuned head", "TF-IDF + tuned gradient boosting"), ("SBERT-MiniLM-L6-v2 fine-tuned head", "TF-IDF + tuned gradient boosting"),
         ("TF-IDF + tuned gradient boosting", "TF-IDF head"), ("TF-IDF + tuned gradient boosting", "TF-IDF cosine (zero-shot)"), ("TF-IDF cosine (zero-shot)", "TF-IDF head"),
         ("SBERT-mpnet-base-v2 fine-tuned head", "TF-IDF cosine (zero-shot)"), ("SBERT-mpnet-base-v2 fine-tuned head", "TF-IDF head"), ("SBERT-MiniLM-L6-v2 fine-tuned head", "TF-IDF head"),
         ("SBERT-mpnet-base-v2 fine-tuned head", "SBERT-MiniLM-L6-v2 fine-tuned head"),
         ("JobBERT-v2 fine-tuned head", "TF-IDF + tuned gradient boosting"), ("JobBERT-v2 fine-tuned head", "TF-IDF head"), ("JobBERT-v2 fine-tuned head", "SBERT-mpnet-base-v2 fine-tuned head")]
rng = np.random.default_rng(0); rows = []
for a, b in pairs:
    if a not in M or b not in M: print("missing", a, b); continue
    d = (M[a] - M[b]).dropna().values; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    sf = np.array([(d * rng.choice([-1, 1], n)).mean() for _ in range(10000)]); p = (np.sum(np.abs(sf) >= abs(d.mean())) + 1) / 10001
    rows.append(dict(a=a, b=b, n_resumes=n, mrr_a=M[a].mean(), mrr_b=M[b].mean(), diff=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), p=p))
T = pd.DataFrame(rows); o = np.argsort(T.p.values); m = len(T); h = np.zeros(m); run = 0
for rank, i in enumerate(o): run = max(run, (m - rank) * T.p.values[i]); h[i] = min(1, run)
T["p_holm"] = h; T.round(4).to_csv(f"{OUT}/pipeline_paired.csv", index=False); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
