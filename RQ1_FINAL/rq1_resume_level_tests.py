"""Review 3.4 / 1.1: resume-level (cluster-aware) paired comparisons, with the category-classifier baseline included.

Why: three pairs share one resume and the five seeds reuse the same 166 resumes, so a McNemar test over pairs overstates the sample.
Here every resume gets ONE number per model (its mean correctness over its three test pairs, averaged over the seeds in which it is a test resume),
the paired difference is taken per resume, and we report: the number of unique test resumes, the mean difference, a percentile bootstrap CI over
resumes, and an exact two-sided sign-flip permutation test p-value on the per-resume differences (Holm-corrected over the family of comparisons).
Inputs : results/colab_v7b_finetune/pair_predictions.csv, results/colab_v7b_newenc/{finetune,hybrid}/pair_predictions.csv, results/category_baseline/pair_predictions.csv
Output : results/resume_level/resume_level_pairwise.csv, per_model_resume_level.csv, NOTE.txt
Run    : python rq1_resume_level_tests.py (after rq1_category_baseline.py)
"""
import os, re
import numpy as np, pandas as pd
OUT = "results/resume_level"
FT, NEW_FT, NEW_H, CAT = "results/colab_v7b_finetune/pair_predictions.csv", "results/colab_v7b_newenc/finetune/pair_predictions.csv", "results/colab_v7b_newenc/hybrid/pair_predictions.csv", "results/category_baseline/pair_predictions.csv"
REF = "TF-IDF head"
CMP = [("SBERT-mpnet-base-v2 fine-tuned head", REF), ("SBERT-MiniLM-L6-v2 fine-tuned head", REF), ("JobBERT-v2 fine-tuned head", REF),
       ("JobBERT-v2 frozen head", REF), ("SBERT-mpnet-base-v2 frozen head", REF), ("SBERT-MiniLM-L6-v2 frozen head", REF), ("TF-IDF cosine (zero-shot)", REF),
       ("Category classifiers (SBERT-mpnet-base-v2)", REF), ("Category classifiers (JobBERT-v2)", REF), ("Category classifiers (TF-IDF)", REF),
       ("SBERT-mpnet-base-v2 fine-tuned head", "Category classifiers (SBERT-mpnet-base-v2)"), ("SBERT-mpnet-base-v2 fine-tuned head", "Category classifiers (TF-IDF)"),
       ("JobBERT-v2 fine-tuned head", "Category classifiers (JobBERT-v2)"), ("SBERT-mpnet-base-v2 fine-tuned head", "Category classifiers (JobBERT-v2)"),
       ("JobBERT-v2 fine-tuned head", "SBERT-mpnet-base-v2 fine-tuned head")]


def load():
    fr = []
    for p in (FT, NEW_FT, NEW_H, CAT):
        if os.path.exists(p):
            d = pd.read_csv(p)
            if p == NEW_H: d["model"] = d.model + " [run B]"          # same configurations as NEW_FT but a separate run: kept apart to show the run-to-run spread (review 3.3)
            d["model"] = d.model.str.replace(r" \(weighted w_tfidf=[0-9.]+\)", " (weighted)", regex=True); fr.append(d[["seed", "model", "resume_id", "job_id", "correct"]])
    d = pd.concat(fr).drop_duplicates(["seed", "model", "resume_id", "job_id"], keep="first")      # TF-IDF head etc. appear in several files: keep the first (they are identical runs)
    return d


def per_resume(d, model):
    x = d[d.model == model]
    return x.groupby(["seed", "resume_id"]).correct.mean().groupby("resume_id").mean()          # per seed-resume -> mean over seeds -> one value per resume


def boot_and_perm(diff, nboot=5000, nperm=20000, seed=0):
    rng = np.random.default_rng(seed); n = len(diff); v = diff.values
    bs = np.array([v[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    obs = abs(v.mean()); signs = rng.choice([-1.0, 1.0], size=(nperm, n)); perm = np.abs((signs * v).mean(1))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((np.sum(perm >= obs - 1e-12) + 1) / (nperm + 1))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    os.makedirs(OUT, exist_ok=True); d = load(); rows = []
    for a, b in CMP:
        if a not in set(d.model) or b not in set(d.model): print("missing", a, "|", b); continue
        ra, rb = per_resume(d, a), per_resume(d, b); j = pd.concat([ra, rb], axis=1, keys=["a", "b"]).dropna(); diff = j.a - j.b
        lo, hi, p = boot_and_perm(diff)
        rows.append(dict(model_a=a, model_b=b, n_unique_test_resumes=len(diff), acc_a=j.a.mean(), acc_b=j.b.mean(), diff=diff.mean(), ci_lo=lo, ci_hi=hi, p_signflip=p,
                         resumes_a_better=int((diff > 0).sum()), resumes_b_better=int((diff < 0).sum()), resumes_tied=int((diff == 0).sum())))
    R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_signflip.values); R.round(4).to_csv(f"{OUT}/resume_level_pairwise.csv", index=False)
    pm = pd.DataFrame([dict(model=m, n_unique_test_resumes=per_resume(d, m).size, resume_level_acc=per_resume(d, m).mean()) for m in sorted(d.model.unique()) if "RRF" not in m]).round(4)
    pm.to_csv(f"{OUT}/per_model_resume_level.csv", index=False)
    pd.set_option("display.width", 250); print(R.round(3).drop(columns=["resumes_tied"]).to_string(index=False))
    open(f"{OUT}/NOTE.txt", "w").write("One number per resume (mean correctness over its 3 test pairs and over the seeds where it is a test resume). CI = percentile bootstrap over resumes; p = exact-style sign-flip permutation test on per-resume differences (20,000 flips); Holm over the listed family. Seeds overlap, so these intervals are still optimistic about generalisation to NEW resumes beyond the 166 available.\n")


if __name__ == "__main__":
    main()
