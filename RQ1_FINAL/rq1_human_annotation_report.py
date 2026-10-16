"""
Human annotation report (240 resume-job pairs) + lexicon validation (131 items).

Reads PRIVATE inputs (annotator texts/notes, KEY) and writes ANONYMISED outputs
(no resume/job text, no free-text notes) to results/human_annotation/.

Usage (from the RQ1 folder):
  python rq1_human_annotation_report.py --kit annotation_kit --lex lex_merged.csv
Inputs:
  <kit>/annotation-sheet.csv, <kit>/annotation-KEY-do-not-share-with-annotators.csv
  --lex : merged lexicon ratings (item_id,direction,original,replacement,
          annotator1_meaning_preserved_0_1,notes_1,annotator2_meaning_preserved_0_1,notes_2)
  data/resume_job_pairs_v3.csv and results/embeddings_pinned/*.npz  (for model-vs-human)
Outputs (results/human_annotation/):
  item_scores.csv, annotation_agreement.csv, weak_label_audit.csv,
  lexicon_items.csv, lexicon_agreement.csv, model_vs_human.csv
"""
import argparse, hashlib, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

A1, A2 = "annotator1_score_0_1_2", "annotator2_score_0_1_2"
OUT = "results/human_annotation"
ENC = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "e5-base-v2",
       "bge-base-en-v1.5", "gte-base", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def boot(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        s = rng.integers(0, len(x), len(x))
        try:
            r = fn(x[s], y[s])
            if not np.isnan(r): v.append(r)
        except ValueError:
            pass
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="annotation_kit")
    ap.add_argument("--lex", default="data_private_ratings/lexicon/lexicon_merged_private.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    s = pd.read_csv(f"{a.kit}/annotation-sheet.csv")
    k = pd.read_csv(f"{a.kit}/annotation-KEY-do-not-share-with-annotators.csv")
    d = s[["item_id", A1, A2]].merge(k, on="item_id")
    d["human_mean"] = (d[A1] + d[A2]) / 2
    d["human_fit_binary"] = (d.human_mean >= 1).astype(int)
    d.rename(columns={A1: "rater1_score", A2: "rater2_score"}, inplace=True)
    keep = ["item_id", "resume_id", "job_id", "resume_category", "job_category", "label",
            "split", "job_source", "rater1_score", "rater2_score", "human_mean", "human_fit_binary"]
    d[keep].to_csv(f"{OUT}/item_scores.csv", index=False)

    # --- agreement
    x, y = d.rater1_score.values, d.rater2_score.values
    rows = []
    def add(name, fn, xx, yy):
        lo, hi = boot(xx, yy, fn); rows.append((name, round(fn(xx, yy), 4), round(lo, 4), round(hi, 4)))
    add("kappa_unweighted_0_1_2", cohen_kappa_score, x, y)
    add("kappa_quadratic_weighted_0_1_2", lambda p, q: cohen_kappa_score(p, q, weights="quadratic"), x, y)
    add("kappa_binary_fit_vs_nofit(>=1)", lambda p, q: cohen_kappa_score(p >= 1, q >= 1), x, y)
    add("kappa_binary_good_fit(==2)", lambda p, q: cohen_kappa_score(p == 2, q == 2), x, y)
    ag = pd.DataFrame(rows, columns=["metric", "value", "ci95_low", "ci95_high"])
    ag.loc[len(ag)] = ["exact_agreement", round((x == y).mean(), 4), np.nan, np.nan]
    ag.loc[len(ag)] = ["n_items", len(d), np.nan, np.nan]
    ag.loc[len(ag)] = ["n_disagreements", int((x != y).sum()), np.nan, np.nan]
    ag.loc[len(ag)] = ["n_disagreements_by_2", int((abs(x - y) == 2).sum()), np.nan, np.nan]
    ag.to_csv(f"{OUT}/annotation_agreement.csv", index=False)

    # --- weak-label audit
    w, h = d.label.values, d.human_fit_binary.values
    au = pd.DataFrame([dict(n=len(d), n_weak_pos=int(w.sum()), n_human_fit=int(h.sum()),
        agreement=round((w == h).mean(), 4), weak_pos_rated_nofit=int(((w == 1) & (h == 0)).sum()),
        weak_neg_rated_fit=int(((w == 0) & (h == 1)).sum()),
        kappa_weak_vs_human=round(cohen_kappa_score(w, h), 4),
        n_human_mean_ge_1_5=int((d.human_mean >= 1.5).sum()))])
    au.to_csv(f"{OUT}/weak_label_audit.csv", index=False)

    # --- lexicon validation
    L = pd.read_csv(a.lex)
    c1, c2 = "annotator1_meaning_preserved_0_1", "annotator2_meaning_preserved_0_1"
    L[[c1, c2]] = L[[c1, c2]].astype(int)
    L["both_preserved"] = ((L[c1] == 1) & (L[c2] == 1)).astype(int)
    L["either_flagged"] = ((L[c1] == 0) | (L[c2] == 0)).astype(int)
    # entries of the lexicon used in the main robustness test (CORE) vs the extra "risky" entries
    from robustness import CORE_PAIRS
    core = {w.lower() for pr in CORE_PAIRS for w in pr}
    L["in_main_lexicon"] = (L.original.str.lower().isin(core) | L.replacement.str.lower().isin(core)).astype(int)
    L[["item_id", "direction", "original", "replacement", "in_main_lexicon", c1, c2, "both_preserved"]].rename(
        columns={c1: "rater1_preserved", c2: "rater2_preserved"}).to_csv(f"{OUT}/lexicon_items.csv", index=False)
    p, q = L[c1].values, L[c2].values
    po = (p == q).mean(); pabak = 2 * po - 1
    lo, hi = boot(p, q, lambda u, v: cohen_kappa_score(u, v))
    la = pd.DataFrame([dict(n_items=len(L), raw_agreement=round(po, 4), cohen_kappa=round(cohen_kappa_score(p, q), 4),
        kappa_ci_low=round(lo, 4), kappa_ci_high=round(hi, 4), pabak=round(pabak, 4),
        preserved_rater1=round(p.mean(), 4), preserved_rater2=round(q.mean(), 4),
        both_preserved=int(L.both_preserved.sum()), either_flagged=int(L.either_flagged.sum()),
        both_flagged=int(((p == 0) & (q == 0)).sum()),
        n_main_lexicon=int(L.in_main_lexicon.sum()), main_both_preserved=int(L[L.in_main_lexicon == 1].both_preserved.sum()),
        n_extra_entries=int((L.in_main_lexicon == 0).sum()), extra_both_preserved=int(L[L.in_main_lexicon == 0].both_preserved.sum()))])
    la.to_csv(f"{OUT}/lexicon_agreement.csv", index=False)

    # --- model vs human (zero-shot cosine, no training; resume-level cluster bootstrap)
    P = pd.read_csv("data/resume_job_pairs_v3.csv")
    m = d.merge(P[["resume_id", "job_id", "resume_text", "job_text"]], on=["resume_id", "job_id"], how="left")
    assert m.resume_text.notna().all() and len(m) == len(d)
    R = P.drop_duplicates("resume_id"); J = P.drop_duplicates("job_id")
    v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(
        pd.concat([R.resume_text, J.job_text]))
    m["TF-IDF"] = np.asarray(v.transform(m.resume_text).multiply(v.transform(m.job_text)).sum(1)).ravel()
    for e in ENC:
        z = np.load(f"results/embeddings_pinned/{e}.npz", allow_pickle=True)
        E = dict(zip(z["keys"], z["vectors"]))
        m[e] = [float(E[sha(r)] @ E[sha(j)]) for r, j in zip(m.resume_text, m.job_text)]
    mods = ["TF-IDF"] + ENC
    rid = m.resume_id.values; urid = np.unique(rid); grp = {r: np.where(rid == r)[0] for r in urid}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([grp[r] for r in rng.choice(urid, len(urid))]) for _ in range(2000)]
    def auc(col, yv, idx=None):
        idx = np.arange(len(m)) if idx is None else idx
        yy = yv[idx]
        return roc_auc_score(yy, m[col].values[idx]) if 0 < yy.sum() < len(yy) else np.nan
    yh, yw = m.human_fit_binary.values, m.label.values
    out = []
    for c in mods:
        bh = np.array([auc(c, yh, i) for i in boots]); bw = np.array([auc(c, yw, i) for i in boots])
        r = dict(model=c, auc_vs_human=auc(c, yh), human_lo=np.nanpercentile(bh, 2.5), human_hi=np.nanpercentile(bh, 97.5),
                 auc_vs_weak=auc(c, yw), weak_lo=np.nanpercentile(bw, 2.5), weak_hi=np.nanpercentile(bw, 97.5),
                 spearman_vs_human_mean=spearmanr(m[c], m.human_mean).correlation)
        if c != "TF-IDF":
            dd = np.array([auc(c, yh, i) - auc("TF-IDF", yh, i) for i in boots])
            r.update(diff_vs_tfidf=auc(c, yh) - auc("TF-IDF", yh), diff_lo=np.nanpercentile(dd, 2.5), diff_hi=np.nanpercentile(dd, 97.5))
        out.append(r)
    pd.DataFrame(out).round(4).to_csv(f"{OUT}/model_vs_human.csv", index=False)
    print("n pairs", len(m), "n resumes", len(urid))
    for f in sorted(os.listdir(OUT)): print(" wrote", f)


if __name__ == "__main__":
    main()
