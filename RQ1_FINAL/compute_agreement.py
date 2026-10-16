import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

A1, A2 = "annotator1_score_0_1_2", "annotator2_score_0_1_2"
KEY = "annotation_kit/annotation-KEY-do-not-share-with-annotators.csv"


def _boot_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x)); vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        try:
            v = fn(x[s], y[s])
            if not np.isnan(v):
                vals.append(v)
        except ValueError:
            pass
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--pilot", type=int, default=None, help="only first N items (pilot round)")
    a = ap.parse_args()

    df = pd.read_csv(a.sheet)
    if a.pilot:
        df = df.head(a.pilot)
    n_all = len(df)
    df[A1] = pd.to_numeric(df[A1], errors="coerce")
    df[A2] = pd.to_numeric(df[A2], errors="coerce")
    ok = df.dropna(subset=[A1, A2]).copy()
    bad = ok[~ok[A1].isin([0, 1, 2]) | ~ok[A2].isin([0, 1, 2])]
    if len(bad):
        sys.exit(f"Scores must be 0/1/2. Bad item_ids: {bad['item_id'].tolist()[:10]}")
    if len(ok) < 2:
        sys.exit(f"Only {len(ok)} fully annotated rows of {n_all}; annotate first.")
    x, y = ok[A1].astype(int).values, ok[A2].astype(int).values

    rows = []
    def add(name, fn, xx, yy):
        k = fn(xx, yy); lo, hi = _boot_ci(xx, yy, fn)
        rows.append(dict(metric=name, value=round(k, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4)))
    add("kappa_quadratic_weighted_0_1_2", lambda p, q: cohen_kappa_score(p, q, weights="quadratic"), x, y)
    add("kappa_unweighted_0_1_2", lambda p, q: cohen_kappa_score(p, q), x, y)
    add("kappa_binary_fit_vs_nofit(>=1)", lambda p, q: cohen_kappa_score(p >= 1, q >= 1), x, y)
    add("kappa_binary_good_fit(==2)", lambda p, q: cohen_kappa_score(p == 2, q == 2), x, y)
    rows.append(dict(metric="exact_agreement", value=round((x == y).mean(), 4), ci95_low=np.nan, ci95_high=np.nan))
    rows.append(dict(metric="n_annotated_both", value=len(ok), ci95_low=np.nan, ci95_high=np.nan))
    rows.append(dict(metric="n_dropped_missing", value=n_all - len(ok), ci95_low=np.nan, ci95_high=np.nan))
    summ = pd.DataFrame(rows)

    out = os.path.dirname(os.path.abspath(a.sheet))
    summ.to_csv(os.path.join(out, "agreement_summary.csv"), index=False)
    dis = ok[x != y][["item_id", A1, A2, "annotator1_notes", "annotator2_notes"]].copy()
    dis["adjudicated_score"] = ""   # NEW column; raw columns are never overwritten
    dis.to_csv(os.path.join(out, "disagreements_to_adjudicate.csv"), index=False)
    print(summ.to_string(index=False))
    print(f"\n{len(dis)} disagreements written to disagreements_to_adjudicate.csv")

    if os.path.exists(KEY):
        key = pd.read_csv(KEY)[["item_id", "label"]]
        m = ok.merge(key, on="item_id")
        m["human_mean"] = (m[A1] + m[A2]) / 2
        m["human_binary"] = (m["human_mean"] >= 1).astype(int)
        audit = pd.DataFrame([dict(
            n=len(m),
            weak_label_agrees_with_human_binary=round((m["human_binary"] == m["label"]).mean(), 4),
            weak_pos_rated_nofit=int(((m.label == 1) & (m.human_mean < 1)).sum()),
            weak_neg_rated_fit=int(((m.label == 0) & (m.human_mean >= 1)).sum()))])
        audit.to_csv(os.path.join(out, "weak_label_audit.csv"), index=False)
        print("\nWeak-label audit:\n", audit.to_string(index=False))
        print("(Only run the audit AFTER both annotators are finished -- the key is hidden on purpose.)")


if __name__ == "__main__":
    main()
