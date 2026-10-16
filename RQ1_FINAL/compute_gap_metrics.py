import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, roc_auc_score
OUT = "results/skill_gap"; TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)

def gold_table(a, b, key):
    d = a[["item_id", "present"]].merge(b[["item_id", "present"]], on="item_id", suffixes=("_a", "_b")).merge(key, on="item_id")
    for c in ("present_a", "present_b"): d[c] = d[c].astype(str).str.strip().str.lower()
    bad = ~d.present_a.isin(["yes", "partly", "no"]) | ~d.present_b.isin(["yes", "partly", "no"])
    if bad.any(): raise SystemExit(f"{int(bad.sum())} rows have a missing/invalid label (allowed: yes, partly, no).")
    ba, bb = (d.present_a != "no").astype(int), (d.present_b != "no").astype(int)
    d["gold_present"] = np.where(ba == bb, ba, -1); return d

def prf(y_missing, pred_missing):
    p, r, f, _ = precision_recall_fscore_support(y_missing, pred_missing, average="binary", zero_division=0); return p, r, f

def main(fa, fb, fk, out=OUT, n_boot=1000):
    import os; os.makedirs(out, exist_ok=True)
    d = gold_table(pd.read_csv(fa), pd.read_csv(fb), pd.read_csv(fk))
    agree = pd.DataFrame([dict(n_items=len(d), kappa_3class=cohen_kappa_score(d.present_a, d.present_b), kappa_binary=cohen_kappa_score(d.present_a != "no", d.present_b != "no"),
                               pct_exact_agreement=float((d.present_a == d.present_b).mean()), pct_binary_disagreement_excluded=float((d.gold_present == -1).mean()),
                               gold_missing_rate=float((d[d.gold_present >= 0].gold_present == 0).mean()))]); agree.round(4).to_csv(f"{out}/gold_agreement.csv", index=False)
    g = d[d.gold_present >= 0].copy(); y_miss = (g.gold_present == 0).astype(int).values; rows = []
    systems = {"exact_string": (1 - g.exact_present.values)} | {f"linking_tfidf_tau{t}": (g.tfidf_char_score.values < t).astype(int) for t in TAUS}
    rng = np.random.default_rng(0); ids = g.resume_id.unique(); idx_by = {r: np.where(g.resume_id.values == r)[0] for r in ids}
    for name, pm in systems.items():
        p, r, f = prf(y_miss, pm); fs = []
        for _ in range(n_boot // 2):
            ii = np.concatenate([idx_by[x] for x in rng.choice(ids, len(ids))]); fs.append(prf(y_miss[ii], pm[ii])[2])
        rows.append(dict(system=name, n_items=len(g), gold_missing_rate=y_miss.mean(), precision_missing=p, recall_missing=r, f1_missing=f, f1_ci_lo=np.percentile(fs, 2.5), f1_ci_hi=np.percentile(fs, 97.5),
                         predicted_missing_rate=float(pm.mean())))
    pd.DataFrame(rows).round(4).to_csv(f"{out}/gold_system_metrics.csv", index=False)
    auc = roc_auc_score(1 - y_miss, g.tfidf_char_score.values) if len(set(y_miss)) == 2 else np.nan
    pd.DataFrame([dict(linking_score_auc_for_present=auc)]).round(4).to_csv(f"{out}/gold_linking_auc.csv", index=False)
    pd.set_option("display.width", 200); print(agree.round(3).to_string(index=False)); print(pd.DataFrame(rows).round(3).to_string(index=False)); print("linking-score AUC for 'present':", round(float(auc), 3))

if __name__ == "__main__":
    if len(sys.argv) < 3: raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "annotation_kit_skillgap/KEY-system-scores-do-not-share.csv")
