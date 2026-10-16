import glob
import itertools
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stats_utils import holm_bonferroni, cliffs_delta

OUT_DIR = "results"

fold_scores = {}
for f in sorted(glob.glob(f"{OUT_DIR}/rq1_taskA_*_folds.csv")):
    name = os.path.basename(f).replace("rq1_taskA_", "").replace("_folds.csv", "")
    df = pd.read_csv(f)
    if len(df) != 10:
        print(f"SKIPPING {name}: has {len(df)} folds, expected 10 (not directly comparable)")
        continue
    fold_scores[name] = df.sort_values("fold")["accuracy"].values

print(f"Representations with usable 10-fold accuracy: {sorted(fold_scores.keys())}")
if "domain" not in "".join(fold_scores.keys()).lower():
    print("NOTE: domain encoder (JobBERT-v2) fold CSV not found -- re-run "
          "rq1_colab_taskA_sbert.py (v2) on Colab to add it, then re-run this script.")

rows = []
for a, b in itertools.combinations(fold_scores.keys(), 2):
    sa, sb = fold_scores[a], fold_scores[b]
    try:
        stat, p = wilcoxon(sa, sb)
    except ValueError:
        stat, p = np.nan, 1.0
    rows.append({"rep_a": a, "rep_b": b, "mean_a": sa.mean(), "mean_b": sb.mean(),
                 "wilcoxon_stat": stat, "p_raw": p, "cliffs_delta": cliffs_delta(sa, sb)})

table = pd.DataFrame(rows)
adj, reject = holm_bonferroni(table["p_raw"].values)
table["p_holm"] = adj
table["significant_holm_0.05"] = reject
table = table.sort_values("p_raw")
table.to_csv(f"{OUT_DIR}/rq1_taskA_ALL_pairwise_tests.csv", index=False)

summary = pd.DataFrame({"representation": list(fold_scores.keys()),
                          "mean_accuracy": [v.mean() for v in fold_scores.values()],
                          "std_accuracy": [v.std() for v in fold_scores.values()]}).sort_values(
    "mean_accuracy", ascending=False)
summary.to_csv(f"{OUT_DIR}/rq1_taskA_ALL_fold_summary.csv", index=False)

print(f"\n{len(fold_scores)} representations, {len(table)} pairwise tests, "
      f"{int(table['significant_holm_0.05'].sum())} significant after Holm.\n")
print(summary.round(4).to_string(index=False))
print(f"\nSaved {OUT_DIR}/rq1_taskA_ALL_pairwise_tests.csv, {OUT_DIR}/rq1_taskA_ALL_fold_summary.csv")
