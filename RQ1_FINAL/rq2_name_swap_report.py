"""Review 3.5: name-swap probe reported as GROUP GAPS with intervals, not as 'shift minus a control that itself moves the scores'.
For every model and seed: the four name groups (male_A, female_A, male_B, female_B) each prepend one name to the test resumes; the quantity of interest is how much the head score
differs BETWEEN groups (max - min over the four mean shifts) and the gender gap (mean of female groups - mean of male groups, within list A and list B).
The neutral prefix control is reported on its own (it moves the scores by itself, so it is not subtracted).  Intervals: mean +- 1.96*SE over the 5 seeds (descriptive; seeds share resumes).
Input: results/colab_v7b_extras/name_swap_per_seed.csv (NOTE: that run used truncation, chunk_long=False -> the dense rows are for truncated texts).
Output: results/colab_v7b_extras/name_swap_gaps.csv.  Run: python rq2_name_swap_report.py"""
import numpy as np, pandas as pd
p = pd.read_csv("results/colab_v7b_extras/name_swap_per_seed.csv"); rows = []
for m, g in p.groupby("model"):
    per = []
    for s, x in g.groupby("seed"):
        sh = x.set_index("name_group").mean_head_score_shift; fl = x.set_index("name_group").decision_flip_rate
        grp = sh[["male_A", "female_A", "male_B", "female_B"]]
        per.append(dict(gap_max_min=grp.max() - grp.min(), gender_gap_A=sh.female_A - sh.male_A, gender_gap_B=sh.female_B - sh.male_B, control_shift=sh.neutral_control,
                        mean_group_shift=grp.mean(), flip_groups=fl[["male_A", "female_A", "male_B", "female_B"]].mean(), flip_control=fl.neutral_control))
    d = pd.DataFrame(per); r = dict(model=m)
    for c in d.columns:
        mu, se = d[c].mean(), d[c].std(ddof=1) / np.sqrt(len(d)); r[c] = round(mu, 4); r[c + "_ci"] = f"[{mu - 1.96 * se:.4f}, {mu + 1.96 * se:.4f}]"
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv("results/colab_v7b_extras/name_swap_gaps.csv", index=False)
pd.set_option("display.width", 250); print(R[["model", "gap_max_min", "gap_max_min_ci", "gender_gap_A", "gender_gap_A_ci", "gender_gap_B", "gender_gap_B_ci", "control_shift", "mean_group_shift", "flip_groups", "flip_control"]].to_string(index=False))
