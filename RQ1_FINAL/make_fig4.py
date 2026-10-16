"""Figure 4: CheckList pass rates with the scaled tolerance (0.1 x score SD), all models, 95% resume-clustered CI."""
import glob, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
import os; os.makedirs("figures", exist_ok=True)
R = "results/robust_stronger/"
C = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(R + "*/checklist_summary.csv"))])
rows = [("TF-IDF cosine", "TF-IDF cosine", "#7a7a7a"), ("TF-IDF head", "TF-IDF head", "#7a7a7a"), ("SBERT-MiniLM-L6-v2 frozen head", "MiniLM frozen", "#d98c3f"), ("SBERT-mpnet-base-v2 frozen head", "mpnet frozen", "#d98c3f"),
        ("bge-base-en-v1.5 frozen head", "bge frozen", "#d98c3f"), ("e5-base-v2 frozen head", "e5 frozen", "#d98c3f"), ("gte-base frozen head", "gte frozen", "#d98c3f"), ("JobBERT-v2 frozen head", "JobBERT-v2 frozen", "#d98c3f"),
        ("SBERT-MiniLM-L6-v2 fine-tuned head", "MiniLM fine-tuned", "#2a6f9e"), ("SBERT-mpnet-base-v2 fine-tuned head", "mpnet fine-tuned", "#2a6f9e"), ("JobBERT-v2 fine-tuned head", "JobBERT-v2 fine-tuned", "#2a6f9e")]
tests = [("INV_filler", "Filler sentence\n(expect no change)"), ("INV_punct", "Punctuation, case\n(expect no change)"), ("DIR_up_copy_job", "Copy job words\n(expect score up)"), ("DIR_down_foreign", "Add foreign text\n(expect score down)")]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 3.2), sharex=True)
for ax, (t, lab) in zip(axs, tests):
    for i, (m, l, c) in enumerate(rows[::-1]):
        x = C[(C.model == m) & (C.test == t)].iloc[0]
        ax.errorbar(x.pass_rate, i, xerr=[[x.pass_rate - x.ci_lo], [x.ci_hi - x.pass_rate]], fmt="o", color=c, ms=3.5, lw=1)
    ax.set_title(lab, fontsize=7.5); ax.set_xlim(0, 1.02); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows[::-1]] if ax is axs[0] else [])
    ax.grid(axis="x", color="#e5e5e5", lw=.6); ax.set_xlabel("Pass rate")
fig.savefig("figures/Figure_4.pdf"); fig.savefig("figures/Figure_4.png", dpi=200); print("ok")
