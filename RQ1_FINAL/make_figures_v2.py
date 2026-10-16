"""Figures 1 and 2 of the revised paper (review: majority line, zero-shot lines, 5 seeds, all encoders, category baseline).
Fine-tuned learning-curve rows still come from the 3-seed GPU run until the pinned 5-seed run (run_everything_gpu.py, stage learning_curve) is available:
set LC_FT to the new summary.csv then. Output: figures/Figure_1.pdf, Figure_2.pdf (+png)."""
import os, sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"; os.makedirs(OUT, exist_ok=True)
C = {"lex": "#7a7a7a", "frozen": "#d98c3f", "ft": "#2a6f9e", "cat": "#6a9a5b"}
a = pd.concat([pd.read_csv("results/colab_v7b_finetune/finetune_summary_clean.csv"), pd.read_csv("results/colab_v7b_newenc/finetune/finetune_summary.csv")]).drop_duplicates("model").set_index("model")
cb = pd.read_csv("results/category_baseline/summary.csv").set_index("model")
ENC = [("MiniLM", "SBERT-MiniLM-L6-v2"), ("mpnet", "SBERT-mpnet-base-v2"), ("e5", "e5-base-v2"), ("bge", "bge-base-en-v1.5"), ("gte", "gte-base"), ("JobBERT-v2", "JobBERT-v2")]
rows = [("TF-IDF head", a.loc["TF-IDF head", "acc_mean"], a.loc["TF-IDF head", "acc_sd"], "lex"), ("TF-IDF cosine", a.loc["TF-IDF cosine (zero-shot)", "acc_mean"], a.loc["TF-IDF cosine (zero-shot)", "acc_sd"], "lex")]
for n, k in ENC: rows.append((f"{n} frozen", a.loc[f"{k} frozen head", "acc_mean"], a.loc[f"{k} frozen head", "acc_sd"], "frozen"))
for n, k in ENC: rows.append((f"{n} fine-tuned", a.loc[f"{k} fine-tuned head", "acc_mean"], a.loc[f"{k} fine-tuned head", "acc_sd"], "ft"))
for n, k in (("TF-IDF", "TF-IDF"), ("mpnet", "SBERT-mpnet-base-v2"), ("JobBERT-v2", "JobBERT-v2")): rows.append((f"{n} category clf.", cb.loc[f"Category classifiers ({k})", "acc_mean"], cb.loc[f"Category classifiers ({k})", "acc_std"], "cat"))
fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(len(rows))
ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], color=[C[r[3]] for r in rows], capsize=2, width=.7)
ax.axhline(0.667, color="r", ls=":", lw=1); ax.text(len(rows) - .4, 0.672, "majority 0.667", color="r", ha="right", fontsize=7)
ax.axhline(a.loc["TF-IDF head", "acc_mean"], ls="--", color="k", lw=.7)
for i, r in enumerate(rows): ax.text(i, r[1] + r[2] + .006, f"{r[1]:.2f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right", fontsize=7); ax.set_ylim(0.6, 1.08); ax.set_ylabel("Accuracy (mean ± SD, 5 seeds)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C[k], label=l) for k, l in (("lex", "TF-IDF"), ("frozen", "frozen encoder + head"), ("ft", "fine-tuned encoder + head"), ("cat", "category classifiers"))], frameon=False, fontsize=7, loc="upper left", ncol=2)
fig.savefig(f"{OUT}/Figure_1.pdf"); fig.savefig(f"{OUT}/Figure_1.png", dpi=200); plt.close()
# Figure 2: learning curve
cpu = pd.read_csv("results/learning_curve_cpu/summary.csv"); LC_FT = os.environ.get("LC_FT", "results/colab_v7b_learning_curve/learning_curve_summary.csv"); g = pd.read_csv(LC_FT)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
for ax, enc in zip(axs, ["SBERT-mpnet-base-v2", "JobBERT-v2"]):
    spec = [(cpu[(cpu.encoder == "TF-IDF") & (cpu.model == "TF-IDF head")], "TF-IDF head", C["lex"], "-"), (cpu[(cpu.encoder == "TF-IDF") & (cpu.model == "TF-IDF cosine (zero-shot)")], "TF-IDF cosine (zero-shot)", C["lex"], ":"),
            (cpu[(cpu.encoder == enc) & (cpu.model == "frozen head")], "frozen head", C["frozen"], "-"), (cpu[(cpu.encoder == enc) & (cpu.model == "frozen cosine (zero-shot)")], "frozen cosine (zero-shot)", C["frozen"], ":"),
            (g[(g.encoder == enc) & (g.model == "fine-tuned head")], "fine-tuned head", C["ft"], "-")]
    for s, lab, c, ls in spec:
        s = s.sort_values("n_train_resumes"); ax.errorbar(s.n_train_resumes, s.acc, yerr=s.acc_sd, marker="o", ms=3, color=c, ls=ls, label=lab, capsize=2, lw=1.1)
    ax.axhline(0.667, color="r", ls=":", lw=.8); ax.set_xscale("log"); ax.set_xticks([12, 29, 58, 116]); ax.set_xticklabels([12, 29, 58, 116]); ax.set_xlabel("Training resumes"); ax.set_title(enc.replace("SBERT-", ""), fontsize=9)
axs[0].set_ylabel("Accuracy (mean ± SD, 5 seeds)"); axs[1].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.savefig(f"{OUT}/Figure_2.pdf"); fig.savefig(f"{OUT}/Figure_2.png", dpi=200); plt.close(); print("ok")
