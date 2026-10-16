# make_figures.py (RQ1_FINAL)  --  run from this folder in VS Code:  python make_figures.py
import os, pandas as pd, numpy as np, matplotlib
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)   # runs from the folder where this file lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.size":9,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})
out="figures/"
C={"lex":"#7a7a7a","frozen":"#d98c3f","ft":"#2a6f9e","hyb":"#1b4565"}

# Fig 1: accuracy
d=pd.read_csv("results/colab_v7b_finetune/finetune_summary_clean.csv").set_index("model")
rows=[("TF-IDF head","TF-IDF head","lex"),
("MiniLM frozen head","SBERT-MiniLM-L6-v2 frozen head","frozen"),
("mpnet frozen head","SBERT-mpnet-base-v2 frozen head","frozen"),
("MiniLM fine-tuned","SBERT-MiniLM-L6-v2 fine-tuned head","ft"),
("mpnet fine-tuned","SBERT-mpnet-base-v2 fine-tuned head","ft"),
("mpnet fine-tuned\n+ TF-IDF hybrid","Hybrid TF-IDF + SBERT-mpnet-base-v2 fine-tuned (weighted)","hyb")]
fig,ax=plt.subplots(figsize=(5.2,3));
x=range(len(rows))
ax.bar(x,[d.loc[r[1],"acc_mean"] for r in rows],yerr=[d.loc[r[1],"acc_sd"] for r in rows],color=[C[r[2]] for r in rows],capsize=3,width=.65)
ax.axhline(d.loc["TF-IDF head","acc_mean"],ls="--",color="k",lw=.8)
for i,r in enumerate(rows): ax.text(i,d.loc[r[1],"acc_mean"]+d.loc[r[1],"acc_sd"]+.008,f'{d.loc[r[1],"acc_mean"]:.3f}',ha="center",fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows],rotation=25,ha="right"); ax.set_ylim(0.6,1.0); ax.set_ylabel("Accuracy (mean ± SD, 5 seeds)")
fig.savefig(out+"Figure_1.pdf"); fig.savefig(out+"Figure_1.png",dpi=300); plt.show()

# Fig 2: learning curve
lc=pd.read_csv("results/colab_v7b_learning_curve/learning_curve_summary.csv")
fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),sharey=True)
for ax,enc in zip(axs,["SBERT-mpnet-base-v2","JobBERT-v2"]):
    for m,lab,c,ls in [("TF-IDF head","TF-IDF head",C["lex"],"-"),("frozen head","frozen head",C["frozen"],"-"),("fine-tuned head","fine-tuned head",C["ft"],"-")]:
        s=lc[(lc.encoder==enc)&(lc.model==m)].sort_values("n_train_resumes")
        ax.errorbar(s.n_train_resumes,s.acc,yerr=s.acc_sd,marker="o",ms=3,color=c,label=lab,capsize=2,lw=1.2)
    ax.set_xscale("log"); ax.set_xticks([12,29,58,116]); ax.set_xticklabels([12,29,58,116]); ax.set_xlabel("Training resumes"); ax.set_title(enc.replace("SBERT-",""),fontsize=9)
axs[0].set_ylabel("Accuracy (3 seeds)"); axs[0].legend(frameon=False,fontsize=8)
fig.savefig(out+"Figure_2.pdf"); fig.savefig(out+"Figure_2.png",dpi=300); plt.show()


# ---- Fig 5 (ROC + PR) and Fig 6 (confusion matrices), pooled over the 5 seeds (750 test pairs)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
pp = pd.read_csv("results/colab_v7b_finetune/pair_predictions.csv")
three = [("TF-IDF head", "TF-IDF head", C["lex"], "-"),
         ("mpnet frozen head", "SBERT-mpnet-base-v2 frozen head", C["frozen"], "--"),
         ("mpnet fine-tuned head", "SBERT-mpnet-base-v2 fine-tuned head", C["ft"], "-")]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
for lab, m, c, ls in three:
    d = pp[pp.model == m]
    fpr, tpr, _ = roc_curve(d.label, d.score); pr, rc, _ = precision_recall_curve(d.label, d.score)
    axs[0].plot(fpr, tpr, color=c, ls=ls, lw=1.4, label=f"{lab} (AUC {auc(fpr, tpr):.2f})")
    axs[1].plot(rc, pr, color=c, ls=ls, lw=1.4, label=f"{lab} (AP {average_precision_score(d.label, d.score):.2f})")
axs[0].plot([0, 1], [0, 1], color="#bbbbbb", lw=.8, ls=":")
axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate")
axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0.2, 1.02)
for a, t in zip(axs, ["(a) ROC", "(b) Precision-recall"]):
    a.legend(frameon=False, fontsize=7, loc="lower right" if t == "(a) ROC" else "lower left"); a.set_title(t, fontsize=9, y=-0.34)
fig.savefig(out + "Figure_3.pdf"); fig.savefig(out + "Figure_3.png", dpi=300); plt.show()

from matplotlib.colors import LinearSegmentedColormap
cm1 = LinearSegmentedColormap.from_list("b", ["#ffffff", C["ft"]])
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.5))
for ax, (lab, m, c, ls), t in zip(axs, three, ["(a)", "(b)", "(c)"]):
    d = pp[pp.model == m]; cm = confusion_matrix(d.label, d.pred, labels=[0, 1]); pc = cm / cm.sum(1, keepdims=True)
    ax.imshow(pc, cmap=cm1, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pc[i, j]*100:.0f}%)", ha="center", va="center", fontsize=8, color="white" if pc[i, j] > .55 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if t == "(a)" else "")
    ax.set_title(f"{t} {lab}", fontsize=8, y=-0.42)
    for s in ax.spines.values(): s.set_visible(False)
fig.savefig(out + "FigureS1_confusion.pdf"); fig.savefig(out + "FigureS1_confusion.png", dpi=300); plt.show()
