# make_figures.py (RQ3_FINAL)  --  run from this folder in VS Code:  python make_figures.py
import os, pandas as pd, numpy as np, matplotlib
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)   # runs from the folder where this file lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.size":9,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})
out="figures/"
C={"lex":"#7a7a7a","frozen":"#d98c3f","ft":"#2a6f9e","hyb":"#1b4565"}

# Fig 4: pipelines MRR
p=pd.read_csv("results/rq3/pipeline_comparison.csv").set_index("model")
rows=[("mpnet fine-tuned","SBERT-mpnet-base-v2 fine-tuned head","ft"),("MiniLM fine-tuned","SBERT-MiniLM-L6-v2 fine-tuned head","ft"),
("TF-IDF + tuned\ngradient boosting","TF-IDF + tuned GradientBoosting (RQ3)","lex"),("LSA + tuned\nRBF-SVM","LSA + tuned RBF-SVM (RQ3)","lex"),
("TF-IDF head\n(untuned)","TF-IDF head","lex"),("mpnet frozen head","SBERT-mpnet-base-v2 frozen head","frozen")]
fig,ax=plt.subplots(figsize=(5.4,3)); v=[p.loc[r[1],"mrr"] for r in rows]
ax.barh(range(len(rows))[::-1],v,color=[C[r[2]] for r in rows],height=.65)
ax.axvline(p.loc["TF-IDF head","mrr"],ls="--",color="k",lw=.8)
for i,val in zip(range(len(rows))[::-1],v): ax.text(val+.003,i,f"{val:.3f}",va="center",fontsize=8)
ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(.8,1.0); ax.set_xlabel("MRR (same 5 frozen test sets)")
fig.savefig(out+"Figure_5.pdf"); fig.savefig(out+"Figure_5.png",dpi=300); plt.show()
