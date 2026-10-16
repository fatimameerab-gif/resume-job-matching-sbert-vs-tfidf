# make_figures.py (RQ2_FINAL)  --  run from this folder in VS Code:  python make_figures.py
import os, pandas as pd, numpy as np, matplotlib
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)   # runs from the folder where this file lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.size":9,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})
out="figures/"
C={"lex":"#7a7a7a","frozen":"#d98c3f","ft":"#2a6f9e","hyb":"#1b4565"}

# Fig 3: CheckList
t=pd.read_csv("results/rq2/checklist_summary.csv"); t=t[t.model=="tfidf_cosLR"]
s1=pd.read_csv("results/rq2_dense/checklist_sbert_summary.csv"); s2=pd.read_csv("results/rq2_dense_more/checklist_sbert_summary.csv")
s=pd.concat([s1,s2]); s=s[s["head"]=="cosLR"]
tests=[("INV_filler","Filler sentence\n(unchanged)"),("INV_punct","Punctuation/case\n(unchanged)"),("DIR_up_copy_job","Copy job words\n(score up)"),("DIR_down_foreign","Add foreign text\n(score down)")]
models=[("TF-IDF",lambda k:t[t.test==k].pass_rate.iloc[0],C["lex"]),
("MiniLM",lambda k:s[(s.encoder=="SBERT-MiniLM-L6-v2")&(s.test==k)].pass_rate.iloc[0],"#9cc3de"),
("mpnet",lambda k:s[(s.encoder=="SBERT-mpnet-base-v2")&(s.test==k)].pass_rate.iloc[0],"#2a6f9e"),
("JobBERT-v2",lambda k:s[(s.encoder=="JobBERT-v2")&(s.test==k)].pass_rate.iloc[0],"#1b4565")]
fig,ax=plt.subplots(figsize=(6.2,3)); w=.2
for j,(n,f,c) in enumerate(models):
    ax.bar(np.arange(4)+(j-1.5)*w,[f(k) for k,_ in tests],w,label=n,color=c)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in tests]); ax.set_ylabel("Pass rate"); ax.set_ylim(0,1.1); ax.legend(frameon=False,ncol=4,fontsize=8,loc="upper center",bbox_to_anchor=(.5,1.15))
fig.savefig(out+"Figure_4.pdf"); fig.savefig(out+"Figure_4.png",dpi=300); plt.show()

