import os; os.makedirs("figures", exist_ok=True)
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
C = {"lex": "#7a7a7a", "frozen": "#d98c3f", "ft": "#2a6f9e"}
p = pd.read_csv("../RQ3_FINAL/results/rq3/pipeline_comparison.csv").set_index("model")
rows = [("mpnet fine-tuned", "SBERT-mpnet-base-v2 fine-tuned head", "ft"), ("JobBERT-v2 fine-tuned", "JobBERT-v2 fine-tuned head", "ft"), ("MiniLM fine-tuned", "SBERT-MiniLM-L6-v2 fine-tuned head", "ft"),
        ("TF-IDF + tuned\ngradient boosting", "TF-IDF + tuned GradientBoosting (RQ3)", "lex"), ("LSA + tuned\nRBF-SVM", "LSA + tuned RBF-SVM (RQ3)", "lex"), ("JobBERT-v2 frozen head", "JobBERT-v2 frozen head", "frozen"),
        ("TF-IDF cosine\n(zero-shot)", "TF-IDF cosine (zero-shot)", "lex"), ("TF-IDF head\n(untuned)", "TF-IDF head", "lex"), ("mpnet frozen head", "SBERT-mpnet-base-v2 frozen head", "frozen")]
fig, ax = plt.subplots(figsize=(5.6, 3.4)); v = [p.loc[r[1], "mrr"] for r in rows]; n = len(rows)
ax.barh(range(n)[::-1], v, color=[C[r[2]] for r in rows], height=.65)
ax.axvline(p.loc["TF-IDF head", "mrr"], ls="--", color="k", lw=.8)
for i, val in zip(range(n)[::-1], v): ax.text(val + .003, i, f"{val:.3f}", va="center", fontsize=8)
ax.set_yticks(range(n)[::-1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(.8, 1.0); ax.set_xlabel("MRR (same five frozen test sets, three candidates)")
fig.savefig("figures/Figure_5.pdf"); fig.savefig("figures/Figure_5.png", dpi=200); print("ok", p.loc["TF-IDF cosine (zero-shot)", "mrr"], p.loc["TF-IDF head", "mrr"])
