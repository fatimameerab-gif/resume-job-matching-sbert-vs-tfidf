import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "results/rq3"
def main(out=OUT):
    A = pd.read_csv(f"{out}/predictions_A.csv"); R = pd.read_csv(f"{out}/grid_per_split_A.csv")
    reps = sorted(A.representation.unique()); fig, axes = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 4), squeeze=False)
    for ax, rep in zip(axes[0], reps):
        best = R[R.representation == rep].groupby("classifier").inner_auc.mean().idxmax(); g = A[(A.representation == rep) & (A.classifier == best)]
        edges = np.linspace(0, 1, 6); idx = np.clip(np.digitize(g.p, edges[1:-1]), 0, 4); xs, ys, ns = [], [], []
        for b in range(5):
            m = idx == b
            if m.sum() >= 5: xs.append(g.p[m].mean()); ys.append(g.label[m].mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.plot(xs, ys, "o-"); ax.set_title(f"{rep} + {best}", fontsize=10); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed positive rate")
        for x, y, n in zip(xs, ys, ns): ax.annotate(str(n), (x, y), fontsize=7, xytext=(3, -9), textcoords="offset points")
    fig.tight_layout(); fig.savefig(f"{out}/fig_reliability_designA.png", dpi=150); plt.close(fig)
    for f in sorted(os.listdir(out)):
        if not (f.startswith("mean_ranks_B_auc") and f.endswith(".csv")): continue
        M = pd.read_csv(f"{out}/{f}"); F = pd.read_csv(f"{out}/friedman_B_auc.csv").set_index("representation")
        reps = M.representation.unique(); fig, axes = plt.subplots(len(reps), 1, figsize=(7, 1.6 * len(reps) + 1), squeeze=False)
        for ax, rep in zip(axes[:, 0], reps):
            g = M[M.representation == rep].sort_values("mean_rank"); cd = F.loc[rep, "nemenyi_cd"]
            ax.hlines(0, 1, g.mean_rank.max() + 0.5, color="k", lw=0.8); ax.plot(g.mean_rank, np.zeros(len(g)), "o")
            for i, (c, r) in enumerate(zip(g.classifier, g.mean_rank)): ax.annotate(c, (r, 0), xytext=(0, 8 + 10 * (i % 3)), textcoords="offset points", ha="center", fontsize=7, rotation=0)
            ax.hlines(-0.15, 1, 1 + cd, color="r", lw=3); ax.text(1, -0.35, f"CD = {cd:.2f}", color="r", fontsize=7); ax.set_ylim(-0.5, 1.3); ax.set_yticks([]); ax.set_title(f"{rep}: mean rank of AUC over {int(F.loc[rep, 'n_blocks'])} splits (lower = better)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{out}/fig_cd_designB_auc.png", dpi=150); plt.close(fig)
if __name__ == "__main__":
    main()
