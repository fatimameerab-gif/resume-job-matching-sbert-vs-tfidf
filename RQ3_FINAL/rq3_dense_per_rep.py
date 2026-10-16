import sys, os, glob, pandas as pd
import rq3_classifier_grid_v3 as g
if sys.argv[1] == "merge":
    for tag in "AB":
        fs = sorted(glob.glob(f"results/rq3_dense/*/grid_summary_{tag}.csv"))
        if fs: pd.concat([pd.read_csv(f) for f in fs]).to_csv(f"results/rq3_dense/grid_summary_{tag}.csv", index=False)
    for pat in ("friedman_B_auc", "mean_ranks_B_auc", "pairwise_holm_B_auc", "friedman_A_auc"):
        fs = sorted(glob.glob(f"results/rq3_dense/*/{pat}.csv"))
        if fs: pd.concat([pd.read_csv(f) for f in fs]).to_csv(f"results/rq3_dense/{pat}.csv", index=False)
    print("merged"); sys.exit()
enc = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rep = [r for r in g.make_reps() if r.name == enc][0]
g.main(out=f"results/rq3_dense/{enc}", n_b=nb, reps=[rep], designs=("A",) if nb == 0 else ("A", "B"))
