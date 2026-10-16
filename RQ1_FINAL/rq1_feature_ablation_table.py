import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from representations import tfidf_features, lsa_features, static_embedding_features, FEATURE_MODES
from stats_utils import resume_clustered_bootstrap_ci

pairs = pd.read_csv("data/resume_job_pairs_v3.csv")
train = pairs[pairs["split"] == "train"].reset_index(drop=True)
test = pairs[pairs["split"] == "test"].reset_index(drop=True)

REPS = {"TF-IDF": tfidf_features, "LSA": lsa_features, "Static-word2vec-local": static_embedding_features}

rows = []
for rep_name, fn in REPS.items():
    for mode in FEATURE_MODES:
        out = fn(train["resume_text"], train["job_text"], test["resume_text"], test["job_text"], feature_mode=mode)
        X_tr, X_te = out[0], out[1]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_tr, train["label"])
        y_pred = clf.predict(X_te)
        acc, lo, hi, _ = resume_clustered_bootstrap_ci(test, y_pred, accuracy_score, n_boot=5000)
        f1, flo, fhi, _ = resume_clustered_bootstrap_ci(test, y_pred, f1_score, n_boot=5000)
        print(f"{rep_name:25s} {mode:28s} dim={X_tr.shape[1]:4d}  accuracy={acc:.4f} (CI {lo:.4f}-{hi:.4f})  f1={f1:.4f}")
        rows.append({"representation": rep_name, "feature_mode": mode, "dim": X_tr.shape[1],
                     "accuracy": acc, "acc_ci_low": lo, "acc_ci_high": hi, "f1": f1, "f1_ci_low": flo, "f1_ci_high": fhi})

table = pd.DataFrame(rows)
table.to_csv("results/rq1_feature_ablation_table.csv", index=False)
print("\nSaved results/rq1_feature_ablation_table.csv")
print("\nBy representation, best feature mode:")
print(table.loc[table.groupby("representation")["accuracy"].idxmax()][["representation", "feature_mode", "accuracy"]].to_string(index=False))
