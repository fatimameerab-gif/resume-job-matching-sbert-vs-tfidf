"""Small shared helpers copied unchanged from rq3_classifier_grid_v3.py (classifier grids, model factory, scoring, pair features, Holm correction),
so that rq3_fullpool_pipelines.py runs from the RQ1 folder without the RQ3 folder."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def _lr(C): return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
def _lsvm(C): return LinearSVC(C=C, class_weight="balanced", max_iter=20000)
GRIDS = {
    "LR": [dict(C=c) for c in (0.01, 0.1, 1, 10)],
    "LinearSVM": [dict(C=c) for c in (0.01, 0.1, 1)],
    "RBF-SVM": [dict(C=c) for c in (1, 10)],
    "RandomForest": [dict(max_depth=d, min_samples_leaf=l) for d in (None, 10) for l in (1, 3)],
    "GradientBoosting": [dict(learning_rate=lr, max_depth=d) for lr in (0.05, 0.1) for d in (3, None)],
    "kNN": [dict(k=k) for k in (5, 15)],
    "MLP": [dict(alpha=a) for a in (1e-3, 1e-1)],
    "CosineThreshold": [dict()],
}
def make(name, hp, seed=42):
    if name == "LR": return _lr(hp["C"])
    if name == "LinearSVM": return _lsvm(hp["C"])
    if name == "RBF-SVM": return SVC(C=hp["C"], kernel="rbf", gamma="scale", class_weight="balanced")
    if name == "RandomForest": return RandomForestClassifier(200, max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"], class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "GradientBoosting": return HistGradientBoostingClassifier(learning_rate=hp["learning_rate"], max_depth=hp["max_depth"], max_iter=100, class_weight="balanced", random_state=seed)
    if name == "kNN": return KNeighborsClassifier(hp["k"], metric="cosine", weights="distance")
    if name == "MLP":     # scale-sensitive: features are standardised INSIDE the model (fit on training rows only); unscaled |u-v| values (~0.01-0.1) left the MLP at chance in a first run
        from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=hp["alpha"], max_iter=400, early_stopping=True, random_state=seed))
    if name == "CosineThreshold": return LogisticRegression(max_iter=1000)       # on the single cosine feature = a tuned threshold
def score(clf, X):
    if hasattr(clf, "decision_function") and not hasattr(clf, "predict_proba"): return clf.decision_function(X)
    if isinstance(clf, SVC): return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]

def head(U, V): return np.hstack([np.abs(U - V), (U * V).sum(1, keepdims=True)])

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, j in enumerate(o): run = max(run, (n - i) * p[j]); adj[j] = min(1, run)
    return adj
