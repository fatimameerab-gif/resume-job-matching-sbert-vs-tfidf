import os, sys, time, hashlib, warnings, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, brier_score_loss, recall_score)
warnings.filterwarnings("ignore")
OUT = "results/rq3"; EMB_DIR = "results/embeddings"; PAIRS = "data/resume_job_pairs_v3.csv"; SEED_DIR = "data/eval_v7/frozen_seeds"
N_B_SPLITS = 20; INNER_FOLDS = 3

# ---------------------------------------------------------------- classifier grids (same for every representation)
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

# ---------------------------------------------------------------- features
def sha(t): return hashlib.sha1(str(t).encode("utf8")).hexdigest()
def head(U, V): return np.hstack([np.abs(U - V), (U * V).sum(1, keepdims=True)])
def cosf(U, V): return (U * V).sum(1, keepdims=True)
def load_emb(name):
    f = f"{EMB_DIR}/{name}.npz"
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); return dict(zip(z["keys"], z["vectors"]))
def emb_rows(E, texts): return np.vstack([E[sha(t)] for t in texts])

class Rep:
    """name + a function (train_df, other_df) -> (Xtr_full, Xte_full, cos_tr, cos_te)."""
    def __init__(self, name, fn): self.name, self.fn = name, fn
def tfidf_rep(tr, te):
    v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
    f = lambda d: (v.transform(d.resume_text).toarray(), v.transform(d.job_text).toarray())
    (a, b), (c, d) = f(tr), f(te); return head(a, b), head(c, d), cosf(a, b), cosf(c, d)
def lsa_rep(tr, te):
    v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text]))
    svd = TruncatedSVD(100, random_state=42).fit(v.transform(pd.concat([tr.resume_text, tr.job_text])))
    def f(d):
        U, V = svd.transform(v.transform(d.resume_text)), svd.transform(v.transform(d.job_text))
        n = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12); return n(U), n(V)
    (a, b), (c, d) = f(tr), f(te); return head(a, b), head(c, d), cosf(a, b), cosf(c, d)
def dense_rep(E):
    def fn(tr, te):
        a, b, c, d = (emb_rows(E, tr.resume_text), emb_rows(E, tr.job_text), emb_rows(E, te.resume_text), emb_rows(E, te.job_text))
        return head(a, b), head(c, d), cosf(a, b), cosf(c, d)
    return fn

# ---------------------------------------------------------------- metrics
def ece(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
def hit_mrr(df, s):
    d = df[["resume_id", "label"]].copy(); d["s"] = np.asarray(s); h, r = [], []
    for _, g in d.groupby("resume_id"):
        if g.label.sum() != 1: continue
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + .5 * (o == p).sum(); h.append(float(k == 1)); r.append(1 / k)
    return float(np.mean(h)), float(np.mean(r))
def minmax_to_unit(s, lo, hi): return np.clip((s - lo) / max(hi - lo, 1e-12), 0, 1)

# ---------------------------------------------------------------- one (representation, classifier) evaluation on one split
def evaluate(rep_name, Xtr, Xte, Ctr, Cte, tr, te, clf_name, seed):
    t0 = time.time(); grid = GRIDS[clf_name]; ytr = tr.label.values
    A_tr, A_te = (Ctr, Cte) if clf_name == "CosineThreshold" else (Xtr, Xte)
    gkf = GroupKFold(INNER_FOLDS); folds = list(gkf.split(A_tr, ytr, groups=tr.resume_id.values))
    best = None
    for hp in grid:                                          # tuning: inner grouped CV, selection by mean inner AUC (threshold-free)
        oof = np.zeros(len(tr)); aucs = []
        for a, b in folds:
            m = make(clf_name, hp, seed).fit(A_tr[a], ytr[a]); oof[b] = score(m, A_tr[b])
            if len(set(ytr[b])) == 2: aucs.append(roc_auc_score(ytr[b], oof[b]))
        mu = float(np.mean(aucs))
        if best is None or mu > best[0]: best = (mu, hp, oof.copy())
    _, hp, oof = best
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(oof[:, None], ytr)           # Platt calibration on OOF scores
    p_oof = cal.predict_proba(oof[:, None])[:, 1]
    cand = np.unique(np.round(p_oof, 3)); thr = float(cand[np.argmax([balanced_accuracy_score(ytr, p_oof >= t) for t in cand])])
    model = make(clf_name, hp, seed).fit(A_tr, ytr); s_te = score(model, A_te); p_te = cal.predict_proba(s_te[:, None])[:, 1]
    pred = (p_te >= thr).astype(int); y = te.label.values; h, m = hit_mrr(te, p_te)
    # untuned/default-threshold view of the same fitted model, to show what the old protocol would have reported
    if hasattr(model, "predict"): pred_default = model.predict(A_te)
    raw_p = p_te if clf_name not in ("RandomForest", "GradientBoosting", "kNN", "MLP", "LR", "CosineThreshold") else (model.predict_proba(A_te)[:, 1])
    return dict(representation=rep_name, classifier=clf_name, hp=str(hp), inner_auc=best[0], threshold=thr,
                auc=roc_auc_score(y, p_te), pr_auc=average_precision_score(y, p_te), hit1=h, mrr=m,
                acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred), mcc=matthews_corrcoef(y, pred), recall=recall_score(y, pred),
                brier=brier_score_loss(y, p_te), ece=ece(y, p_te), brier_raw=brier_score_loss(y, np.clip(raw_p, 0, 1)) if clf_name not in ("LinearSVM", "RBF-SVM") else np.nan,
                acc_default_threshold=accuracy_score(y, pred_default), recall_default_threshold=recall_score(y, pred_default),
                fit_seconds=time.time() - t0), p_te

def make_reps():
    reps = [Rep("TF-IDF", tfidf_rep), Rep("LSA", lsa_rep)]
    if os.path.isdir(EMB_DIR):
        for f in sorted(os.listdir(EMB_DIR)):
            if f.endswith(".npz"): reps.append(Rep(f[:-4], dense_rep(load_emb(f[:-4]))))
    return reps

def run_design(name, splits, reps, classifiers, out):
    rows, preds = [], []
    for sid, (tr, te) in enumerate(splits):
        for rep in reps:
            Xtr, Xte, Ctr, Cte = rep.fn(tr, te)
            for c in classifiers:
                r, p = evaluate(rep.name, Xtr, Xte, Ctr, Cte, tr.reset_index(drop=True), te.reset_index(drop=True), c, seed=sid)
                r.update(design=name, split=sid); rows.append(r)
                preds.append(pd.DataFrame(dict(design=name, split=sid, representation=rep.name, classifier=c, resume_id=te.resume_id.values, job_id=te.job_id.values, label=te.label.values, p=p)))
        print(f"[{name}] split {sid + 1}/{len(splits)} done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{out}/grid_per_split_{name}.csv", index=False)
    pd.concat(preds).to_csv(f"{out}/predictions_{name}.csv", index=False); return R

def frozen_splits():
    out = []
    for s in range(5):
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv"); out.append((d[d.split == "train"], d[d.split == "test"]))
    return out
def repeated_splits(n):
    d = pd.read_csv(PAIRS); out = []
    for s in range(n):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=100 + s).split(d, groups=d.resume_id)); out.append((d.iloc[tr], d.iloc[te]))
    return out

# ---------------------------------------------------------------- statistics (Demsar)
def nemenyi_cd(k, n, alpha=0.05):
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2); return q * np.sqrt(k * (k + 1) / (6 * n))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, j in enumerate(o): run = max(run, (n - i) * p[j]); adj[j] = min(1, run)
    return adj
def rank_tests(R, metric, out, tag):
    rows, cd_rows, pw = [], [], []
    for rep, g in R.groupby("representation"):
        M = g.pivot(index="split", columns="classifier", values=metric)
        if M.shape[1] < 3 or M.shape[0] < 3: continue
        st, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
        ranks = M.rank(axis=1, ascending=False).mean(); cd = nemenyi_cd(M.shape[1], M.shape[0])
        rows.append(dict(representation=rep, metric=metric, n_blocks=M.shape[0], n_classifiers=M.shape[1], friedman_chi2=st, friedman_p=p, nemenyi_cd=cd))
        for c in M.columns: cd_rows.append(dict(representation=rep, metric=metric, classifier=c, mean_rank=ranks[c], mean_metric=M[c].mean()))
        pairs = list(itertools.combinations(M.columns, 2)); pv = []
        for a, b in pairs:
            d = M[a] - M[b]; pv.append(stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0)
        for (a, b), pr, ph in zip(pairs, pv, holm(pv)): pw.append(dict(representation=rep, metric=metric, a=a, b=b, mean_diff=float((M[a] - M[b]).mean()), wilcoxon_p=pr, holm_p=ph))
    pd.DataFrame(rows).to_csv(f"{out}/friedman_{tag}_{metric}.csv", index=False)
    pd.DataFrame(cd_rows).to_csv(f"{out}/mean_ranks_{tag}_{metric}.csv", index=False)
    pd.DataFrame(pw).to_csv(f"{out}/pairwise_holm_{tag}_{metric}.csv", index=False)

def summarise(R, out, tag):
    cols = ["auc", "pr_auc", "hit1", "mrr", "acc", "bal_acc", "f1", "mcc", "recall", "brier", "ece", "acc_default_threshold", "recall_default_threshold", "fit_seconds"]
    S = R.groupby(["representation", "classifier"]).agg({c: "mean" for c in cols}).reset_index()
    sd = R.groupby(["representation", "classifier"]).agg(auc_sd=("auc", "std"), acc_sd=("acc", "std")).reset_index()
    S = S.merge(sd, on=["representation", "classifier"]).sort_values(["representation", "auc"], ascending=[True, False]); S.round(4).to_csv(f"{out}/grid_summary_{tag}.csv", index=False); return S

def main(out=OUT, n_b=N_B_SPLITS, reps=None, classifiers=None, designs=("A", "B")):
    os.makedirs(out, exist_ok=True); reps = reps or make_reps(); classifiers = classifiers or list(GRIDS)
    print("representations:", [r.name for r in reps], "| classifiers:", classifiers)
    res = {}
    if "A" in designs: res["A"] = run_design("A", frozen_splits(), reps, classifiers, out)
    if "B" in designs: res["B"] = run_design("B", repeated_splits(n_b), reps, classifiers, out)
    for tag, R in res.items():
        S = summarise(R, out, tag)
        for m in ("auc", "mrr", "bal_acc"): rank_tests(R, m, out, tag)
        pd.set_option("display.width", 250, "display.max_columns", 30); print(f"\n== design {tag} (mean over splits) ==\n", S[["representation", "classifier", "auc", "hit1", "mrr", "acc", "bal_acc", "recall", "brier", "ece", "acc_default_threshold", "recall_default_threshold"]].round(3).to_string(index=False))
    open(f"{out}/NOTE_rq3.txt", "w").write("Hyper-parameters: inner resume-grouped 3-fold CV on training pairs, selection by inner AUC. Threshold: max balanced accuracy on Platt-calibrated inner OOF scores. "
        "'acc_default_threshold/recall_default_threshold' = the fitted model's own predict() (RF: 0.5 on its probability scale) to show the old artefact. "
        "Design B blocks come from repeated splits of ONE pairs file (job texts can repeat across train/test) and are not independent samples of the population: use them for ranking classifiers, not as precise population estimates. "
        "Gradient boosting = sklearn HistGradientBoosting (LightGBM/XGBoost not installed).\n")

if __name__ == "__main__":
    # python rq3_classifier_grid_v3.py cpu     -> TF-IDF + LSA only            -> results/rq3
    # python rq3_classifier_grid_v3.py dense   -> every results/embeddings/*.npz -> results/rq3_dense   (after colab_export_embeddings.py)
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"; nb = int(sys.argv[2]) if len(sys.argv) > 2 else N_B_SPLITS
    if mode == "cpu": main(out=OUT, n_b=nb, reps=[Rep("TF-IDF", tfidf_rep), Rep("LSA", lsa_rep)])
    else: main(out=OUT + "_dense", n_b=nb, reps=[r for r in make_reps() if r.name not in ("TF-IDF", "LSA")])
