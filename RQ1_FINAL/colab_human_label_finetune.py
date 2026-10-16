"""
SCRIPT 8 (GPU, ~20-40 min on a Tesla T4): fine-tuned encoders on HUMAN labels, with no leakage.

Question (review point 16b): do the fine-tuned models also rank the 240 human-annotated pairs well, when none of those
pairs (and none of their resumes or job texts) was in the fine-tuning data?

Design (cross-fitting by resume, so every annotated pair is scored by a model that never saw its resume or its job text):
  * the 145 resumes of the 240 annotated pairs are split into N_FOLDS folds (GroupKFold by resume, fixed seed).
  * for each fold: TRAINING = all pairs of data/resume_job_pairs_v3.csv whose resume is NOT in the fold and whose job text
    is NOT used by a pair of a fold resume (so neither resume nor job text leaks); TEST = the annotated pairs of the fold.
  * models (same recipe as script 5): TF-IDF head and cosine (TF-IDF fitted on the training texts), frozen encoder head and
    cosine, FINE-TUNED encoder head and cosine (CosineSimilarityLoss, 3 epochs, batch 8, lr 2e-5, 256 training tokens, AMP).
  * chunk pooling at inference. No threshold is used: the metric is the ROC AUC of the raw score (threshold-free).
  * metrics: AUC against human fit (mean of the two annotators >= 1) and against the weak label, Spearman against the
    mean human score, resume-clustered bootstrap CIs, paired difference to the TF-IDF head.

HOW TO RUN (Kaggle / Colab, GPU on): put the RQ1 project folder in the working directory (this file, colab_common.py,
colab_v7b_finetune_hybrid.py, representations.py, data/resume_job_pairs_v3.csv, results/human_annotation/item_scores.csv)
and run:   python colab_human_label_finetune.py
Quick check on CPU (no model download):   python colab_human_label_finetune.py --encoders stub
Output: results/human_label_finetune/{predictions.csv, human_label_finetune.csv, NOTE.txt}
Nothing here is tuned toward any model; labels of the 240 pairs are only used for scoring.
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings("ignore")

OUT = "results/human_label_finetune"
ITEMS = "results/human_annotation/item_scores.csv"
PAIRS = "data/resume_job_pairs_v3.csv"
DEFAULT_ENCODERS = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,JobBERT-v2"


def make_folds(items, n_folds, seed):
    """GroupKFold by resume with a seeded shuffle of the resume order (GroupKFold itself is deterministic)."""
    rng = np.random.RandomState(seed)
    rids = items.resume_id.unique(); perm = dict(zip(rids, rng.permutation(len(rids))))
    g = items.resume_id.map(perm).values
    fold = np.zeros(len(items), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_folds).split(items, groups=g)):
        fold[te] = k
    return fold


def training_pairs(pairs, test_items):
    """Pairs that share neither a resume nor a job text with the held-out annotated pairs."""
    bad_r = set(test_items.resume_id)
    bad_j = set(test_items.job_id) | set(pairs.loc[pairs.resume_id.isin(bad_r), "job_id"])
    bad_jt = set(pairs.loc[pairs.job_id.isin(bad_j), "job_text"])            # same text under another job_id
    keep = (~pairs.resume_id.isin(bad_r)) & (~pairs.job_text.isin(bad_jt))
    return pairs[keep].reset_index(drop=True)


def run_models(encoders, items, pairs, n_folds, fold_seed, train_seed, epochs, finetune, chunk_long):
    from colab_common import load_encoders, ENCODERS, Encoder, TfidfBackend
    import colab_v7b_finetune_hybrid as fth
    frozen = load_encoders(encoders, chunk_long=chunk_long)
    items = items.merge(pairs[["resume_id", "job_id", "resume_text", "job_text"]], on=["resume_id", "job_id"], how="left")
    assert items.resume_text.notna().all(), "some annotated pairs are missing from the pairs file"
    items["fold"] = make_folds(items, n_folds, fold_seed)
    out, t0 = [], time.time()
    for k in range(n_folds):
        te = items[items.fold == k].reset_index(drop=True)
        tr = training_pairs(pairs, te)
        assert not set(tr.resume_id) & set(te.resume_id) and not set(tr.job_text) & set(te.job_text)
        print(f"fold {k}: train {len(tr)} pairs / {tr.resume_id.nunique()} resumes, test {len(te)} annotated pairs / {te.resume_id.nunique()} resumes", flush=True)
        ytr = tr.label.values
        tf = TfidfBackend(pd.concat([tr.resume_text, tr.job_text]))
        Xtr, _ = fth._pair_feats(tf, tr); Xte, cte = fth._pair_feats(tf, te)
        scores = {"TF-IDF head": fth._head(Xtr, ytr, train_seed).predict_proba(Xte)[:, 1], "TF-IDF cosine": cte}
        for enc in frozen:
            Xtr_e, _ = fth._pair_feats(enc, tr); Xte_e, cte_e = fth._pair_feats(enc, te)
            scores[f"{enc.key} frozen head"] = fth._head(Xtr_e, ytr, train_seed).predict_proba(Xte_e)[:, 1]
            scores[f"{enc.key} frozen cosine"] = cte_e
            if finetune and enc.key != "stub":
                ft_model = fth._finetune(enc.key, tr, train_seed, epochs, fth.BATCH_SIZE, fth.LEARNING_RATE)
                _, jp, rp = ENCODERS[enc.key]
                ft = Encoder(f"FT[{enc.key}]", ENCODERS[enc.key][0], jp, rp, chunk_long=chunk_long, model=ft_model)
                Xtr_f, _ = fth._pair_feats(ft, tr); Xte_f, cte_f = fth._pair_feats(ft, te)
                scores[f"{enc.key} fine-tuned head"] = fth._head(Xtr_f, ytr, train_seed).predict_proba(Xte_f)[:, 1]
                scores[f"{enc.key} fine-tuned cosine"] = cte_f
                del ft_model, ft
                import gc, torch; gc.collect(); torch.cuda.empty_cache()
        for m, s in scores.items():
            out.append(pd.DataFrame(dict(item_id=te.item_id, fold=k, model=m, score=s)))
        print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    return items, pd.concat(out, ignore_index=True)


def analyse(items, preds, n_boot=2000, seed=0):
    P = preds.pivot(index="item_id", columns="model", values="score").reindex(items.item_id)
    yh, yw = items.human_fit_binary.values, items.label.values
    rid = items.resume_id.values; urid = np.unique(rid); grp = {r: np.where(rid == r)[0] for r in urid}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([grp[r] for r in rng.choice(urid, len(urid))]) for _ in range(n_boot)]
    def auc(m, y, idx=None):
        idx = np.arange(len(items)) if idx is None else idx
        return roc_auc_score(y[idx], P[m].values[idx]) if 0 < y[idx].sum() < len(idx) else np.nan
    rows = []
    for m in P.columns:
        bh = np.array([auc(m, yh, i) for i in boots]); bw = np.array([auc(m, yw, i) for i in boots])
        r = dict(model=m, auc_vs_human=auc(m, yh), human_lo=np.nanpercentile(bh, 2.5), human_hi=np.nanpercentile(bh, 97.5),
                 auc_vs_weak=auc(m, yw), weak_lo=np.nanpercentile(bw, 2.5), weak_hi=np.nanpercentile(bw, 97.5),
                 spearman_vs_human_mean=spearmanr(P[m], items.human_mean.values).correlation)
        if m != "TF-IDF head":
            d = np.array([auc(m, yh, i) - auc("TF-IDF head", yh, i) for i in boots])
            r.update(diff_vs_tfidf_head=auc(m, yh) - auc("TF-IDF head", yh), diff_lo=np.nanpercentile(d, 2.5), diff_hi=np.nanpercentile(d, 97.5))
        rows.append(r)
    return pd.DataFrame(rows).round(4).sort_values("auc_vs_human", ascending=False)


def main(encoders=DEFAULT_ENCODERS, n_folds=5, fold_seed=0, train_seed=0, epochs=3, finetune=True, chunk_long=True, out=OUT, n_boot=2000):
    os.makedirs(out, exist_ok=True)
    pairs = pd.read_csv(PAIRS); items = pd.read_csv(ITEMS)
    items, preds = run_models(encoders, items, pairs, n_folds, fold_seed, train_seed, epochs, finetune, chunk_long)
    preds.round(6).to_csv(f"{out}/predictions.csv", index=False)
    res = analyse(items, preds, n_boot=n_boot)
    res.to_csv(f"{out}/human_label_finetune.csv", index=False)
    open(f"{out}/NOTE.txt", "w").write(
        "Fine-tuned encoders scored on the 240 human-annotated pairs by cross-fitting: each annotated pair is scored by a model whose training data\n"
        "contain neither its resume nor its job text (GroupKFold by resume, %d folds, fold seed %d, train seed %d). Recipe as in script 5 (%d epochs, batch 8, lr 2e-5).\n"
        "Metric: ROC AUC of the raw score against human fit (mean of two annotators >= 1) and against the weak label; 95%% CIs from a resume-clustered bootstrap (%d resamples).\n"
        "One fold seed and one training seed only: the intervals do not include training randomness. The negatives of the sample are random pairs from other categories.\n"
        % (n_folds, fold_seed, train_seed, epochs, n_boot))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default=DEFAULT_ENCODERS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    main(a.encoders, a.folds, a.fold_seed, a.train_seed, a.epochs, not a.no_finetune, True, a.out, a.boot)
