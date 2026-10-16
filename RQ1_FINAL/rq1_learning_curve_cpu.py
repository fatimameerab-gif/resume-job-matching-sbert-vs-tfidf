"""Review 1.2: the CPU half of the learning curve, with the zero-shot lines the review asks for, for FIVE seeds and all six encoders.

Same protocol as colab_v7b_learning_curve.py (nested resume subsets in the order RandomState(1000+seed), 80/20 resume-grouped train/validation split of the subset,
test = the full 150 pairs of the seed, same head and same threshold rules - the functions are imported from colab_v7b_finetune_hybrid), but the frozen embeddings are read from
results/embeddings, so no GPU is needed. Rows added to the three-seed GPU curve:
  * TF-IDF cosine (zero-shot) and <encoder> frozen cosine, threshold chosen on the validation part of the SAME subset (the fair lexical/frozen competitor of a head),
  * TF-IDF head, <encoder> frozen head for seeds 0-4 (the GPU run had seeds 0-2),
  * a "majority" line (0.667) and a "TF-IDF head, validation threshold" variant (the original used 0.5, which a 36-pair head cannot calibrate).
Fine-tuned rows come from the GPU run (colab_v7b_learning_curve.py, now with SEEDS=5); `rq1_learning_curve_merge.py` joins both.
Output: results/learning_curve_cpu/{per_seed.csv, summary.csv}.  Run: python rq1_learning_curve_cpu.py   (about 2 min)
"""
import os, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")
import colab_v7b_finetune_hybrid as ft
from colab_common import TfidfBackend
SEED_DIR = "data/eval_v7/frozen_seeds"; OUT = "results/learning_curve_cpu"; FRACTIONS = (0.10, 0.25, 0.50, 1.00)
from emb_paths import EMB_DIR
ENC = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "e5-base-v2", "bge-base-en-v1.5", "gte-base", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


class EmbBackend:
    def __init__(self, key):
        z = np.load(f"{EMB_DIR}/{key}.npz", allow_pickle=True); self.M = dict(zip(z["keys"], z["vectors"])); self.key = key
    def encode(self, texts, kind=None): return np.vstack([self.M[sha(t)] for t in texts])


def ev(name, score, thr, y, seed, frac, n_res, n_pairs, enc):
    pred = (score >= thr).astype(int)
    try: auc = roc_auc_score(y, score)
    except ValueError: auc = np.nan
    return dict(seed=seed, fraction=frac, n_train_resumes=n_res, n_train_pairs=n_pairs, encoder=enc, model=name, acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred), auc=auc)


def main():
    os.makedirs(OUT, exist_ok=True); rows = []; E = {e: EmbBackend(e) for e in ENC if os.path.exists(f"{EMB_DIR}/{e}.npz")}
    for seed in range(5):
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{seed}.csv"); full_tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True); yte = te.label.values
        order = np.random.RandomState(1000 + seed).permutation(np.sort(full_tr.resume_id.unique()))
        for frac in FRACTIONS:
            keep = order[: max(4, int(round(frac * len(order))))]; sub = full_tr[full_tr.resume_id.isin(keep)]
            i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(sub, groups=sub.resume_id)); tr, va = sub.iloc[i_tr].reset_index(drop=True), sub.iloc[i_va].reset_index(drop=True)
            ytr, yva = tr.label.values, va.label.values; n_res, n_p = len(keep), len(sub)
            tf = TfidfBackend(pd.concat([tr.resume_text, tr.job_text])); Xtr, _ = ft._pair_feats(tf, tr); Xva, cva = ft._pair_feats(tf, va); Xte, cte = ft._pair_feats(tf, te)
            m = ft._head(Xtr, ytr, seed); p = m.predict_proba(Xte)[:, 1]; pva = m.predict_proba(Xva)[:, 1]
            rows.append(ev("TF-IDF head", p, 0.5, yte, seed, frac, n_res, n_p, "TF-IDF"))
            rows.append(ev("TF-IDF head (validation threshold)", p, ft._best_threshold(pva, yva), yte, seed, frac, n_res, n_p, "TF-IDF"))
            rows.append(ev("TF-IDF cosine (zero-shot)", cte, ft._best_threshold(cva, yva), yte, seed, frac, n_res, n_p, "TF-IDF"))
            rows.append(ev("majority class", np.zeros(len(yte)), 0.5, yte, seed, frac, n_res, n_p, "none"))
            for e, B in E.items():
                Xtr_e, _ = ft._pair_feats(B, tr); Xva_e, cva_e = ft._pair_feats(B, va); Xte_e, cte_e = ft._pair_feats(B, te)
                m = ft._head(Xtr_e, ytr, seed); p = m.predict_proba(Xte_e)[:, 1]
                rows.append(ev("frozen head", p, 0.5, yte, seed, frac, n_res, n_p, e)); rows.append(ev("frozen cosine (zero-shot)", cte_e, ft._best_threshold(cva_e, yva), yte, seed, frac, n_res, n_p, e))
        print("seed", seed, "done", flush=True)
    R = pd.DataFrame(rows); R.round(4).to_csv(f"{OUT}/per_seed.csv", index=False)
    S = R.groupby(["encoder", "model", "fraction"]).agg(n_train_resumes=("n_train_resumes", "mean"), n_seeds=("seed", "nunique"), acc=("acc", "mean"), acc_sd=("acc", "std"), auc=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
    S.round(4).to_csv(f"{OUT}/summary.csv", index=False); pd.set_option("display.width", 200)
    print(S[S.encoder.isin(["TF-IDF", "SBERT-mpnet-base-v2", "JobBERT-v2"])].pivot_table(index=["encoder", "model"], columns="fraction", values="acc").round(3).to_string())


if __name__ == "__main__":
    main()
