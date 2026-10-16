import os, time, types
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import warnings; warnings.filterwarnings("ignore")

ENCODERS_TO_RUN = "SBERT-mpnet-base-v2,JobBERT-v2"
SEEDS = 3
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
OUT_DIR = "results/colab_v7b_learning_curve"

import colab_v7b_finetune_hybrid as ft
from colab_common import load_encoders, find_seed_dir, load_seed_pairs, TfidfBackend, ENCODERS, Encoder


def _finetuned_encoder(enc, tr, seed, epochs, chunk_long):
    if enc.key == "stub":                       # smoke tests only: no real fine-tuning
        return enc
    m = ft._finetune(enc.key, tr, seed, epochs, ft.BATCH_SIZE, ft.LEARNING_RATE)
    _, jp, rp = ENCODERS[enc.key][0], ENCODERS[enc.key][1], ENCODERS[enc.key][2]
    return Encoder(f"FT[{enc.key}]", ENCODERS[enc.key][0], jp, rp, chunk_long=chunk_long, model=m)


def _eval(name, score, thr, yte, seed, frac, n_res, n_pairs, key):
    pred = (score >= thr).astype(int)
    try: auc = roc_auc_score(yte, score)
    except ValueError: auc = np.nan
    return dict(seed=seed, fraction=frac, n_train_resumes=n_res, n_train_pairs=n_pairs, encoder=key, model=name,
                acc=accuracy_score(yte, pred), bal_acc=balanced_accuracy_score(yte, pred), auc=auc)


def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, fractions=FRACTIONS, epochs=3, chunk_long=True, seed_dir=None, out=OUT_DIR):
    os.makedirs(out, exist_ok=True)
    path = f"{out}/learning_curve_per_seed.csv"
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    done_keys = {(r["seed"], r["encoder"], r["fraction"]) for r in rows}
    sd = find_seed_dir(seed_dir); frozen = load_encoders(encoders, chunk_long=chunk_long); t0 = time.time()
    for seed in range(seeds):
        full_tr, te = load_seed_pairs(sd, seed); yte = te.label.values
        order = np.random.RandomState(1000 + seed).permutation(np.sort(full_tr.resume_id.unique()))
        for enc in frozen:
            for frac in fractions:
                if (seed, enc.key, frac) in done_keys: continue
                keep = order[: max(4, int(round(frac * len(order))))]
                sub = full_tr[full_tr.resume_id.isin(keep)]
                i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(sub, groups=sub.resume_id))
                tr, va = sub.iloc[i_tr].reset_index(drop=True), sub.iloc[i_va].reset_index(drop=True)
                ytr, yva = tr.label.values, va.label.values
                new = []
                tf = TfidfBackend(pd.concat([tr.resume_text, tr.job_text]))
                Xtr, _ = ft._pair_feats(tf, tr); Xte, _ = ft._pair_feats(tf, te)
                p = ft._head(Xtr, ytr, seed).predict_proba(Xte)[:, 1]
                new.append(_eval("TF-IDF head", p, 0.5, yte, seed, frac, len(keep), len(sub), enc.key))
                Xtr_e, _ = ft._pair_feats(enc, tr); Xva_e, cva_e = ft._pair_feats(enc, va); Xte_e, cte_e = ft._pair_feats(enc, te)
                p = ft._head(Xtr_e, ytr, seed).predict_proba(Xte_e)[:, 1]
                new.append(_eval("frozen head", p, 0.5, yte, seed, frac, len(keep), len(sub), enc.key))
                new.append(_eval("frozen cosine", cte_e, ft._best_threshold(cva_e, yva), yte, seed, frac, len(keep), len(sub), enc.key))
                fe = _finetuned_encoder(enc, tr, seed, epochs, chunk_long)
                Xtr_f, _ = ft._pair_feats(fe, tr); Xva_f, cva_f = ft._pair_feats(fe, va); Xte_f, cte_f = ft._pair_feats(fe, te)
                p = ft._head(Xtr_f, ytr, seed).predict_proba(Xte_f)[:, 1]
                new.append(_eval("fine-tuned head", p, 0.5, yte, seed, frac, len(keep), len(sub), enc.key))
                new.append(_eval("fine-tuned cosine", cte_f, ft._best_threshold(cva_f, yva), yte, seed, frac, len(keep), len(sub), enc.key))
                rows += new; pd.DataFrame(rows).to_csv(path, index=False)
                if enc.key != "stub":
                    import gc, torch; del fe; gc.collect(); torch.cuda.empty_cache()
                print(f"seed {seed} {enc.key} fraction {frac:.2f} ({len(keep)} resumes, {len(sub)} pairs) done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    S = R.groupby(["encoder", "model", "fraction"]).agg(n_train_resumes=("n_train_resumes", "mean"), n_seeds=("seed", "nunique"),
        acc=("acc", "mean"), acc_sd=("acc", "std"), auc=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
    S.round(4).to_csv(f"{out}/learning_curve_summary.csv", index=False)
    open(f"{out}/NOTE_learning_curve.txt", "w").write(
        "Nested resume subsets per seed; thresholds chosen on a validation split of the SAME subset; test = full 150 pairs of the seed. "
        "Seeds share resumes -> seed SDs are descriptive only. Weak labels (category equality).\n"
        "Each model is compared with the TF-IDF head at the SAME fraction. 'Crossing point' = smallest fraction where fine-tuned > TF-IDF head by more than the seed SD.\n")
    pd.set_option("display.width", 200)
    print(S[S.model.isin(["TF-IDF head", "frozen head", "fine-tuned head"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
