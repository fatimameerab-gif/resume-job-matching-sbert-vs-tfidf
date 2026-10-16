import os, time, types
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS (change these, then run) =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2"   # quick check: "SBERT-MiniLM-L6-v2"
SEEDS = 5                                # seeds 0-4 (same 5 frozen files as the other scripts); quick check = 1
EPOCHS = 3                               # fine-tuning epochs (fixed in advance, not tuned on test)
BATCH_SIZE = 8                           # smaller batch = less GPU memory (was 16)
TRAIN_MAX_SEQ_LENGTH = 256               # fine-tuning sees at most this many tokens per text (inference still chunk-pools everything)
USE_AMP = True                           # mixed precision: about half the memory
LEARNING_RATE = 2e-5
FINETUNE = True                          # False = skip fine-tuning (frozen + hybrid only; fast)
CHUNK_LONG_TEXTS = True                  # True = chunk-pool long texts at inference (no truncation)
SEED_DIR = None                          # folder holding pairs_seed0.csv ...; None = find it automatically
N_BOOT = 1000                            # bootstrap resamples for the paired CI
OUT_DIR = "results/colab_v7b_finetune"
# ==================================================================================

from representations import build_pair_features
from colab_common import load_encoders, find_seed_dir, load_seed_pairs, TfidfBackend, ENCODERS, Encoder, load_st

K_RRF = 60


# ---------------------------------------------------------------------------------------------------------------------
def _head(Xtr, ytr, seed):
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(Xtr, ytr)


def _pair_feats(backend, df):
    R, J = backend.encode(df.resume_text.tolist(), "resume"), backend.encode(df.job_text.tolist(), "job")
    return build_pair_features(R, J, "absdiff_cos")[0], build_pair_features(R, J, "cos_only")[0].ravel()


def _best_threshold(score, y):
    """Threshold maximising accuracy on VALIDATION (ties -> the one closest to the median score)."""
    cand = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    accs = np.array([accuracy_score(y, (score >= t).astype(int)) for t in cand])
    best = cand[accs == accs.max()]
    return float(best[np.argmin(np.abs(best - np.median(score)))])


def _rank01(x):
    return stats.rankdata(x) / len(x)


def _rrf(*scores):
    out = np.zeros(len(scores[0]))
    for s in scores:
        out += 1.0 / (K_RRF + stats.rankdata(-np.asarray(s)))
    return out


class Scored:
    """Scores of one model on validation and test + the rule that turns a score into a 0/1 prediction."""
    def __init__(self, name, val, test, thr):
        self.name, self.val, self.test, self.thr = name, np.asarray(val, float), np.asarray(test, float), thr

    def pred(self):
        return (self.test >= self.thr).astype(int)


def _scored_head(name, clf, Xva, Xte):
    return Scored(name, clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1], 0.5)


def _scored_cos(name, cva, cte, yva):
    return Scored(name, cva, cte, _best_threshold(cva, yva))


def _hybrid_weighted(name, a, b, yva):
    """w*a + (1-w)*b on probabilities, w chosen on validation (grid 0..1, ties -> 0.5)."""
    grid = np.linspace(0, 1, 11)
    accs = [accuracy_score(yva, ((w * a.val + (1 - w) * b.val) >= 0.5).astype(int)) for w in grid]
    best = grid[np.array(accs) == max(accs)]
    w = float(best[np.argmin(np.abs(best - 0.5))])
    sc = Scored(f"{name} (weighted)", w * a.val + (1 - w) * b.val, w * a.test + (1 - w) * b.test, 0.5)
    sc.w = w                                  # the chosen weight is stored as a column, NOT in the model name (a name with the weight split the rows)
    return sc, w


def _hybrid_rrf(name, a, b, yva):
    sv, st = _rrf(a.val, b.val), _rrf(a.test, b.test)
    # validation RRF scores are on the validation set's own ranks; the threshold is a fraction of the max possible score
    return Scored(f"{name} (RRF)", sv / sv.max(), st / st.max(), _best_threshold(sv / sv.max(), yva))


def _metrics(sc, yte):
    p = sc.pred()
    try:
        auc = roc_auc_score(yte, sc.test)
    except ValueError:
        auc = np.nan
    return dict(acc=accuracy_score(yte, p), bal_acc=balanced_accuracy_score(yte, p), f1=f1_score(yte, p, zero_division=0), auc=auc)


FT_MODE = os.environ.get("FT_MODE", "manual")      # "manual" (default) or "fit" (library model.fit with automatic fallback)


# ---------------------------------------------------------------------------------------------------------------------
def _finetune(enc_key, tr, seed, epochs, batch, lr):
    """Returns a fine-tuned SentenceTransformer (CosineSimilarityLoss on (resume, job, 1/0))."""
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_st(ENCODERS[enc_key][0])
    orig_len = model.max_seq_length
    model.max_seq_length = min(orig_len, TRAIN_MAX_SEQ_LENGTH)
    jp, rp = ENCODERS[enc_key][1], ENCODERS[enc_key][2]
    ex = [InputExample(texts=[rp + r, jp + j], label=float(l)) for r, j, l in zip(tr.resume_text, tr.job_text, tr.label)]
    dl = DataLoader(ex, shuffle=True, batch_size=batch)
    loss = losses.CosineSimilarityLoss(model)
    if FT_MODE == "manual":                  # DEFAULT (pinned rerun): one explicit loop for every encoder, no dependence on model.fit()/datasets/router_mapping of a given library version
        _manual_loop(model, tr, jp, rp, epochs, batch, lr)
        model.max_seq_length = orig_len
        return model
    try:
        model.fit(train_objectives=[(dl, loss)], epochs=epochs, warmup_steps=max(1, int(0.1 * len(dl) * epochs)),
                  optimizer_params={"lr": lr}, show_progress_bar=False, use_amp=USE_AMP)
    except (TypeError, ImportError) as e:    # some sentence-transformers versions break model.fit() for Asym models (JobBERT-v2): router_mapping error
        print(f"  [note] model.fit() failed ({e}); using the manual training loop (same loss, optimizer, schedule)", flush=True)
        _manual_loop(model, tr, jp, rp, epochs, batch, lr)
    model.max_seq_length = orig_len          # restore: chunk pooling at inference uses the model's real limit
    return model


def _manual_loop(model, tr, jp, rp, epochs, batch, lr):
    """Same objective as CosineSimilarityLoss: MSE(cosine(resume, job), label); AdamW, 10% linear warm-up, AMP, grad-clip 1.0."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    R = [rp + r for r in tr.resume_text]; J = [jp + j for j in tr.job_text]
    y = torch.tensor(tr.label.values, dtype=torch.float)
    n = len(R); steps = int(np.ceil(n / batch)) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * steps)), steps)
    use_amp = (dev == "cuda") and USE_AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    to_dev = lambda f: {k: v.to(dev) for k, v in f.items() if torch.is_tensor(v)}
    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            fa, fb = to_dev(model.tokenize([R[k] for k in idx])), to_dev(model.tokenize([J[k] for k in idx]))
            with torch.autocast(device_type="cuda", enabled=use_amp):
                ea, eb = model(fa)["sentence_embedding"], model(fb)["sentence_embedding"]
                l = torch.nn.functional.mse_loss(torch.nn.functional.cosine_similarity(ea.float(), eb.float()), y[idx].to(dev))
            opt.zero_grad(); scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    model.eval()


def _paired_vs_tfidf(pred_a, pred_b, y, resume_ids, n_boot, rng):
    """pred_a = candidate, pred_b = TF-IDF. Exact McNemar (two-sided) + resume-clustered bootstrap CI of acc_a - acc_b (pp)."""
    ca, cb = (pred_a == y), (pred_b == y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())          # b: only candidate right, c: only TF-IDF right
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0
    df = pd.DataFrame({"r": resume_ids, "d": ca.astype(float) - cb.astype(float)})
    g = df.groupby("r")["d"].agg(["sum", "count"])
    sums, cnts = g["sum"].values, g["count"].values
    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, len(g), len(g))
        boots.append(100 * sums[i].sum() / cnts[i].sum())
    return dict(diff_pp=100 * (ca.mean() - cb.mean()), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                only_candidate_right=b, only_tfidf_right=c, mcnemar_p=p)


def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, epochs=EPOCHS, finetune=FINETUNE, chunk_long=CHUNK_LONG_TEXTS,
         seed_dir=SEED_DIR, out=OUT_DIR, n_boot=N_BOOT):
    a = types.SimpleNamespace(encoders=encoders, seeds=seeds, epochs=epochs, finetune=finetune, chunk_long=chunk_long,
                              seed_dir=seed_dir, out=out, n_boot=n_boot)
    os.makedirs(a.out, exist_ok=True)
    sd = find_seed_dir(a.seed_dir)
    frozen = load_encoders(a.encoders, chunk_long=a.chunk_long)
    rng = np.random.RandomState(0)
    rows, paired, preds, t0 = [], [], [], time.time()

    for seed in range(a.seeds):
        full_tr, te = load_seed_pairs(sd, seed)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        i_tr, i_va = next(gss.split(full_tr, groups=full_tr.resume_id))
        tr, va = full_tr.iloc[i_tr].reset_index(drop=True), full_tr.iloc[i_va].reset_index(drop=True)
        ytr, yva, yte = tr.label.values, va.label.values, te.label.values

        tf = TfidfBackend(pd.concat([tr.resume_text, tr.job_text]))
        Xtr, _ = _pair_feats(tf, tr); Xva, cva = _pair_feats(tf, va); Xte, cte = _pair_feats(tf, te)
        tf_head = _scored_head("TF-IDF head", _head(Xtr, ytr, seed), Xva, Xte)
        models = [tf_head, _scored_cos("TF-IDF cosine (zero-shot)", cva, cte, yva)]

        for enc in frozen:
            Xtr_e, _ = _pair_feats(enc, tr); Xva_e, cva_e = _pair_feats(enc, va); Xte_e, cte_e = _pair_feats(enc, te)
            fz_head = _scored_head(f"{enc.key} frozen head", _head(Xtr_e, ytr, seed), Xva_e, Xte_e)
            models += [fz_head, _scored_cos(f"{enc.key} frozen cosine (zero-shot)", cva_e, cte_e, yva)]
            for h, w in (_hybrid_weighted(f"Hybrid TF-IDF + {enc.key} frozen", tf_head, fz_head, yva),):
                models.append(h)
            models.append(_hybrid_rrf(f"Hybrid TF-IDF + {enc.key} frozen", tf_head, fz_head, yva))

            if a.finetune and enc.key != "stub":
                ft_model = _finetune(enc.key, tr, seed, a.epochs, BATCH_SIZE, LEARNING_RATE)
                _, jp, rp = ENCODERS[enc.key][0], ENCODERS[enc.key][1], ENCODERS[enc.key][2]
                ft = Encoder(f"FT[{enc.key}]", ENCODERS[enc.key][0], jp, rp, chunk_long=a.chunk_long, model=ft_model)
                Xtr_f, _ = _pair_feats(ft, tr); Xva_f, cva_f = _pair_feats(ft, va); Xte_f, cte_f = _pair_feats(ft, te)
                ft_head = _scored_head(f"{enc.key} fine-tuned head", _head(Xtr_f, ytr, seed), Xva_f, Xte_f)
                models += [_scored_cos(f"{enc.key} fine-tuned cosine", cva_f, cte_f, yva), ft_head]
                models.append(_hybrid_weighted(f"Hybrid TF-IDF + {enc.key} fine-tuned", tf_head, ft_head, yva)[0])
                models.append(_hybrid_rrf(f"Hybrid TF-IDF + {enc.key} fine-tuned", tf_head, ft_head, yva))
                del ft_model, ft
                import gc, torch; gc.collect(); torch.cuda.empty_cache()   # free GPU memory before the next model

        for m in models:
            rows.append(dict(seed=seed, model=m.name, n_test=len(te), threshold=m.thr, fusion_w_tfidf=getattr(m, "w", np.nan), **_metrics(m, yte)))
            preds.append(pd.DataFrame(dict(seed=seed, model=m.name, resume_id=te.resume_id.values, job_id=te.job_id.values,
                                           resume_category=te.resume_category.values, job_category=te.job_category.values,
                                           negative_reason=te.negative_reason.values, label=yte, score=m.test, pred=m.pred(),
                                           correct=(m.pred() == yte).astype(int))))
            if m.name != tf_head.name:
                paired.append(dict(seed=seed, model=m.name, **_paired_vs_tfidf(m.pred(), tf_head.pred(), yte, te.resume_id.values, a.n_boot, rng)))
        print(f"seed {seed} done  ({time.time() - t0:.0f}s)", flush=True)

    per_seed = pd.DataFrame(rows); per_seed.to_csv(f"{a.out}/finetune_per_seed.csv", index=False)
    summ = per_seed.groupby("model").agg(n_seeds=("seed", "nunique"), acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                         bal_acc_mean=("bal_acc", "mean"), f1_mean=("f1", "mean"), auc_mean=("auc", "mean"),
                                         auc_sd=("auc", "std")).reset_index().sort_values("acc_mean", ascending=False)
    summ.round(4).to_csv(f"{a.out}/finetune_summary.csv", index=False)
    pd.DataFrame(paired).round(4).to_csv(f"{a.out}/paired_vs_tfidf.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(f"{a.out}/pair_predictions.csv", index=False)
    open(f"{a.out}/NOTE_finetune.txt", "w").write(
        "Thresholds / fusion weights were chosen on a validation split (by resume) of the training pairs, never on test.\n"
        "Fine-tuning: CosineSimilarityLoss on (resume, job, label), %d epochs, batch %d, lr %g; training texts are cut at 256 tokens (or the model limit if smaller), mixed precision, "
        "inference uses chunk pooling (%s).\nThe 5 seeds re-use the same resumes: do NOT treat seed-level t-intervals as "
        "independent; use paired_vs_tfidf.csv per seed. Labels are weak (resume category == job category).\n"
        "RRF rows: the validation-chosen threshold does not transfer (RRF scores are rank-based and the validation and test sets differ in size), "
        "so use their AUC only, not their accuracy.\n"
        % (a.epochs, BATCH_SIZE, LEARNING_RATE, "ON" if a.chunk_long else "OFF"))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n=== mean over seeds (test) ===")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
