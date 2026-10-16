"""Review 1.1(b): leave-category-out evaluation.  Does the gain of fine-tuning transfer to job/resume categories the model never saw in training?

Design (all choices fixed before looking at results):
  * the 25 categories are split into 5 folds of 5 categories (random partition, `n_partitions` different partitions with seeds 0..n-1 -> 5 x n_partitions runs).
  * a fold holds out its 5 categories H completely:
        TRAIN : resumes with category not in H; each gets 1 positive job (same category) and 2 negatives (random job of another category NOT in H).
        TEST  : resumes with category in H;     each gets 1 positive job (same category, never seen in training) and 2 negatives (jobs of OTHER held-out categories),
                so every test job and every test resume belongs to an unseen category and no "seen category => not a match" shortcut exists.
  * a 20% resume-grouped validation split of the training pairs fixes the decision threshold (balanced accuracy); the test pairs never touch any choice.
  * models: TF-IDF cosine (zero-shot), TF-IDF head, frozen-encoder cosine and head (embeddings from results/embeddings, CPU), category classifiers (reference: they cannot
    predict an unseen category by construction), and, with --finetune on a GPU, fine-tuned encoders (cosine and head) trained ONLY on the training pairs of the fold.
Metrics: AUC, accuracy at the validation threshold, Hit@1 and MRR over the 3 candidates of each resume (random = 0.33 / 0.61), majority-class accuracy = 0.667.
Outputs: results/leave_category_out/{per_fold.csv, summary.csv, NOTE.txt}   (fine-tuned rows are added when --finetune is used; the file is resumable per fold/model)
Run CPU part:  python rq1_leave_category_out.py
Run GPU part:  python rq1_leave_category_out.py --finetune --encoders SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2     (Colab / Kaggle T4)
"""
import os, sys, hashlib, argparse, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")
SEED_DIR = "data/eval_v7/frozen_seeds"; OUT = "results/leave_category_out"
from emb_paths import EMB_DIR
FROZEN = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def pools():
    allp = pd.concat([pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv") for s in range(5)])
    res = allp.drop_duplicates("resume_id")[["resume_id", "resume_text", "resume_category"]].reset_index(drop=True)
    job = allp.drop_duplicates("job_id")[["job_id", "job_text", "job_category"]].reset_index(drop=True)
    return res, job


def build_fold(res, job, H, rng):
    H = set(H); by_cat = {c: g.reset_index(drop=True) for c, g in job.groupby("job_category")}
    def pick(cats, n):
        cats = [c for c in cats if c in by_cat]; out = []
        for _ in range(n):
            c = cats[rng.integers(len(cats))]; g = by_cat[c]; out.append(g.iloc[rng.integers(len(g))])
        return out
    rows = []
    for _, r in res.iterrows():
        inH = r.resume_category in H
        if r.resume_category not in by_cat: continue
        negc = [c for c in (H if inH else set(by_cat) - H) if c != r.resume_category]
        if len(negc) < 1: continue
        for j, lab in [(pick([r.resume_category], 1)[0], 1)] + [(j, 0) for j in pick(negc, 2)]:
            rows.append(dict(resume_id=r.resume_id, resume_text=r.resume_text, resume_category=r.resume_category, job_id=j.job_id, job_text=j.job_text, job_category=j.job_category, label=lab, split="test" if inH else "train"))
    return pd.DataFrame(rows)


def rank_metrics(df, s):
    d = df.assign(s=s); hit, rr = [], []
    for _, g in d.groupby("resume_id"):
        if (g.label == 1).sum() != 1: continue
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + 0.5 * (o == p).sum(); hit.append(float(k == 1)); rr.append(1 / k)
    return float(np.mean(hit)), float(np.mean(rr))


def best_thr(score, y):
    c = np.unique(score); c = np.concatenate([[c.min() - 1e-9], (c[1:] + c[:-1]) / 2, [c.max() + 1e-9]]) if len(c) > 1 else c
    return c[np.argmax([balanced_accuracy_score(y, score >= t) for t in c])]


def feats(U, V):
    cos = (np.sum(U * V, 1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + 1e-12)).reshape(-1, 1); return np.hstack([np.abs(U - V), cos]), cos.ravel()


def evaluate(name, tr, va, te, Utr, Vtr, Uva, Vva, Ute, Vte):
    rows = []; Xtr, ctr = feats(Utr, Vtr); Xva, cva = feats(Uva, Vva); Xte, cte = feats(Ute, Vte); y = te.label.values
    for tag, (s_va, s_te) in {"cosine (zero-shot)": (cva, cte)}.items():
        t = best_thr(s_va, va.label.values); rows.append((f"{name} {tag}", s_te, t))
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, tr.label.values)
    pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]; rows.append((f"{name} head", pte, best_thr(pva, va.label.values)))
    out = []
    for nm, s, t in rows:
        h, mrr = rank_metrics(te, s)
        out.append(dict(model=nm, auc=roc_auc_score(y, s), acc=accuracy_score(y, s >= t), bal_acc=balanced_accuracy_score(y, s >= t), hit1=h, mrr=mrr))
    return out


def category_ref(tr, te, tfidf=True):
    classes = sorted(set(tr.resume_category) | set(tr.job_category) | set(te.resume_category) | set(te.job_category))
    rtr = tr.drop_duplicates("resume_id"); jtr = tr.drop_duplicates("job_id")
    tv_r = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(rtr.resume_text); tv_j = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(jtr.job_text)
    mr = LogisticRegression(C=10, max_iter=2000).fit(tv_r.transform(rtr.resume_text), rtr.resume_category); mj = LogisticRegression(C=10, max_iter=2000).fit(tv_j.transform(jtr.job_text), jtr.job_category)
    def full(m, X):
        P = np.zeros((X.shape[0], len(classes))); idx = {c: i for i, c in enumerate(classes)}; pr = m.predict_proba(X)
        for k, c in enumerate(m.classes_): P[:, idx[c]] = pr[:, k]
        return P
    Pr, Pj = full(mr, tv_r.transform(te.resume_text)), full(mj, tv_j.transform(te.job_text)); s = (Pr * Pj).sum(1); y = te.label.values; h, mrr = rank_metrics(te, s)
    hard = (Pr.argmax(1) == Pj.argmax(1)).astype(int)
    return dict(model="Category classifiers (TF-IDF) [cannot predict unseen categories]", auc=roc_auc_score(y, s), acc=accuracy_score(y, hard), bal_acc=balanced_accuracy_score(y, hard), hit1=h, mrr=mrr)


def main(n_partitions=3, finetune=False, encoders=",".join(FROZEN), epochs=3):
    os.makedirs(OUT, exist_ok=True); res, job = pools(); cats = sorted(set(res.resume_category) & set(job.job_category))
    path = f"{OUT}/per_fold.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(); rows = done.to_dict("records") if len(done) else []
    done_keys = {(r["partition"], r["fold"], r["model"]) for r in rows}
    embs = {e: dict(zip(*(lambda z: (z["keys"], z["vectors"]))(np.load(f"{EMB_DIR}/{e}.npz", allow_pickle=True)))) for e in FROZEN if os.path.exists(f"{EMB_DIR}/{e}.npz")}
    ft_mod = None
    if finetune:
        import colab_v7b_finetune_hybrid as ft_mod
        from colab_common import ENCODERS, Encoder
    for part in range(n_partitions):
        perm = np.random.default_rng(100 + part).permutation(cats); folds = [perm[i::5] for i in range(5)]
        for k, H in enumerate(folds):
            d = build_fold(res, job, H, np.random.default_rng(1000 * part + k)); tr_all, te = d[d.split == "train"], d[d.split == "test"]
            gi = next(GroupShuffleSplit(1, test_size=0.2, random_state=part * 10 + k).split(tr_all, groups=tr_all.resume_id)); tr, va = tr_all.iloc[gi[0]], tr_all.iloc[gi[1]]
            meta = dict(partition=part, fold=k, heldout="; ".join(H), n_train_resumes=tr_all.resume_id.nunique(), n_test_resumes=te.resume_id.nunique(), n_test_pairs=len(te))
            new = []
            if any((part, k, m) not in done_keys for m in ["TF-IDF cosine (zero-shot)", "TF-IDF head"]):
                tv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=3000, ngram_range=(1, 2)).fit(pd.concat([tr.resume_text, tr.job_text]))
                M = lambda col, df: tv.transform(df[col]).toarray()
                new += evaluate("TF-IDF", tr, va, te, M("resume_text", tr), M("job_text", tr), M("resume_text", va), M("job_text", va), M("resume_text", te), M("job_text", te))
                new.append(category_ref(tr_all, te))
            for e, E in embs.items():
                if (part, k, f"{e} frozen head") in done_keys: continue
                g = lambda col, df: np.vstack([E[sha(t)] for t in df[col]])
                for r in evaluate(f"{e} frozen", tr, va, te, g("resume_text", tr), g("job_text", tr), g("resume_text", va), g("job_text", va), g("resume_text", te), g("job_text", te)): new.append(r)
            if finetune:
                for e in encoders.split(","):
                    if (part, k, f"{e} fine-tuned head") in done_keys: continue
                    model = ft_mod._finetune(e, tr, part * 10 + k, epochs, ft_mod.BATCH_SIZE, ft_mod.LEARNING_RATE)
                    enc = Encoder(f"FT[{e}]", ENCODERS[e][0], ENCODERS[e][1], ENCODERS[e][2], chunk_long=True, model=model)
                    g = lambda col, df, kind: enc.encode(list(df[col]), kind)
                    for r in evaluate(f"{e} fine-tuned", tr, va, te, g("resume_text", tr, "resume"), g("job_text", tr, "job"), g("resume_text", va, "resume"), g("job_text", va, "job"), g("resume_text", te, "resume"), g("job_text", te, "job")): new.append(r)
                    del model, enc
                    try:
                        import gc, torch; gc.collect(); torch.cuda.empty_cache()
                    except Exception: pass
            for r in new:
                if (part, k, r["model"]) not in done_keys: rows.append({**meta, **r})
            pd.DataFrame(rows).to_csv(path, index=False); print(f"partition {part} fold {k} done ({len(te)} test pairs, held-out: {meta['heldout']})", flush=True)
    per = pd.DataFrame(rows); s = per.groupby("model")[["auc", "acc", "bal_acc", "hit1", "mrr"]].agg(["mean", "std"]).round(3); s.columns = ["_".join(c) for c in s.columns]
    s = s.reset_index().sort_values("auc_mean", ascending=False); s["n_runs"] = per.groupby("model").size().reindex(s.model).values; s.to_csv(f"{OUT}/summary.csv", index=False)
    print(s.to_string(index=False))
    open(f"{OUT}/NOTE.txt", "w").write(open(__file__).read().split('"""')[1] + "\nRandom reference: AUC 0.50, accuracy at a fitted threshold is not comparable to 0.667 majority rate; Hit@1 0.33, MRR 0.61.\nRuns share resumes and jobs across folds/partitions: SDs are descriptive.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--finetune", action="store_true"); ap.add_argument("--encoders", default="SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2")
    ap.add_argument("--partitions", type=int, default=3); ap.add_argument("--epochs", type=int, default=3); a = ap.parse_args()
    main(n_partitions=a.partitions, finetune=a.finetune, encoders=a.encoders, epochs=a.epochs)
