import os, hashlib, argparse, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")
from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from eval_protocol_v7 import grouped_split_resume_and_job_text, build_pairs_v7
from representations import REPRESENTATIONS
from stats_utils import friedman_nemenyi, holm_bonferroni
OUT = "results/design_b_dense"; N = 20
from emb_paths import EMB_DIR
ENC = ["SBERT-MiniLM-L6-v2", "SBERT-mpnet-base-v2", "e5-base-v2", "bge-base-en-v1.5", "gte-base", "JobBERT-v2"]
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()


def feats(U, V):
    cos = (np.sum(U * V, 1) / (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + 1e-12)).reshape(-1, 1); return np.hstack([np.abs(U - V), cos])


def head_eval(name, Xtr, ytr, Xte, yte, ntr, nte, split):
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    return dict(split=split, model=name, acc=accuracy_score(yte, pred), bal_acc=balanced_accuracy_score(yte, pred), auc=roc_auc_score(yte, p) if len(set(yte)) == 2 else np.nan, n_train=ntr, n_test=nte)


def main(finetune=False, encoders="SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2", epochs=3):
    os.makedirs(OUT, exist_ok=True); path = f"{OUT}/per_split.csv"; rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []; done = {(r["split"], r["model"]) for r in rows}
    R, J = build_deduped_resumes(), build_deduped_job_pool(); E = {e: dict(zip(*(lambda z: (z["keys"], z["vectors"]))(np.load(f"{EMB_DIR}/{e}.npz", allow_pickle=True)))) for e in ENC if os.path.exists(f"{EMB_DIR}/{e}.npz")}
    if finetune:
        import colab_v7b_finetune_hybrid as fth
        from colab_common import ENCODERS, Encoder
    for seed in range(N):
        tr_r, va_r, te_r, tr_j, va_j, te_j = grouped_split_resume_and_job_text(R, J, seed=seed)
        tr = build_pairs_v7(tr_r, tr_j, "train", np.random.RandomState(100 + seed)); te = build_pairs_v7(te_r, te_j, "test", np.random.RandomState(200 + seed))
        if len(te) < 10 or te.label.nunique() < 2 or tr.label.nunique() < 2: continue
        new = []
        if (seed, "TF-IDF head") not in done:
            o = REPRESENTATIONS["TF-IDF"](tr.resume_text, tr.job_text, te.resume_text, te.job_text); new.append(head_eval("TF-IDF head", o[0], tr.label.values, o[1], te.label.values, len(tr), len(te), seed))
        for e in E:
            if (seed, f"{e} frozen head") in done: continue
            g = lambda col, df: np.vstack([E[e][sha(t)] for t in df[col]]); new.append(head_eval(f"{e} frozen head", feats(g("resume_text", tr), g("job_text", tr)), tr.label.values, feats(g("resume_text", te), g("job_text", te)), te.label.values, len(tr), len(te), seed))
        if finetune:
            for e in encoders.split(","):
                if (seed, f"{e} fine-tuned head") in done: continue
                model = fth._finetune(e, tr, seed, epochs, fth.BATCH_SIZE, fth.LEARNING_RATE); enc = Encoder(f"FT[{e}]", ENCODERS[e][0], ENCODERS[e][1], ENCODERS[e][2], chunk_long=True, model=model)
                g = lambda col, df, kind: enc.encode(list(df[col]), kind)
                new.append(head_eval(f"{e} fine-tuned head", feats(g("resume_text", tr, "resume"), g("job_text", tr, "job")), tr.label.values, feats(g("resume_text", te, "resume"), g("job_text", te, "job")), te.label.values, len(tr), len(te), seed))
                del model, enc
                try:
                    import gc, torch; gc.collect(); torch.cuda.empty_cache()
                except Exception: pass
        rows += new; pd.DataFrame(rows).to_csv(path, index=False); print("split", seed, "done", flush=True)
    P = pd.DataFrame(rows); S = P.groupby("model").agg(n_splits=("split", "nunique"), acc=("acc", "mean"), acc_sd=("acc", "std"), bal_acc=("bal_acc", "mean"), auc=("auc", "mean"), auc_sd=("auc", "std"), mean_test_pairs=("n_test", "mean")).round(4).reset_index()
    S.sort_values("acc", ascending=False).to_csv(f"{OUT}/summary.csv", index=False)
    # Nadeau-Bengio corrected paired t vs TF-IDF head, Holm
    base = P[P.model == "TF-IDF head"].set_index("split"); out = []
    for m, g in P.groupby("model"):
        if m == "TF-IDF head": continue
        g = g.set_index("split"); idx = g.index.intersection(base.index)
        for met in ("acc", "auc"):
            d = (g.loc[idx, met] - base.loc[idx, met]).values; k = len(d); ratio = float(np.mean(g.loc[idx, "n_test"] / g.loc[idx, "n_train"])); sd = d.std(ddof=1)
            t = d.mean() / (np.sqrt(1 / k + ratio) * sd) if sd > 0 else np.nan; p = 2 * stats.t.sf(abs(t), k - 1) if sd > 0 else np.nan; h = stats.t.ppf(0.975, k - 1) * np.sqrt(1 / k + ratio) * sd
            out.append(dict(model=m, metric=met, k_splits=k, mean_diff_vs_tfidf=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h, t=t, p_nadeau_bengio=p, splits_better=int((d > 0).sum())))
    V = pd.DataFrame(out)
    for met in ("acc", "auc"):
        mk = V.metric == met; V.loc[mk, "p_holm"] = holm_bonferroni(V.loc[mk, "p_nadeau_bengio"].fillna(1).values)[0]
    V.round(4).to_csv(f"{OUT}/vs_tfidf_nadeau_bengio.csv", index=False)
    full = P.pivot(index="split", columns="model", values="acc").dropna()
    for met in ("acc", "auc"):
        Z = P.pivot(index="split", columns="model", values=met).dropna()
        if Z.shape[0] >= 5 and Z.shape[1] >= 3:
            st, p, nem = friedman_nemenyi({c: Z[c].values for c in Z.columns}); nem.round(4).to_csv(f"{OUT}/nemenyi_{met}.csv")
            pd.DataFrame([dict(metric=met, friedman_chi2=st, p=p, n_splits=Z.shape[0], n_models=Z.shape[1], mean_ranks=dict(Z.rank(axis=1, ascending=False).mean().round(2)))]).to_csv(f"{OUT}/friedman_{met}.csv", index=False)
    pd.set_option("display.width", 250); print(S.sort_values("acc", ascending=False).to_string(index=False)); print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--finetune", action="store_true"); ap.add_argument("--encoders", default="SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2"); a = ap.parse_args(); main(a.finetune, a.encoders)
