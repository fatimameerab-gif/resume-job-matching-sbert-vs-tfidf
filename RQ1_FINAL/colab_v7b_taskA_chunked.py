import os, sys, time, types
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS (change these, then run) =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,JobBERT-v2"   # or "all" (adds e5, bge, gte) ; "SBERT-MiniLM-L6-v2" = quick check
CHUNK_MODES = ("truncate", "chunk")      # keep both: the difference between them is the point
TASKA_DIR = None                         # None = find it (or unzip it) automatically; else e.g. "/content/taskA_en"
SAVE_PAIR_SCORES = True                  # per-pair cosine/probability, for error analysis
OUT_DIR = "results/colab_v7b_taskA"
# ==================================================================================

from colab_common import ENCODERS, Encoder, TfidfBackend, load_taskA, load_st
from representations import build_pair_features
from rq1_colab_taskA_sbert import _ranking_row          # same ranking metrics as the earlier TaskA scripts

RANK_METRICS = ["map", "ndcg@10", "ndcg@100", "mrr", "r_precision", "recall@10"]


def loqo(pairs, R, Q, label):
    """R: resume_id -> vector, Q: query_id -> vector. Returns per-query rows (trained + zero-shot) and pair-level outputs."""
    r_all = np.array([R[i] for i in pairs.resume_id]); j_all = np.array([Q[i] for i in pairs.query_id])
    X, cos = build_pair_features(r_all, j_all, "absdiff_cos"); cos = np.ravel(cos)
    y, qids = pairs.label.values, pairs.query_id.values
    prob, trained, zero = np.zeros(len(pairs)), [], []
    for q in pd.unique(qids):
        te = qids == q
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1]; prob[te] = p
        yt = y[te]; n_rel = int(yt.sum())
        trained.append({**_ranking_row(label, q, n_rel, yt[np.argsort(-p)]), "kind": "trained"})
        zero.append({**_ranking_row(label, q, n_rel, yt[np.argsort(-cos[te])]), "kind": "zeroshot"})
    return trained + zero, prob, cos


def summary_row(label, model, mode, pairs, prob, cos, per_q):
    y, pred = pairs.label.values, (prob >= 0.5).astype(int)
    row = dict(model=model, mode=mode, accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
               auc_trained=roc_auc_score(y, prob), auc_zeroshot=roc_auc_score(y, cos))
    d = pd.DataFrame(per_q)
    for kind in ("trained", "zeroshot"):
        for m in RANK_METRICS:
            row[f"{m}_{kind}"] = d[d.kind == kind][m].mean()
    return row


def paired(per_q):
    """Per-query paired differences vs TF-IDF (n = 10 queries only: descriptive, Wilcoxon cannot go below p = 0.002)."""
    base = per_q[(per_q.model == "TF-IDF")].set_index(["kind", "query_id"])
    out = []
    for (model, mode), g in per_q[per_q.model != "TF-IDF"].groupby(["model", "mode"]):
        for kind in ("trained", "zeroshot"):
            for m in ("map", "ndcg@100", "mrr"):
                gg = g[g.kind == kind].set_index("query_id")[m]
                d = (gg - base.loc[kind][m].reindex(gg.index)).dropna().values
                se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
                try:
                    pw = stats.wilcoxon(d).pvalue
                except ValueError:
                    pw = np.nan
                out.append(dict(model=model, mode=mode, kind=kind, metric=m, mean_diff_vs_tfidf=d.mean(),
                                ci_lo=d.mean() - t * se, ci_hi=d.mean() + t * se, n_queries_better=int((d > 0).sum()),
                                n_queries=len(d), wilcoxon_p=pw))
    return pd.DataFrame(out)


def chunk_effect(per_q):
    out = []
    for model, g in per_q[per_q.model != "TF-IDF"].groupby("model"):
        for kind in ("trained", "zeroshot"):
            for m in ("map", "ndcg@100", "mrr"):
                a = g[(g["mode"] == "chunk") & (g.kind == kind)].set_index("query_id")[m]
                b = g[(g["mode"] == "truncate") & (g.kind == kind)].set_index("query_id")[m]
                if len(a) and len(b):
                    d = (a - b.reindex(a.index)).dropna().values
                    se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
                    out.append(dict(model=model, kind=kind, metric=m, chunk_minus_truncate=d.mean(),
                                    ci_lo=d.mean() - t * se, ci_hi=d.mean() + t * se, n_queries_better=int((d > 0).sum())))
    return pd.DataFrame(out)


def main(encoders=ENCODERS_TO_RUN, chunk_modes=CHUNK_MODES, taska_dir=TASKA_DIR, save_scores=SAVE_PAIR_SCORES, out=OUT_DIR):
    a = types.SimpleNamespace(encoders=encoders, chunk_modes=chunk_modes, taska_dir=taska_dir, save_scores=save_scores, out=out)
    os.makedirs(a.out, exist_ok=True)
    pairs = load_taskA(a.taska_dir)
    uj = pairs.drop_duplicates("query_id")[["query_id", "job_text"]]; ur = pairs.drop_duplicates("resume_id")[["resume_id", "resume_text"]]
    keys = ["stub"] if a.encoders == "stub" else (list(ENCODERS) if a.encoders == "all" else [k.strip() for k in a.encoders.split(",")])

    summ, per_q_all, scores = [], [], []

    def run(backend, model, mode):
        Q = dict(zip(uj.query_id, backend.encode(uj.job_text, "job"))); R = dict(zip(ur.resume_id, backend.encode(ur.resume_text, "resume")))
        label = f"{model}|{mode}"
        per_q, prob, cos = loqo(pairs, R, Q, label)
        for r in per_q:
            r.update(model=model, mode=mode)
        per_q_all.extend(per_q); summ.append(summary_row(label, model, mode, pairs, prob, cos, per_q))
        if a.save_scores:
            scores.append(pd.DataFrame(dict(model=model, mode=mode, query_id=pairs.query_id, resume_id=pairs.resume_id,
                                            label=pairs.label, cosine=cos, prob_trained=prob)))
        s = summ[-1]
        print(f"{model:24s} {mode:9s} acc={s['accuracy']:.3f} | trained MAP={s['map_trained']:.3f} nDCG@100={s['ndcg@100_trained']:.3f}"
              f" | zero-shot MAP={s['map_zeroshot']:.3f} nDCG@100={s['ndcg@100_zeroshot']:.3f}", flush=True)

    run(TfidfBackend(list(uj.job_text) + list(ur.resume_text)), "TF-IDF", "full-text")
    for key in keys:
        enc_models = {}
        for mode in a.chunk_modes:
            if key == "stub":
                enc = Encoder("stub", chunk_long=(mode == "chunk"))
            else:
                if "m" not in enc_models:                                  # load each model once, wrap it per mode
                    try:
                        enc_models["m"] = load_st(ENCODERS[key][0])
                    except Exception as e:
                        print(f"[SKIP] {key}: {type(e).__name__}: {e}", flush=True); break
                mid, jp, rp = ENCODERS[key]
                enc = Encoder(key, mid, jp, rp, chunk_long=(mode == "chunk"), model=enc_models["m"])
            run(enc, key, mode)
            print("   ", enc.report(), flush=True)

    per_q = pd.DataFrame(per_q_all)
    pd.DataFrame(summ).round(4).to_csv(f"{a.out}/taskA_summary.csv", index=False)
    per_q.to_csv(f"{a.out}/taskA_per_query.csv", index=False)
    paired(per_q).round(4).to_csv(f"{a.out}/taskA_paired_vs_tfidf.csv", index=False)
    chunk_effect(per_q).round(4).to_csv(f"{a.out}/taskA_chunk_effect.csv", index=False)
    if a.save_scores:
        pd.concat(scores, ignore_index=True).to_csv(f"{a.out}/taskA_pair_scores.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n=== summary (MAP, nDCG@100: zero-shot and trained) ===")
    print(pd.DataFrame(summ)[["model", "mode", "accuracy", "map_zeroshot", "ndcg@100_zeroshot", "map_trained", "ndcg@100_trained"]].round(3).to_string(index=False))
    print("\nWith only 10 queries every comparison is descriptive: read CIs and n_queries_better, not p-values.")


if __name__ == "__main__":
    main()
