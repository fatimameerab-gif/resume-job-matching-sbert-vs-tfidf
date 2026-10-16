"""
SCRIPT 9 (GPU, ~15-25 min on a Tesla T4): do the fine-tuned encoders transfer to TalentCLEF 2026 Task A (human-annotated, review point 16a)?

TalentCLEF 2026 Task A has NO training set (only a 10-query development and a 40-query test split; overview paper, Table 1), so 'fine-tune on the TalentCLEF
training data and test on its development set' is not possible. What can be done: fine-tune each encoder on OUR weak-label training pairs
(the training part of frozen seeds 0, 1 and 2; same recipe as script 5: CosineSimilarityLoss, 3 epochs, batch 8, lr 2e-5, 256 training tokens, AMP),
then rank the 472 resumes of every development query by the cosine of the fine-tuned encoder (chunk pooling, no TalentCLEF text or label is used for
any training step) and compare with the frozen encoder and TF-IDF on the human qrels. This is a zero-shot transfer test of the fine-tuned models.
Metrics: MAP, nDCG@100, MRR per query (same functions as the earlier Task A scripts); the unit is the query, n = 10, so every interval is descriptive.
HOW TO RUN (Kaggle / Colab, GPU on): same folder as for colab_human_label_finetune.py, plus the TalentCLEF folder data/taskA_en (queries, corpus, qrels.tsv),
data/eval_v7/frozen_seeds/pairs_seed{0,1,2}.csv, and the scripts colab_common.py, colab_v7b_finetune_hybrid.py, colab_v7b_taskA_chunked.py,
rq1_colab_taskA_sbert.py, representations.py.    python colab_taskA_finetuned_transfer.py     (quick check: --encoders SBERT-MiniLM-L6-v2 --seeds 0)
CPU check with no download:  python colab_taskA_finetuned_transfer.py --encoders stub
Output: results/taskA_finetuned/{per_query.csv, summary.csv, paired.csv, NOTE.txt}
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

OUT = "results/taskA_finetuned"; SEED_DIR = "data/eval_v7/frozen_seeds"
DEFAULT_ENCODERS = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,JobBERT-v2"
METRICS = ["map", "ndcg@100", "mrr"]


def rank_rows(label, pairs, R, Q, kind_scores=None):
    """Zero-shot cosine ranking of all resumes for each query; returns per-query metric rows."""
    from rq1_colab_taskA_sbert import _ranking_row
    rows = []
    for q in pd.unique(pairs.query_id):
        g = pairs[pairs.query_id == q]; u = Q[q]; s = np.array([R[r] @ u for r in g.resume_id]); y = g.label.values
        rows.append({**_ranking_row(label, q, int(y.sum()), y[np.argsort(-s, kind="stable")]), "model": label})
    return rows


def main(encoders=DEFAULT_ENCODERS, seeds=(0, 1, 2), epochs=3, out=OUT, taska_dir=None, chunk_long=True):
    from colab_common import load_taskA, load_encoders, ENCODERS, Encoder, TfidfBackend
    import colab_v7b_finetune_hybrid as fth
    os.makedirs(out, exist_ok=True)
    pairs = load_taskA(taska_dir)
    uj = pairs.drop_duplicates("query_id")[["query_id", "job_text"]]; ur = pairs.drop_duplicates("resume_id")[["resume_id", "resume_text"]]
    def vecs(b):
        return dict(zip(ur.resume_id, b.encode(ur.resume_text, "resume"))), dict(zip(uj.query_id, b.encode(uj.job_text, "job")))
    rows, t0 = [], time.time()
    tf = TfidfBackend(list(uj.job_text) + list(ur.resume_text)); R, Q = vecs(tf)
    rows += [{**r, "seed": -1, "kind": "TF-IDF"} for r in rank_rows("TF-IDF cosine", pairs, R, Q)]
    for enc in load_encoders(encoders, chunk_long=chunk_long):
        R, Q = vecs(enc)
        rows += [{**r, "seed": -1, "kind": "frozen"} for r in rank_rows(f"{enc.key} frozen cosine", pairs, R, Q)]
        if enc.key == "stub":
            continue
        for s in seeds:
            d = pd.read_csv(f"{SEED_DIR}/pairs_seed{s}.csv"); tr = d[d.split == "train"].reset_index(drop=True)
            m = fth._finetune(enc.key, tr, s, epochs, fth.BATCH_SIZE, fth.LEARNING_RATE)
            _, jp, rp = ENCODERS[enc.key]
            ft = Encoder(f"FT[{enc.key}]", ENCODERS[enc.key][0], jp, rp, chunk_long=chunk_long, model=m)
            R, Q = vecs(ft)
            rows += [{**r, "seed": s, "kind": "fine-tuned"} for r in rank_rows(f"{enc.key} fine-tuned cosine", pairs, R, Q)]
            del m, ft
            import gc, torch; gc.collect(); torch.cuda.empty_cache()
            print(f"{enc.key} seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    P = pd.DataFrame(rows); P.to_csv(f"{out}/per_query.csv", index=False)
    # summary: per model, mean over queries (fine-tuned: first mean over seeds per query), seed SD of the query-mean
    pq = P.groupby(["model", "kind", "query_id"])[METRICS].mean().reset_index()
    S = pq.groupby(["model", "kind"])[METRICS].mean()
    sd = P[P.kind == "fine-tuned"].groupby(["model", "seed"])[METRICS].mean().groupby("model").std().add_suffix("_seed_sd")
    S = S.join(sd, on="model").round(4).reset_index(); S.to_csv(f"{out}/summary.csv", index=False)
    # paired per-query differences (descriptive, n = 10 queries)
    wide = {m: pq.pivot(index="query_id", columns="model", values=m) for m in METRICS}; res = []
    for a in [c for c in wide["map"].columns if "fine-tuned" in c]:
        enc = a.replace(" fine-tuned cosine", "")
        for b in ("TF-IDF cosine", f"{enc} frozen cosine"):
            for m in METRICS:
                dv = (wide[m][a] - wide[m][b]).dropna().values; se = dv.std(ddof=1) / np.sqrt(len(dv)); t = stats.t.ppf(0.975, len(dv) - 1)
                res.append(dict(a=a, b=b, metric=m, mean_diff=dv.mean(), ci_lo=dv.mean() - t * se, ci_hi=dv.mean() + t * se, n_queries_better=int((dv > 0).sum()), n_queries=len(dv)))
    pd.DataFrame(res).round(4).to_csv(f"{out}/paired.csv", index=False)
    open(f"{out}/NOTE.txt", "w").write(
        "Zero-shot transfer of fine-tuned encoders to TalentCLEF 2026 Task A (development set, 10 queries x 472 resumes, human qrels). TalentCLEF Task A has no training set.\n"
        "Encoders were fine-tuned on the weak-label training pairs of frozen seeds %s (script 5 recipe, %d epochs); no TalentCLEF text or label was used for training.\n"
        "Ranking = cosine with chunk pooling. Unit = query (n = 10): intervals are descriptive (t interval over queries), not tests. Fine-tuned rows average the per-query metric over seeds.\n" % (list(seeds), epochs))
    pd.set_option("display.width", 200); print(S.to_string(index=False)); print(pd.DataFrame(res).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default=DEFAULT_ENCODERS); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--out", default=OUT); ap.add_argument("--taska-dir", default=None)
    a = ap.parse_args()
    main(a.encoders, tuple(int(x) for x in a.seeds.split(",")), a.epochs, a.out, a.taska_dir)
