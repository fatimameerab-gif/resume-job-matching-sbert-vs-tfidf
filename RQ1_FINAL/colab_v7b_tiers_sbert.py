import os, sys, time, types
import numpy as np, pandas as pd
from scipy import stats, sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS (change these, then run) =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,bge-base-en-v1.5"   # or "SBERT-MiniLM-L6-v2" for a quick check, or "all"
SEEDS = 5                                # 5 seeds (seeds 0-4)
CHUNK_LONG_TEXTS = True                  # True = long texts are split in chunks and averaged (no truncation); False = old behaviour
TRAIN_TIER = "random_other_category"     # heads are trained on this tier only (keep the default)
OUT_DIR = "results/colab_v7b_chunked"    # different folder, so the earlier truncated results are not overwritten
# ==================================================================================

from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from eval_protocol_v7 import fit_hard_index
from representations import tfidf_features, build_pair_features
from tier_eval_v7b import build_tier_pairs, _lr, _job_prior_auc, _job_length_auc, NEG_REUSE_CAP
from colab_common import load_encoders, DenseIndex

METRICS = ["full_acc", "full_bal_acc", "full_auc", "cosLR_auc", "zeroshot_cos_auc", "job_prior_auc", "job_length_auc"]


class TfidfHeads:
    name = "TF-IDF"

    def __init__(self, tr):
        self.tr, one = tr, tr.iloc[:2]
        Xtr = tfidf_features(tr.resume_text, tr.job_text, one.resume_text, one.job_text)[0]
        Ctr = tfidf_features(tr.resume_text, tr.job_text, one.resume_text, one.job_text, feature_mode="cos_only")[0]
        self.full, self.cos = _lr(Xtr, tr.label), _lr(Ctr, tr.label)

    def scores(self, te):
        tr = self.tr
        Xte = tfidf_features(tr.resume_text, tr.job_text, te.resume_text, te.job_text)[1]
        _, Cte, _, cte, _ = tfidf_features(tr.resume_text, tr.job_text, te.resume_text, te.job_text, feature_mode="cos_only")
        return (self.full.predict(Xte), self.full.predict_proba(Xte)[:, 1],
                self.cos.predict_proba(Cte)[:, 1], np.ravel(cte))


class EncHeads:
    def __init__(self, enc, tr):
        self.enc, self.name = enc, enc.key
        U, V = enc.encode(tr.resume_text, "resume"), enc.encode(tr.job_text, "job")
        self.full = _lr(build_pair_features(U, V, "absdiff_cos")[0], tr.label)
        self.cos = _lr(build_pair_features(U, V, "cos_only")[0], tr.label)

    def scores(self, te):
        U, V = self.enc.encode(te.resume_text, "resume"), self.enc.encode(te.job_text, "job")
        Xte, cte = build_pair_features(U, V, "absdiff_cos")
        Cte = build_pair_features(U, V, "cos_only")[0]
        return (self.full.predict(Xte), self.full.predict_proba(Xte)[:, 1],
                self.cos.predict_proba(Cte)[:, 1], np.ravel(cte))


class ConsensusIndex:
    """Mining index whose cosine = weighted mean of the per-model cosines, weights chosen so every model counts EQUALLY.
    Why: raw cosines have different spreads (embedding cosines vary much more than TF-IDF cosines), so a plain average is
    dominated by one model and the tier is NOT equally hard for all. Blocks are scaled by w_i = 1/sqrt(spread_i); the
    cosine of the concatenation is then sum(w_i^2 * cos_i)/sum(w_i^2), i.e. the same ranking as averaging z-scores."""
    def __init__(self, tfidf_vec, encs, weights):
        self.v, self.encs, self.w = tfidf_vec, encs, weights

    def transform(self, texts):
        texts = list(texts)
        blocks = [self.w[0] * self.v.transform(texts)]
        blocks += [self.w[i + 1] * sparse.csr_matrix(e.encode(texts, "resume")) for i, e in enumerate(self.encs)]
        return sparse.hstack(blocks).tocsr()


def _spread(R, J):
    """Mean over resumes of the std of cos(resume, job) across the job pool (rows are L2-normalised)."""
    S = (sparse.csr_matrix(R) @ sparse.csr_matrix(J).T).toarray()
    return float(S.std(axis=1).mean())


def _row(model, tier, seed, tr, te, pred, sf, sc, cte):
    return dict(seed=seed, tier=tier, model=model, n_test=len(te),
                full_acc=accuracy_score(te.label, pred), full_bal_acc=balanced_accuracy_score(te.label, pred),
                full_auc=roc_auc_score(te.label, sf), cosLR_auc=roc_auc_score(te.label, sc),
                zeroshot_cos_auc=roc_auc_score(te.label, cte),
                job_prior_auc=_job_prior_auc(tr, te), job_length_auc=_job_length_auc(te))


def summarise(per_seed):
    base = per_seed[per_seed.tier == "random_other_category"].set_index(["model", "seed"])
    out = []
    for (tier, model), g in per_seed.groupby(["tier", "model"]):
        g = g.set_index("seed")
        row = dict(tier=tier, model=model, self_mined=(tier == f"hard[{model}]"), n_seeds=len(g), n_test=g.n_test.mean())
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else np.nan
            if tier != "random_other_category" and len(g) > 2:
                b = base.loc[model][m].reindex(g.index); d = (g[m] - b).dropna().values
                if len(d) > 2:
                    se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.975, len(d) - 1)
                    row[f"{m}_diff_vs_random"] = d.mean()
                    row[f"{m}_diff_lo"], row[f"{m}_diff_hi"] = d.mean() - t * se, d.mean() + t * se
        out.append(row)
    return pd.DataFrame(out)


def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, train_tier=TRAIN_TIER, out=OUT_DIR, chunk_long=CHUNK_LONG_TEXTS):
    a = types.SimpleNamespace(encoders=encoders, seeds=seeds, train_tier=train_tier, out=out, chunk_long=chunk_long)
    os.makedirs(a.out, exist_ok=True)
    encs = load_encoders(a.encoders, chunk_long=a.chunk_long)

    resumes, pool = build_deduped_resumes(), build_deduped_job_pool()
    hv, hm = fit_hard_index(pool)
    jobs_txt, res_txt = pool["text_clean"].fillna("").tolist(), resumes["text_clean"].tolist()
    idx = {"TF-IDF": (hv, hm)}
    dense_mats = []
    for e in encs:                                                          # one mining index per encoder
        e.encode(res_txt, "resume")                                         # warm cache (mining looks up resumes one by one)
        mat = sparse.csr_matrix(e.encode(jobs_txt, "job")); dense_mats.append(mat)
        idx[e.key] = (DenseIndex(e), mat)
    spreads = [_spread(hv.transform(res_txt), hm)] + [
        _spread(e.encode(res_txt, "resume"), m) for e, m in zip(encs, dense_mats)]
    w = [1.0 / np.sqrt(sp) for sp in spreads]                               # equal say for every model in the consensus tier
    cons_mat = sparse.hstack([w[0] * hm] + [w[i + 1] * dense_mats[i] for i in range(len(encs))]).tocsr()
    idx["consensus"] = (ConsensusIndex(hv, encs, w), cons_mat)
    # family-balanced consensus: TF-IDF (lexical family) and the dense encoders (dense family) get HALF of the total weight each,
    # the dense half is split equally among the encoders. The plain 'consensus' above has 1 TF-IDF vs n dense -> leans dense.
    fam = [0.5] + [0.5 / len(encs)] * len(encs)
    wf = [float(np.sqrt(f / sp)) for f, sp in zip(fam, spreads)]
    cons_f_mat = sparse.hstack([wf[0] * hm] + [wf[i + 1] * dense_mats[i] for i in range(len(encs))]).tocsr()
    idx["consensus_family"] = (ConsensusIndex(hv, encs, wf), cons_f_mat)
    print("cosine spread per model (TF-IDF first):", [round(x, 3) for x in spreads],
          "-> consensus weights w_i^2:", [round(x * x, 1) for x in w], flush=True)
    print("mining indexes:", list(idx), "| heads trained on:", a.train_tier, flush=True)

    rows, t0 = [], time.time()
    for seed in range(a.seeds):
        tiers = build_tier_pairs(seed, resumes, pool, hv, hm, NEG_REUSE_CAP)
        tiers["hard[TF-IDF]"] = tiers.pop("hard_mined")
        for name, (v, m) in idx.items():
            if name != "TF-IDF":
                tiers[f"hard[{name}]"] = build_tier_pairs(seed, resumes, pool, v, m, NEG_REUSE_CAP)["hard_mined"]
        train_df = tiers[a.train_tier]; tr = train_df[train_df.split == "train"]
        heads = [TfidfHeads(tr)] + [EncHeads(e, tr) for e in encs]           # trained ONCE, tested on every tier
        for tier, df in tiers.items():
            te = df[df.split == "test"]
            for h in heads:
                rows.append(_row(h.name, tier, seed, tr, te, *h.scores(te)))
        print(f"seed {seed} done  ({time.time() - t0:.0f}s)", flush=True)

    per_seed = pd.DataFrame(rows); per_seed.to_csv(f"{a.out}/tiers_sbert_per_seed.csv", index=False)
    summ = summarise(per_seed); summ.round(4).to_csv(f"{a.out}/tiers_sbert_summary.csv", index=False)
    bad = per_seed[(per_seed.cosLR_auc - per_seed.zeroshot_cos_auc).abs() > 1e-6]
    open(f"{a.out}/NOTE_tiers_how_to_read.txt", "w").write(
        "Heads were trained ONLY on '%s' and tested on every tier. Encoders: %s\n"
        "Read tier difficulty from zeroshot_cos_auc. A model's own hard[<model>] tier is biased AGAINST it: compare models on\n"
        "hard tiers they did NOT mine, and on hard[consensus]. full_acc/full_auc on mined tiers can still contain a job-identity\n"
        "effect; job_prior_auc (computed against the training set used) shows whether it is learnable.\n"
        "hard[consensus] weights every model equally (blocks scaled by 1/sqrt(within-resume cosine spread)); cosine spreads: %s\n"
        "hard[consensus_family] gives TF-IDF and the dense encoders HALF of the weight each (use this one as the fair shared tier;\n"
        "plain hard[consensus] has 1 TF-IDF vs several dense models and leans against the dense family).\n"
        "Rows where the cosine head disagrees with zero-shot AUC (sign flip): %d\n" % (a.train_tier, " | ".join(e.report() for e in encs), [round(x, 3) for x in spreads], len(bad)))
    for e in encs:
        print(e.report())
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    for m in ("zeroshot_cos_auc_mean", "full_auc_mean", "job_prior_auc_mean"):
        print(f"\n=== {m} (rows = tier, cols = model) ===")
        print(summ.pivot(index="tier", columns="model", values=m).round(3).to_string())
    if len(bad):
        print(f"\n[WARNING] {len(bad)} rows where the cosine head flipped sign vs zero-shot -- check the training tier.")


if __name__ == "__main__":
    main()
