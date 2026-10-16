import os, time, zlib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import perturbations as P
SEED_DIR = "data/eval_v7/frozen_seeds"
SETTINGS = [("esco_all", 0.5), ("esco_all", 1.0), ("esco_multiword", 0.5), ("esco_multiword", 1.0), ("typo", 0.05), ("typo", 0.10), ("typo", 0.20), ("reorder", 1.0)]
SIDES = ("resume_only", "job_only", "both")


def _rank(te, s):
    d = te[["resume_id", "label"]].assign(s=s); hit, rr = [], []
    for _, g in d.groupby("resume_id"):
        if (g.label == 1).sum() != 1: continue
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + 0.5 * (o == p).sum(); hit.append(float(k == 1)); rr.append(1 / k)
    return float(np.mean(hit)), float(np.mean(rr))


PARAPHRASES = {}          # original text -> paraphrased text; filled by set_paraphrases() (paraphrase_lib.load_map) for the kind "paraphrase"


def set_paraphrases(mapping):
    PARAPHRASES.clear(); PARAPHRASES.update(mapping)


def _apply(kind, param, text, lex, rng):
    if kind == "paraphrase": return PARAPHRASES.get(text, text)      # deterministic; texts without an accepted paraphrase stay unchanged (coverage is reported)
    if kind == "esco_all": return P.esco_synonyms(text, lex, param, rng)[0]
    if kind == "esco_multiword": return P.esco_synonyms(text, lex, param, rng, min_words=2)[0]
    if kind == "typo": return P.typos(text, param, rng)[0]
    if kind == "reorder": return P.reorder(text, rng)
    raise ValueError(kind)


def run_suite(make_model, model_name, out_dir, seeds=range(5), repeats=10, settings=SETTINGS):
    os.makedirs(out_dir, exist_ok=True); path = f"{out_dir}/robust_raw.csv"
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []; done = {(r["seed"], r["kind"], r["param"], r["side"], r["rep"]) for r in rows}
    lex = P.build_esco_lexicon(); t0 = time.time()
    for seed in seeds:
        d = pd.read_csv(f"{SEED_DIR}/pairs_seed{seed}.csv"); tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
        if all((seed, k, p, sd, r) in done for k, p in settings for sd in SIDES for r in range(repeats)): continue
        m = make_model(tr, seed); y = te.label.values; s0 = m.score(te); pred0 = (s0 >= m.thr).astype(int); acc0 = float((pred0 == y).mean()); h0, mrr0 = _rank(te, s0); sd0 = float(np.std(s0))
        for kind, param in settings:
            for rep in range(repeats):
                rng_r = np.random.default_rng(zlib.crc32(repr((seed, kind, param, rep, "r")).encode())); rng_j = np.random.default_rng(zlib.crc32(repr((seed, kind, param, rep, "j")).encode()))
                rmap = {t: _apply(kind, param, t, lex, rng_r) for t in te.resume_text.unique()}; jmap = {t: _apply(kind, param, t, lex, rng_j) for t in te.job_text.unique()}
                for side in SIDES:
                    if (seed, kind, param, side, rep) in done: continue
                    te2 = te.copy()
                    if side in ("resume_only", "both"): te2["resume_text"] = te.resume_text.map(rmap)
                    if side in ("job_only", "both"): te2["job_text"] = te.job_text.map(jmap)
                    s1 = m.score(te2); pred1 = (s1 >= m.thr).astype(int); h1, mrr1 = _rank(te, s1)
                    ch_r = np.mean([rmap[t] != t for t in rmap]) if side != "job_only" else 0.0; ch_j = np.mean([jmap[t] != t for t in jmap]) if side != "resume_only" else 0.0
                    rows.append(dict(model=model_name, seed=seed, kind=kind, param=param, side=side, rep=rep, acc_clean=acc0, dacc=float((pred1 == y).mean()) - acc0, flip=float((pred1 != pred0).mean()),
                                     mean_abs_dscore=float(np.mean(np.abs(s1 - s0))), dscore_z=float(np.mean(np.abs(s1 - s0)) / sd0), score_sd_clean=sd0, dhit1=h1 - h0, dmrr=mrr1 - mrr0, share_resumes_changed=ch_r, share_jobs_changed=ch_j))
        pd.DataFrame(rows).to_csv(path, index=False); print(f"{model_name} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows); R.to_csv(path, index=False)
    S = R.groupby(["model", "kind", "param", "side"]).agg(acc_clean=("acc_clean", "mean"), dacc=("dacc", "mean"), dacc_sd=("dacc", "std"), flip=("flip", "mean"), flip_sd=("flip", "std"), dscore_z=("dscore_z", "mean"),
          dhit1=("dhit1", "mean"), dmrr=("dmrr", "mean"), dmrr_sd=("dmrr", "std"), resumes_changed=("share_resumes_changed", "mean"), jobs_changed=("share_jobs_changed", "mean"), n_cells=("rep", "size")).round(4).reset_index()
    S.to_csv(f"{out_dir}/robust_summary.csv", index=False); return S
