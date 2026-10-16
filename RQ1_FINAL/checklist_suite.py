import os, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
SEED_DIR = "data/eval_v7/frozen_seeds"; FILL = " references available upon request. available for interview at your convenience. thank you for your consideration."


def run_checklist(make_model, model_name, out_dir, seeds=range(5), z_tol=0.1, nboot=2000):
    os.makedirs(out_dir, exist_ok=True); rows = []
    for seed in seeds:
        df = pd.read_csv(f"{SEED_DIR}/pairs_seed{seed}.csv"); tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True); rng = np.random.default_rng(seed)
        v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text])); vocab = np.array(v.get_feature_names_out())
        topwords = lambda job, k=10: " ".join(w for w in vocab[np.argsort(-v.transform([job]).toarray()[0])[:k]] if " " not in w)
        by_cat = te.groupby("resume_category").resume_text.apply(list).to_dict()
        def foreign(cat):
            others = [c for c in by_cat if c != cat]; c = others[rng.integers(len(others))]; t = by_cat[c][rng.integers(len(by_cat[c]))]; return " ".join(t.split()[:30])
        m = make_model(tr, seed); s0 = m.score(te); sd = float(np.std(s0))
        variants = {"INV_filler": (te.resume_text + FILL, "inv"), "INV_punct": (te.resume_text.map(lambda s: re.sub(r"\s+", "  ", re.sub(r"[^\w\s]", "", s.upper()))), "inv"),
                    "DIR_up_copy_job": (te.resume_text + " " + te.job_text.map(topwords), "up"), "DIR_down_foreign": (te.resume_text + " " + te.resume_category.map(foreign), "down")}
        for name, (newres, mode) in variants.items():
            te2 = te.copy(); te2["resume_text"] = newres; d = m.score(te2) - s0
            use = np.ones(len(te), bool) if mode != "down" else (te.label.values == 1)
            ok = {"inv": np.abs(d) < z_tol * sd, "up": d > 1e-9, "down": d < -1e-9}[mode]
            ok_old = np.abs(d) < 0.02 if mode == "inv" else ok
            for i in np.where(use)[0]: rows.append(dict(model=model_name, seed=seed, test=name, resume_id=te.resume_id.iloc[i], passed=int(ok[i]), passed_fixed002=int(ok_old[i]), dscore=float(d[i]), dscore_z=float(d[i] / sd), score_sd=sd))
    R = pd.DataFrame(rows); R.to_csv(f"{out_dir}/checklist_pairs.csv", index=False); rng = np.random.default_rng(0); out = []
    for t, g in R.groupby("test"):
        per = g.groupby("resume_id")[["passed", "passed_fixed002"]].agg(["sum", "count"]); ids = per.index.values
        P, N = per[("passed", "sum")].values, per[("passed", "count")].values; P2 = per[("passed_fixed002", "sum")].values
        bs = []
        for _ in range(nboot):
            k = rng.integers(0, len(ids), len(ids)); bs.append(P[k].sum() / N[k].sum())
        out.append(dict(model=model_name, test=t, n_pairs=int(N.sum()), n_resumes=len(ids), passes=int(P.sum()), pass_rate=round(P.sum() / N.sum(), 3), ci_lo=round(float(np.percentile(bs, 2.5)), 3), ci_hi=round(float(np.percentile(bs, 97.5)), 3),
                        pass_rate_fixed_tol_0p02=round(P2.sum() / N.sum(), 3), mean_dscore_z=round(float(g.dscore_z.mean()), 3), tolerance=f"{z_tol} x SD" if t.startswith("INV") else "sign"))
    S = pd.DataFrame(out); S.to_csv(f"{out_dir}/checklist_summary.csv", index=False); return S
