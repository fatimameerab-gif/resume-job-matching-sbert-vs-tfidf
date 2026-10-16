import os, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from data_loading import load_resumes

OUT = "results/skill_gap"
# category -> (ESCO preferredLabel, quality). 'approx' = ESCO has no occupation with the same specificity (e.g. no Python-vs-Java split).
CATEGORY_TO_ESCO = {
    "Data Science": ("data scientist", "direct"), "HR": ("human resources manager", "direct"), "Civil Engineer": ("civil engineer", "direct"),
    "Mechanical Engineer": ("mechanical engineer", "direct"), "Electrical Engineering": ("electrical engineer", "direct"), "Sales": ("sales manager", "direct"),
    "Business Analyst": ("business analyst", "direct"), "Database": ("database administrator", "direct"), "Operations Manager": ("operations manager", "direct"),
    "Advocate": ("lawyer", "direct"), "Testing": ("software tester", "direct"), "Web Designing": ("web developer", "approx"),
    "Java Developer": ("software developer", "approx"), "Python Developer": ("software developer", "approx"), "DotNet Developer": ("software developer", "approx"),
    "Hadoop": ("data engineer", "approx"), "ETL Developer": ("data warehouse designer", "approx"), "DevOps Engineer": ("ICT system administrator", "approx"),
    "Network Security Engineer": ("ICT security administrator", "approx"), "PMO": ("project manager", "approx"), "Health and fitness": ("fitness instructor", "approx"),
    "SAP Developer": ("ICT application developer", "approx"), "Automation Testing": ("software tester", "approx"), "Arts": ("graphic designer", "approx"),
}
WINDOW_SIZES = (4, 8); STRIDE = 2
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)

def load_esco():
    sk = pd.read_csv("data/esco_skills.csv"); oc = pd.read_csv("data/esco_occupations.csv"); mp = pd.read_csv("data/occupation_skills_mapping.csv")
    labels = {}
    for r in sk.itertuples():
        alts = [a.strip().lower() for a in str(r.altLabels).split("\n")] if isinstance(r.altLabels, str) else []
        labels[r.skill_uri] = [r.preferredLabel.lower()] + [a for a in alts if a]
    return sk, oc, mp, labels

def essential_skills(occ_label, oc, mp, labels):
    row = oc[oc.preferredLabel.str.lower() == occ_label.lower()]
    if row.empty: return []
    uris = mp[(mp.occupation_uri == row.iloc[0].occupation_uri) & (mp.relation_type == "essential")].skill_uri.tolist()
    return [(u, labels[u]) for u in uris if u in labels]

def windows(text):
    w = str(text).split(); out = []
    for k in WINDOW_SIZES:
        out += [" ".join(w[i:i + k]) for i in range(0, max(len(w) - k + 1, 1), STRIDE)]
    return out or [str(text)]

class TfidfCharScorer:
    """score(resume_text, skills) -> array[n_skills] = max cosine between any label of the skill and any window of the resume."""
    name = "tfidf_char"
    def __init__(self, all_labels):
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(all_labels); self._cache = {}
    def _win(self, text):
        if text not in self._cache: self._cache = {text: self.v.transform(windows(text))}
        return self._cache[text]
    def score(self, text, skills):
        W = self._win(text); flat = [l for _, ls in skills for l in ls]; owner = np.repeat(np.arange(len(skills)), [len(ls) for _, ls in skills])
        S = (self.v.transform(flat) @ W.T).max(axis=1).toarray().ravel(); out = np.zeros(len(skills)); np.maximum.at(out, owner, S); return out

class DenseScorer:
    """Same interface with a bi-encoder (colab_common.Encoder): cosine of label embedding vs window embeddings. Used by colab_skill_gap_dense.py."""
    def __init__(self, enc): self.enc = enc; self.name = f"dense_{enc.key}"; self._c = None
    def score(self, text, skills):
        if self._c is None or self._c[0] != text: self._c = (text, self.enc.encode(windows(text), "resume"))
        W = self._c[1]; flat = [l for _, ls in skills for l in ls]; owner = np.repeat(np.arange(len(skills)), [len(ls) for _, ls in skills])
        S = (self.enc.encode(flat, "job") @ W.T).max(axis=1); out = np.full(len(skills), -1.0); np.maximum.at(out, owner, S); return out

def exact_present(text, skills):
    t = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
    return np.array([float(any(f" {l} " in t for l in ls)) for _, ls in skills])

def coverage(scores, tau): return float((scores >= tau).mean()) if len(scores) else np.nan

def validity_check(resumes, targets, scorer, n_other=5, seed=0):
    """own-occupation coverage vs other-occupation coverage. Returns long DataFrame with one row per (resume, occupation)."""
    rng = np.random.default_rng(seed); rows = []; cats = sorted(targets)
    for r in resumes.itertuples():
        if r.category not in targets or not targets[r.category][1]: continue
        others = [c for c in cats if targets[c][0] != targets[r.category][0] and c != r.category and targets[c][1]]
        picks = list(rng.choice(others, min(n_other, len(others)), replace=False))
        for kind, c in [("own", r.category)] + [("other", p) for p in picks]:
            occ, sk, _ = targets[c]
            s = scorer.score(r.text_clean, sk); ex = exact_present(r.text_clean, sk)
            rows.append(dict(resume_id=r.resume_id, resume_category=r.category, kind=kind, target_category=c, occupation=occ, n_skills=len(sk),
                             exact_cov=float(ex.mean()), mean_best_sim=float(s.mean()), **{f"cov_tau{t}": coverage(s, t) for t in TAUS}))
    return pd.DataFrame(rows)

def auc_table(V, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); out = []; ids = V.resume_id.unique()
    for m in ["exact_cov", "mean_best_sim"] + [f"cov_tau{t}" for t in TAUS]:
        def auc(d): return roc_auc_score((d.kind == "own").astype(int), d[m]) if d.kind.nunique() == 2 else np.nan
        a = auc(V); g = {r: d for r, d in V.groupby("resume_id")}
        bs = [auc(pd.concat([g[i] for i in rng.choice(ids, len(ids))])) for _ in range(n_boot // 5)]
        own, oth = V[V.kind == "own"][m].mean(), V[V.kind == "other"][m].mean()
        out.append(dict(metric=m, auc_own_vs_other=a, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5), mean_own=own, mean_other=oth))
    return pd.DataFrame(out)

def main(out=OUT, scorer=None, max_resumes=None):
    os.makedirs(out, exist_ok=True)
    sk, oc, mp, labels = load_esco()
    targets = {}
    for cat, (occ, q) in CATEGORY_TO_ESCO.items():
        s = essential_skills(occ, oc, mp, labels)
        if s: targets[cat] = (occ, s, q)
    scorer = scorer or TfidfCharScorer([l for ls in labels.values() for l in ls])
    resumes = load_resumes().drop_duplicates(subset="text_clean").reset_index(drop=True); resumes["resume_id"] = resumes.index
    if max_resumes: resumes = resumes.sample(max_resumes, random_state=0)
    V = validity_check(resumes, targets, scorer)
    V.to_csv(f"{out}/validity_per_resume_{scorer.name}.csv", index=False)
    T = auc_table(V); T.insert(0, "scorer", scorer.name); T.round(4).to_csv(f"{out}/validity_auc_{scorer.name}.csv", index=False)
    # quality-stratified view (direct vs approx targets)
    q = {c: v[2] for c, v in targets.items()}; V["quality"] = V.resume_category.map(q)
    rows = [dict(quality=qq, **{m: roc_auc_score((d.kind == "own").astype(int), d[m]) for m in ("exact_cov", "mean_best_sim", "cov_tau0.5")}) for qq, d in V.groupby("quality")]
    pd.DataFrame(rows).round(4).to_csv(f"{out}/validity_auc_by_target_quality_{scorer.name}.csv", index=False)
    # old behaviour for comparison: how many essential skills does exact matching find at all?
    cmp = V[V.kind == "own"][["exact_cov", "cov_tau0.5", "cov_tau0.7"]].mean()
    pd.DataFrame(dict(mean_own_coverage=cmp)).round(4).to_csv(f"{out}/own_coverage_exact_vs_linking_{scorer.name}.csv")
    # illustrative reports (qualitative only)
    ex = []
    for cat in ("Data Science", "HR", "Civil Engineer", "Java Developer", "Sales", "Testing"):
        r = resumes[resumes.category == cat]
        if r.empty or cat not in targets: continue
        occ, sks, _ = targets[cat]; s = scorer.score(r.iloc[0].text_clean, sks); order = np.argsort(s)
        ex.append(dict(resume_category=cat, target=occ, n_essential=len(sks), n_present_tau0_5=int((s >= .5).sum()),
                       top_present=" | ".join(sks[i][1][0] for i in order[::-1][:5]), top_missing=" | ".join(sks[i][1][0] for i in order[:5])))
    pd.DataFrame(ex).to_csv(f"{out}/illustrative_reports_{scorer.name}.csv", index=False)
    open(f"{out}/NOTE_skill_gap.txt", "w").write(
        "Automatic validity only: AUC of 'own occupation covered more than other occupations'. This is NOT precision/recall of the gap list. "
        "Tau values are a sensitivity grid, not tuned. Gold-set annotation (make_gap_gold_kit.py / compute_gap_metrics.py) is required for P/R/F1. "
        "ESCO essential skills are generic (e.g. 'comply with legal regulations'), so low coverage of own occupation is expected even for a perfect extractor.\n")
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False)); print(pd.read_csv(f"{out}/own_coverage_exact_vs_linking_{scorer.name}.csv").to_string(index=False))

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore"); main()
