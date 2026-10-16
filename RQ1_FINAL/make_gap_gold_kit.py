import os, sys, numpy as np, pandas as pd
from skill_gap_v2 import CATEGORY_TO_ESCO, load_esco, essential_skills, TfidfCharScorer, exact_present
from data_loading import load_resumes
OUT = "annotation_kit_skillgap"
GUIDE = """# Skill-gap annotation guidelines
Task: for each row you see a RESUME (look it up by resume_id in gap_resumes.csv), a TARGET OCCUPATION and one ESSENTIAL SKILL of that occupation (ESCO). Decide whether the resume shows that the person HAS the skill.
Fill the column `present` with exactly one of:
- yes    : the resume names the skill or a clear synonym/abbreviation, or describes work that obviously requires it (e.g. 'built regression models in scikit-learn' for 'apply statistical analysis techniques').
- partly : the resume shows a related or lower-level version of the skill, or only weak indirect evidence.
- no     : no evidence in the resume. (Absence of evidence is 'no', even if the person might have the skill.)
Rules: judge ONLY from the resume text; do not look at system scores; do not talk to the other annotator until both sheets are finished; use `evidence` for a short quote (optional).
ESCO essential skills are often generic (e.g. 'comply with legal regulations'); label them the same way - 'no' if the resume gives no evidence.
Resume texts are lower-cased and cleaned; ignore that. Time: about 20-30 seconds per row.
"""

def main(n_resumes=30, k=20, out=OUT, seed=0):
    os.makedirs(out, exist_ok=True); rng = np.random.default_rng(seed)
    sk, oc, mp, labels = load_esco(); scorer = TfidfCharScorer([l for ls in labels.values() for l in ls])
    R = load_resumes().drop_duplicates(subset="text_clean").reset_index(drop=True); R["resume_id"] = R.index
    cats = [c for c in CATEGORY_TO_ESCO if essential_skills(CATEGORY_TO_ESCO[c][0], oc, mp, labels)]
    per = max(1, n_resumes // len(cats)); pick = []
    for c in cats:
        g = R[R.category == c]; pick += list(g.sample(min(per, len(g)), random_state=seed).index)
    rest = [i for i in R.index if i not in pick and R.category[i] in cats]; pick += list(rng.choice(rest, max(0, n_resumes - len(pick)), replace=False))
    pick = pick[:n_resumes]; rows, key = [], []
    for ri in pick:
        r = R.loc[ri]; occ, q = CATEGORY_TO_ESCO[r.category]; sks = essential_skills(occ, oc, mp, labels)
        idx = sorted(rng.choice(len(sks), min(k, len(sks)), replace=False)); sub = [sks[i] for i in idx]
        sc = scorer.score(r.text_clean, sub); ex = exact_present(r.text_clean, sub)
        desc = sk.set_index("skill_uri").description
        for (u, ls), s, e in zip(sub, sc, ex):
            iid = f"G{len(rows):04d}"
            rows.append(dict(item_id=iid, resume_id=int(r.resume_id), target_occupation=occ, skill_label=ls[0], skill_alt_labels=" | ".join(ls[1:6]), skill_description=str(desc.get(u, ""))[:300], present="", evidence=""))
            key.append(dict(item_id=iid, resume_id=int(r.resume_id), resume_category=r.category, target_quality=q, skill_uri=u, exact_present=int(e), tfidf_char_score=round(float(s), 4)))
    pd.DataFrame(rows).sample(frac=1, random_state=seed).sort_values(["resume_id", "item_id"]).to_csv(f"{out}/annotation-sheet-skillgap.csv", index=False)
    R.loc[pick, ["resume_id", "category", "text_clean"]].rename(columns={"text_clean": "resume_text"}).drop(columns="category").to_csv(f"{out}/gap_resumes.csv", index=False)
    pd.DataFrame(key).to_csv(f"{out}/KEY-system-scores-do-not-share.csv", index=False); open(f"{out}/ANNOTATION-GUIDELINES-skillgap.md", "w").write(GUIDE)
    print(f"{len(pick)} resumes, {len(rows)} rows to annotate (about {len(rows) * 25 / 3600:.1f} h per annotator)")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, int(sys.argv[2]) if len(sys.argv) > 2 else 20)
