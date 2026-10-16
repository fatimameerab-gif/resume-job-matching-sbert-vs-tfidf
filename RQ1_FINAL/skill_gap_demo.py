import re
import pandas as pd

from data_loading import load_resumes, clean_text


def load_esco_data():
    skills = pd.read_csv("data/esco_skills.csv")
    occupations = pd.read_csv("data/esco_occupations.csv")
    mapping = pd.read_csv("data/occupation_skills_mapping.csv")
    return skills, occupations, mapping


def build_skill_matcher(skills_df, min_len=4):
    """Builds a single compiled regex over ESCO preferredLabels (skills
    with reasonably short, matchable labels) for fast skill detection in
    free text."""
    terms = skills_df["preferredLabel"].dropna().unique().tolist()
    terms = [t for t in terms if len(t) >= min_len and len(t.split()) <= 4]
    terms_sorted = sorted(terms, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t.lower()) for t in terms_sorted) + r")\b")
    return pattern


def extract_skills(text: str, pattern) -> set:
    text = text.lower()
    return set(pattern.findall(text))


def skill_gap_report(resume_text, occupation_label, skills_df, occupations_df, mapping_df, pattern, top_n_missing=10):
    occ_row = occupations_df[occupations_df["preferredLabel"].str.lower() == occupation_label.lower()]
    if len(occ_row) == 0:
        return None
    occ_uri = occ_row.iloc[0]["occupation_uri"]

    required_skill_uris = mapping_df[
        (mapping_df["occupation_uri"] == occ_uri) & (mapping_df["relation_type"] == "essential")
    ]["skill_uri"].tolist()
    required_skills = skills_df[skills_df["skill_uri"].isin(required_skill_uris)]["preferredLabel"].str.lower().tolist()

    resume_skills = extract_skills(resume_text, pattern)
    required_set = set(required_skills)

    present = resume_skills & required_set
    missing = required_set - resume_skills

    return {
        "occupation": occupation_label,
        "n_required_skills_in_esco": len(required_set),
        "n_skills_present_in_resume": len(present),
        "n_skills_missing": len(missing),
        "present_skills": sorted(present),
        "missing_skills": sorted(list(missing))[:top_n_missing],
    }


def main():
    print("=" * 70)
    print("CAVEAT: this is an UNEVALUATED exact-string-matching demo, not a")
    print("validated skill-extraction system. See this file's docstring.")
    print("Do not report these numbers as measured precision/recall.")
    print("=" * 70)
    skills_df, occupations_df, mapping_df = load_esco_data()
    resumes = load_resumes().drop_duplicates(subset="text_clean").reset_index(drop=True)

    print("Building ESCO skill-matching regex (this takes a few seconds)...")
    pattern = build_skill_matcher(skills_df)

    # Demo: run skill-gap analysis for a few sample resumes against a
    # plausible target occupation from the same broad category
    demo_targets = {
        "Data Science": "data scientist",
        "HR": "human resources manager",
        "Civil Engineer": "civil engineer",
    }

    reports = []
    for category, target_occ in demo_targets.items():
        sample = resumes[resumes["category"] == category]
        if len(sample) == 0:
            continue
        sample_resume = sample.iloc[0]

        report = skill_gap_report(
            sample_resume["text_clean"], target_occ,
            skills_df, occupations_df, mapping_df, pattern
        )
        if report is None:
            print(f"\n[skip] No ESCO occupation found matching '{target_occ}'")
            continue

        print(f"\n=== Skill-Gap Report: {category} resume -> target role '{target_occ}' ===")
        print(f"Required skills (ESCO, essential): {report['n_required_skills_in_esco']}")
        print(f"Present in resume: {report['n_skills_present_in_resume']}  -> {report['present_skills'][:8]}")
        print(f"Missing (gap), showing top {len(report['missing_skills'])}: {report['missing_skills']}")

        reports.append({
            "resume_category": category,
            "target_occupation": target_occ,
            **{k: v for k, v in report.items() if k != "occupation"},
        })

    pd.DataFrame(reports).to_csv("results/skill_gap_demo_reports.csv", index=False)
    print("\nSaved results/skill_gap_demo_reports.csv")
    print("Reminder: illustrative only -- exact-string matching, no gold-standard "
          "evaluation exists for these numbers (see docstring).")


if __name__ == "__main__":
    main()
