import re
import pandas as pd

RESUME_PATH = "data/UpdatedResumeDataSet.csv"
JOBS_DS_PATH = "data/data_science_job_postings_672.csv"
JOBS_MIXED_PATH = "data/_archive_unlicensed/mixed_domain_job_postings_157.csv"
ESCO_OCCUPATIONS_PATH = "data/esco_occupations.csv"


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip URLs/special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)          # keep +,# for C++, C# etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_resumes() -> pd.DataFrame:
    df = pd.read_csv(RESUME_PATH)
    df = df.rename(columns={"Category": "category", "Resume": "text"})
    df["text_clean"] = df["text"].apply(clean_text)
    df["source"] = "resume"
    return df[["category", "text", "text_clean", "source"]]


INCLUDE_UNLICENSED_157 = False   # that GitHub file has NO licence -> never used unless someone sets this on purpose


def load_job_postings(include_unlicensed=None) -> pd.DataFrame:
    """Job postings = data_science_job_postings_672.csv (Kaggle, CC0) + SEEK Australia (Zenodo, CC BY 4.0,
    via seek_adapter.py). The licence-less 157-row file is only read if INCLUDE_UNLICENSED_157 is switched on."""
    if include_unlicensed is None:
        include_unlicensed = INCLUDE_UNLICENSED_157
    df1 = pd.read_csv(JOBS_DS_PATH)
    df1 = df1.rename(columns={"Job Title": "title", "Job Description": "text"})
    df1 = df1[["title", "text"]]
    df1["job_source"] = "kaggle_672"
    df1["seek_category"] = None

    from seek_adapter import load_seek_jobs
    df_seek = load_seek_jobs(with_category=True).rename(columns={})
    df_seek["job_source"] = "seek_australia"
    parts = [df1, df_seek[["title", "text", "job_source", "seek_category"]]]

    if include_unlicensed:
        df2 = pd.read_csv(JOBS_MIXED_PATH, index_col=0)
        df2 = df2.rename(columns={"position": "title", "Job Description": "text"})[["title", "text"]]
        df2["job_source"] = "github_157_unlicensed"
        df2["seek_category"] = None
        parts.append(df2)

    jobs = pd.concat(parts, ignore_index=True)
    jobs["text_clean"] = jobs["text"].apply(clean_text)
    jobs["source"] = "job_posting"
    return jobs


# Keyword-based mapping from job titles to the resume dataset's category
# vocabulary. Used to derive weak-supervision match/no-match labels
# (see src/pairing.py). Extend this dict as needed after manual inspection.
TITLE_TO_CATEGORY = {
    "data scien": "Data Science",
    "data analy": "Data Science",
    "machine learning": "Data Science",
    "data engineer": "Hadoop",
    "hadoop": "Hadoop",
    "java develop": "Java Developer",
    "python develop": "Python Developer",
    ".net develop": "DotNet Developer",
    "dotnet": "DotNet Developer",
    "devops": "DevOps Engineer",
    "web develop": "Web Designing",
    "web design": "Web Designing",
    "sap": "SAP Developer",
    "sales": "Sales",
    "human resource": "HR",
    "hr ": "HR",
    "recruit": "HR",
    "test": "Testing",
    "qa engineer": "Testing",
    "automation test": "Automation Testing",
    "network security": "Network Security Engineer",
    "network engineer": "Network Security Engineer",
    "database": "Database",
    "dba": "Database",
    "business analy": "Business Analyst",
    "operations manager": "Operations Manager",
    "project management": "PMO",
    "pmo": "PMO",
    "electrical engineer": "Electrical Engineering",
    "mechanical engineer": "Mechanical Engineer",
    "civil engineer": "Civil Engineer",
    "blockchain": "Blockchain",
    "health": "Health and fitness",
    "fitness": "Health and fitness",
    "etl": "ETL Developer",
    "advocate": "Advocate",
    "legal": "Advocate",
}


def map_title_to_category(title: str):
    if not isinstance(title, str):
        return None
    t = title.lower()
    for keyword, category in TITLE_TO_CATEGORY.items():
        if keyword in t:
            return category
    return None


def load_esco_occupations() -> pd.DataFrame:
    """
    Loads ESCO occupation descriptions as a *supplementary* job-description
    source, used to fill in resume categories for which the scraped job
    postings (data_science_job_postings_672.csv / mixed_domain..._157.csv)
    have little or no coverage (see ESCO_CATEGORY_TERMS below).
    """
    esco = pd.read_csv(ESCO_OCCUPATIONS_PATH)
    esco = esco.rename(columns={"preferredLabel": "title", "description": "text"})
    esco = esco[["title", "text"]].dropna(subset=["text"])
    esco["text_clean"] = esco["text"].apply(clean_text)
    esco["source"] = "esco_occupation"
    return esco


# Manually verified keyword mapping from ESCO occupation labels to the
# resume dataset's 25 categories. Used only for categories where the
# scraped job-postings data has little/no coverage (see README for the
# rationale and the coverage numbers before/after adding this source).
ESCO_CATEGORY_TERMS = {
    "Mechanical Engineer": ["mechanical engineer"],
    "Civil Engineer": ["civil engineer"],
    "Electrical Engineering": ["electrical engineer"],
    "Sales": ["sales representative", "sales manager", "sales engineer"],
    "Business Analyst": ["business analyst", "business intelligence"],
    "Database": ["database administrator", "database developer", "database designer"],
    "Blockchain": ["blockchain developer", "blockchain architect"],
    "Health and fitness": ["fitness instructor", "personal trainer", "health and safety"],
    "Automation Testing": ["test automation", "automation engineer"],
    "Operations Manager": ["operations manager"],
    "Web Designing": ["web designer", "web developer"],
    "Advocate": ["lawyer", "legal counsel", "solicitor"],
    "HR": ["human resources", "recruitment consultant", "hr manager"],
    "Network Security Engineer": ["network engineer", "security engineer", "cybersecurity"],
    "DevOps Engineer": ["devops engineer"],
    "Arts": ["artist", "art director", "graphic designer"],
    "Java Developer": ["software developer", "applications developer"],
    "Python Developer": ["software developer", "applications developer"],
    "DotNet Developer": ["software developer", "applications developer"],
    "ETL Developer": ["data engineer", "data warehouse designer"],
    "PMO": ["project manager", "programme manager"],
    "Testing": ["test engineer", "test analyst", "quality assurance", "software tester"],
}


def map_esco_title_to_category(title: str):
    """Returns a list of ALL matching resume categories (an ESCO occupation
    such as 'software developer' can validly represent multiple resume
    categories, e.g. Java/Python/DotNet Developer, since ESCO does not
    distinguish occupations by programming language)."""
    if not isinstance(title, str):
        return []
    t = title.lower()
    matches = []
    for category, terms in ESCO_CATEGORY_TERMS.items():
        for term in terms:
            if term in t:
                matches.append(category)
                break
    return matches


if __name__ == "__main__":
    resumes = load_resumes()
    jobs = load_job_postings()
    esco = load_esco_occupations()

    print(f"Resumes loaded: {len(resumes)} rows, {resumes['category'].nunique()} categories")
    print(f"Job postings loaded: {len(jobs)} rows")
    print(f"ESCO occupations loaded: {len(esco)} rows")

    jobs["mapped_category"] = jobs["title"].apply(map_title_to_category)
    esco["mapped_categories"] = esco["title"].apply(map_esco_title_to_category)

    job_cats = set(jobs["mapped_category"].dropna().unique())
    esco_cats = set(cat for cats in esco["mapped_categories"] for cat in cats)
    combined_cats = job_cats | esco_cats

    print(f"\nCategories covered by job postings alone: {len(job_cats)} / 25")
    print(f"Categories covered by ESCO occupations alone: {len(esco_cats)} / 25")
    print(f"Categories covered when COMBINED: {len(combined_cats)} / 25")
    print(f"\nStill missing: {set(resumes['category'].unique()) - combined_cats}")
