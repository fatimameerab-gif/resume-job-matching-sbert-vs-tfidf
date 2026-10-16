import os
import re
import pandas as pd

SEEK_PATH = "data/seek_australia.csv"                 # raw Zenodo file (optional, not shipped)
SELECTED_PATH = "data/seek_selected_postings.csv"       # the 563 selected postings: title, description, category only (shipped)
CAP_PER_CATEGORY = 30
MIN_WORDS = 50
SEED = 42

# (category, include-regex, exclude-regex or None); matched on the lower-cased job title
RULES = [
    ("Advocate", r"\b(lawyer|solicitor|barrister|advocate|paralegal|legal counsel|legal (associate|adviser|advisor|officer))\b", r"secretar|receptionist|assistant"),
    ("Business Analyst", r"\bbusiness analyst", None),
    ("Civil Engineer", r"\bcivil engineer", None),
    ("Data Science", r"\b(data scientist|data analyst|machine learning)\b", r"\b(sales|developer)\b"),
    ("Database", r"\b(dba|database (administrator|analyst|developer|engineer|designer))\b", None),
    ("DevOps Engineer", r"\bdevops\b", None),
    ("DotNet Developer", r"(\.net\b|\bdotnet\b)", None),
    ("ETL Developer", r"\betl\b", r"\bchef\b"),
    ("Electrical Engineering", r"\belectrical engineer", None),
    ("HR", r"\b(human resources|hr)\b", r"driver|chr"),
    ("Hadoop", r"\b(hadoop|big data engineer|data engineer)", None),
    ("Health and fitness", r"\b(fitness|personal trainer|gym instructor|health club)\b", None),
    ("Java Developer", r"\bjava (software )?(developer|engineer)", None),
    ("Mechanical Engineer", r"\bmechanical engineer", None),
    ("Network Security Engineer", r"\b(network engineer|network security|security engineer)", None),
    ("Operations Manager", r"\boperations manager", None),
    ("PMO", r"\bpmo\b", None),
    ("Python Developer", r"\bpython (software )?(developer|engineer)", None),
    ("SAP Developer", r"\bsap (abap|developer|consultant|analyst|sd|mm|fico|hana)", None),
    ("Sales", r"\bsales (manager|representative|consultant|executive|officer|associate)", r"\b(data|analytics)\b"),
    ("Testing", r"\b(test (engineer|analyst|manager|lead)|software tester|qa (engineer|analyst|tester))", None),
    ("Web Designing", r"\bweb (designer|developer)", None),
]


def _category(title):
    t = str(title).lower()
    hits = [c for c, inc, exc in RULES if re.search(inc, t) and not (exc and re.search(exc, t))]
    return hits[0] if len(hits) == 1 else None


def load_seek_jobs(path=SEEK_PATH, cap=CAP_PER_CATEGORY, with_category=False):
    """Uses the raw Zenodo file if present; otherwise the shipped selection (identical rows, only title/description/category)."""
    if not os.path.exists(path):
        if os.path.exists(SELECTED_PATH):
            df = pd.read_csv(SELECTED_PATH)
            return df if with_category else df[["title", "text"]]
        raise FileNotFoundError(f"Neither {path} nor {SELECTED_PATH} found. Download seek_australia.csv from "
                                "https://zenodo.org/records/1220874 (CC BY 4.0) and put it in data/.")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["job_title", "job_description"])  # nothing else is read
    df = df.rename(columns={"job_title": "title", "job_description": "text"}).dropna()
    df["title"] = df["title"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset=["title", "text"]).drop_duplicates(subset=["text"])
    df["seek_category"] = df["title"].apply(_category)
    df = df.dropna(subset=["seek_category"])
    parts = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby("seek_category")]
    df = pd.concat(parts, ignore_index=True)
    return df if with_category else df[["title", "text"]]


if __name__ == "__main__":
    d = load_seek_jobs(with_category=True)
    if os.path.exists(SEEK_PATH):
        d.to_csv(SELECTED_PATH, index=False)
        print("wrote", SELECTED_PATH)
    print(len(d), "SEEK postings kept")
    print(d.seek_category.value_counts().to_string())
