import re

# (abbreviation, expansion)
CORE_PAIRS = [
    ("js", "javascript"), ("postgres", "postgresql"), ("k8s", "kubernetes"),
    ("nlp", "natural language processing"), ("aws", "amazon web services"),
    ("gcp", "google cloud platform"), ("sklearn", "scikit-learn"),
    ("ai", "artificial intelligence"), ("qa", "quality assurance"),
    ("ui", "user interface"), ("ux", "user experience"), ("db", "database"),
    ("oop", "object oriented programming"), ("sdlc", "software development life cycle"),
    ("erp", "enterprise resource planning"), ("crm", "customer relationship management"),
    ("api", "application programming interface"), ("etl", "extract transform load"),
    ("rdbms", "relational database management system"), ("dbms", "database management system"),
    ("iot", "internet of things"), ("svm", "support vector machine"),
    ("seo", "search engine optimization"), ("vpn", "virtual private network"),
    ("gui", "graphical user interface"), ("hvac", "heating ventilation and air conditioning"),
    ("plc", "programmable logic controller"),
    ("scada", "supervisory control and data acquisition"),
    ("kpi", "key performance indicator"), ("roi", "return on investment"),
    ("b2b", "business to business"), ("ms office", "microsoft office"),
    ("ms excel", "microsoft excel"), ("ms word", "microsoft word"),
]
RISKY_PAIRS = [
    ("ml", "machine learning"), ("dl", "deep learning"), ("ci", "continuous integration"),
    ("tf", "tensorflow"), ("cad", "computer aided design"),
    ("mis", "management information system"), ("cnn", "convolutional neural network"),
    ("bi", "business intelligence"),
]
ALL_PAIRS = CORE_PAIRS + RISKY_PAIRS
DIRECTIONS = ("abbr_to_full", "full_to_abbr")

# legacy name kept for tests / older scripts: expansion -> abbreviation, CORE only
SAFE_SYNONYMS = {full: abbr for abbr, full in CORE_PAIRS}


def build_mapping(pairs=CORE_PAIRS, direction="abbr_to_full"):
    if direction == "abbr_to_full":
        return {a: f for a, f in pairs}
    if direction == "full_to_abbr":
        return {f: a for a, f in pairs}
    raise ValueError(direction)


def compile_mapping(mapping):
    keys = sorted(mapping, key=len, reverse=True)          # longest first: 'ms excel' before 'ms'
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)


def perturb_counted(text, mapping, regex=None, rate=1.0, rng=None):
    """Single regex pass. Each match is replaced with probability `rate`.
    Returns (new_text, n_matches, n_replaced, hit_keys)."""
    regex = regex or compile_mapping(mapping)
    stats = {"m": 0, "r": 0, "hits": []}

    def _sub(m):
        stats["m"] += 1
        if rate >= 1.0 or (rng is not None and rng.random() < rate):
            stats["r"] += 1
            stats["hits"].append(m.group(0).lower())
            return mapping[m.group(0).lower()]
        return m.group(0)

    out = regex.sub(_sub, str(text))
    return out, stats["m"], stats["r"], stats["hits"]


def perturb_text(text, mapping=None):
    """Backward-compatible: full-rate, one direction (default expansion->abbreviation)."""
    mapping = mapping or SAFE_SYNONYMS
    return perturb_counted(text, mapping)[0]


def perturb_pairs(df, columns=("resume_text", "job_text"), mapping=None):
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda t: perturb_text(t, mapping))
    out["perturbed"] = True
    return out
