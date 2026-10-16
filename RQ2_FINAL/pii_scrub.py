"""Pattern-based identifier scrubbing (shared by scrub_pii.py and data_protocol_v2.py). Does NOT remove personal names."""
import re
PATS = [(r"\b0?\d{10}\b|\b91\d{10}\b", " PHONE "), (r"\b[a-z]{5}\d{4}[a-z]\b", " PANID "), (r"\bpassport\s+[a-z]\d{7}\b", " passport PASSID "),
        (r"linkedin\.com\s+in\s+[a-z]+\s+[a-z]+\s+[a-z0-9]+", " LINKEDIN "), (r"\b[\w.]+\s+(gmail|yahoo|hotmail|outlook)\s+com\b", " EMAIL ")]
def scrub(t):
    for p, r in PATS: t = re.sub(p, r, t)
    return t
