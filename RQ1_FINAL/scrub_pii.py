"""Dry-run by default: counts identifier-like strings in the CLEANED resume texts of the pair files. `python scrub_pii.py --apply` replaces them
with placeholders in every pair file under data/ that has a resume_text column (backup *.bak first) and writes data/PII_SCRUBBED; data_protocol_v2 then applies the
same scrub to the pool texts, so pair texts and pool texts stay identical (embeddings are looked up by the hash of the text).
The raw data/UpdatedResumeDataSet.csv is NOT modified (cleaning would make scrubbing it differ from scrubbing the cleaned text).
Does NOT remove personal names. Re-run the TF-IDF baselines afterwards: numbers may move in the 3rd decimal."""
import sys, glob, shutil, pandas as pd
from pii_scrub import scrub
apply = "--apply" in sys.argv
for f in ["data/resume_job_pairs_v3.csv"] + glob.glob("data/eval_v7/**/*.csv", recursive=True):
    df = pd.read_csv(f)
    if "resume_text" not in df.columns: continue
    new = df["resume_text"].astype(str).map(scrub); n = int((new != df["resume_text"].astype(str)).sum())
    print(f, "rows changed:", n)
    if apply and n:
        shutil.copy(f, f + ".bak"); df["resume_text"] = new; df.to_csv(f, index=False)
if apply: open("data/PII_SCRUBBED", "w").write("pair files and pool resume texts scrubbed with pii_scrub.scrub\n"); print("marker written: data/PII_SCRUBBED")
