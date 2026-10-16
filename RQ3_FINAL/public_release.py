import sys, shutil, io, glob, os, contextlib
import pandas as pd

RAW = ["data/UpdatedResumeDataSet.csv", "data/data_science_job_postings_672.csv"]
SKIP = {".venv", "venv", "__pycache__", ".pytest_cache", ".git"}


def kaggle_job_ids():
    import data_protocol_v2 as dp
    with contextlib.redirect_stdout(io.StringIO()):
        pool = dp.build_deduped_job_pool()
    return set(pool.loc[pool.job_source == "kaggle_672", "job_id"])


def main(out):
    src = os.getcwd()
    out = os.path.abspath(out)
    assert not out.startswith(src + os.sep), "choose an output folder outside this folder"
    kid = kaggle_job_ids()
    shutil.copytree(src, out, ignore=lambda d, names: [n for n in names if n in SKIP or n.startswith("RQ2_09")])
    for r in RAW:
        if os.path.exists(os.path.join(out, r)):
            os.remove(os.path.join(out, r))
    n = 0
    for f in glob.glob(os.path.join(out, "**", "*.csv"), recursive=True):
        try:
            head = pd.read_csv(f, nrows=0).columns
        except Exception:
            continue
        cols = [c for c in ("resume_text", "job_text") if c in head]
        if not cols:
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if "resume_text" in df:
            df["resume_text"] = ""
        if "job_text" in df:
            if "job_id" in df and "resume_id" in df:
                df.loc[df["job_id"].astype(int).isin(kid), "job_text"] = ""
            else:
                df["job_text"] = ""
        df.to_csv(f, index=False)
        n += 1
    print(f"public copy written to {out}; texts removed in {n} CSV files; raw Kaggle files deleted")


if __name__ == "__main__":
    main(sys.argv[1])
