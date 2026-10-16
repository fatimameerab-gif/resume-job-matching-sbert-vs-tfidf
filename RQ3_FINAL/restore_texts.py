import io, glob, contextlib
import pandas as pd
import data_protocol_v2 as dp


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        res = dp.build_deduped_resumes()
        pool = dp.build_deduped_job_pool()
    rt = dict(zip(res.resume_id, res.text_clean))
    jt = dict(zip(pool.job_id, pool.text_clean))
    n = 0
    for f in glob.glob("data/**/*.csv", recursive=True):
        try:
            head = pd.read_csv(f, nrows=0).columns
        except Exception:
            continue
        if not {"resume_id", "job_id", "resume_text", "job_text"} <= set(head):
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df["resume_text"] = df["resume_id"].astype(int).map(rt)
        empty = df["job_text"] == ""
        df.loc[empty, "job_text"] = df.loc[empty, "job_id"].astype(int).map(jt)
        df.to_csv(f, index=False)
        n += 1
    print(f"texts restored in {n} CSV files")


if __name__ == "__main__":
    main()
