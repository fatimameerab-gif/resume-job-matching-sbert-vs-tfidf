import hashlib
import os
import pandas as pd


def _text_key(s):
    s = "" if pd.isna(s) else str(s)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def diagnose_synthetic_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits = sorted(pairs["split"].unique())
    for a, b in [(x, y) for i, x in enumerate(splits) for y in splits[i + 1:]]:
        ra, rb = set(pairs.loc[pairs["split"] == a, "resume_id"]), set(pairs.loc[pairs["split"] == b, "resume_id"])
        ja, jb = set(pairs.loc[pairs["split"] == a, "job_id"]), set(pairs.loc[pairs["split"] == b, "job_id"])
        ta = set(pairs.loc[pairs["split"] == a, "resume_text"].map(_text_key))
        tb = set(pairs.loc[pairs["split"] == b, "resume_text"].map(_text_key))
        jta = set(pairs.loc[pairs["split"] == a, "job_text"].map(_text_key))
        jtb = set(pairs.loc[pairs["split"] == b, "job_text"].map(_text_key))
        rows.append({
            "dataset": "synthetic_pairs_v2",
            "split_a": a, "split_b": b,
            "resume_id_overlap": len(ra & rb),
            "resume_text_overlap": len(ta & tb),
            "job_id_overlap": len(ja & jb),
            "job_text_overlap": len(jta & jtb),
            "n_pairs_a": int((pairs["split"] == a).sum()),
            "n_pairs_b": int((pairs["split"] == b).sum()),
        })

    n_job_ids = pairs["job_id"].nunique()
    n_job_texts = pairs["job_text"].map(_text_key).nunique()
    dup_jobs = n_job_ids - n_job_texts
    rows.append({
        "dataset": "synthetic_pairs_v2",
        "split_a": "ALL", "split_b": "ALL",
        "resume_id_overlap": 0,
        "resume_text_overlap": 0,
        "job_id_overlap": 0,
        "job_text_overlap": 0,
        "n_unique_job_ids": n_job_ids,
        "n_unique_job_texts": n_job_texts,
        "job_ids_minus_unique_texts": dup_jobs,
        "note": "job_id is assigned after dedup on (text_clean, mapped_category); "
                "identical text with two categories counts as two job_ids.",
    })
    return pd.DataFrame(rows)


def diagnose_taskA(pairs: pd.DataFrame) -> pd.DataFrame:
    n_q = pairs["query_id"].nunique()
    n_r = pairs["resume_id"].nunique()
    n_job_text = pairs.drop_duplicates("query_id")["job_text"].map(_text_key).nunique()
    n_res_text = pairs.drop_duplicates("resume_id")["resume_text"].map(_text_key).nunique()
    return pd.DataFrame([{
        "dataset": "taskA_en",
        "n_queries": n_q,
        "n_resume_ids": n_r,
        "n_unique_job_texts": n_job_text,
        "n_unique_resume_texts": n_res_text,
        "duplicate_job_texts_as_queries": n_q - n_job_text,
        "duplicate_resume_texts": n_r - n_res_text,
        "note": "Leave-one-query-out trains on other queries' pairs. The same resume "
                "document appears in every fold (paired with a different query). "
                "qrels are positive-only; unjudged pairs are treated as negatives.",
    }])


def assert_resume_disjoint(pairs: pd.DataFrame):
    by_split = {s: set(g["resume_id"]) for s, g in pairs.groupby("split")}
    for a in by_split:
        for b in by_split:
            if a >= b:
                continue
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(f"resume_id overlap {a} vs {b}: {sorted(overlap)[:8]}")


def write_reports(pairs_path="data/resume_job_pairs_v3.csv", out_dir="results/v7"):
    os.makedirs(out_dir, exist_ok=True)
    syn = pd.read_csv(pairs_path)
    assert_resume_disjoint(syn)
    syn_rep = diagnose_synthetic_pairs(syn)
    syn_rep.to_csv(os.path.join(out_dir, "leakage_synthetic_pairs.csv"), index=False)

    from taskA_loader import load_taskA_pairs
    task = load_taskA_pairs()
    ta_rep = diagnose_taskA(task)
    ta_rep.to_csv(os.path.join(out_dir, "leakage_taskA.csv"), index=False)
    return syn_rep, ta_rep


if __name__ == "__main__":
    s, t = write_reports()
    print(s.to_string(index=False))
    print(t.to_string(index=False))
