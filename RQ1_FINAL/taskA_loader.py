import os
import pandas as pd

BASE_DIR = "data/taskA_en"


def _read_text_dir(path):
    out = {}
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                out[fname] = f.read()
    return out


def load_taskA_pairs(base_dir=BASE_DIR):
    """Returns a DataFrame with one row per (query_id, resume_id) pair,
    covering the FULL query x corpus grid (every query paired with every
    candidate resume), with:
      query_id, resume_id, job_text, resume_text, label (1 if in qrels else 0)

    ⚠️  See module docstring for the provenance / citation flag that must
    be resolved before this data can appear in a submitted manuscript.
    """
    # PROVENANCE.txt is now present in data/taskA_en/ (placed 2026-09-29).
    # The [VERIFY] checklist inside that file still requires human confirmation
    # (byte-level match against Zenodo, licence verification, etc.) before
    # results can appear in a submitted manuscript. See the docstring above
    # and data/taskA_en/PROVENANCE.txt for the full checklist.
    if not os.path.exists(os.path.join(base_dir, "PROVENANCE.txt")):
        import warnings
        warnings.warn(
            "data/taskA_en/PROVENANCE.txt not found. "
            "The source, licence, and citation for this dataset have not been "
            "verified. Do not include TaskA results in a manuscript submission "
            "until provenance is confirmed. See taskA_loader.py docstring.",
            UserWarning,
            stacklevel=2,
        )
    queries = _read_text_dir(os.path.join(base_dir, "queries"))
    corpus = _read_text_dir(os.path.join(base_dir, "corpus"))

    qrels_df = pd.read_csv(os.path.join(base_dir, "qrels.tsv"), sep="\t", header=None,
                            names=["query_id", "iteration", "doc_id", "relevance"], dtype=str)
    positive_pairs = set(zip(qrels_df["query_id"], qrels_df["doc_id"]))

    rows = []
    for qid, job_text in queries.items():
        for did, resume_text in corpus.items():
            label = 1 if (qid, did) in positive_pairs else 0
            rows.append({
                "query_id": qid, "resume_id": did,
                "job_text": job_text, "resume_text": resume_text,
                "label": label,
            })
    pairs = pd.DataFrame(rows)
    n_pos = pairs["label"].sum()
    print(f"TaskA (English, dev): {len(queries)} queries x {len(corpus)} resumes "
          f"= {len(pairs)} pairs ({n_pos} positive, {len(pairs)-n_pos} assumed-negative, "
          f"{100*n_pos/len(pairs):.1f}% positive rate)")
    return pairs


if __name__ == "__main__":
    pairs = load_taskA_pairs()
    print(pairs.head())
    print(pairs.groupby("query_id")["label"].sum())
