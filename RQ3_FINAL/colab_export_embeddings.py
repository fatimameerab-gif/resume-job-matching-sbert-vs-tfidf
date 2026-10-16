import os, hashlib, numpy as np, pandas as pd
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
OUT_DIR = "results/embeddings"
from colab_common import load_encoders, find_seed_dir

def sha(t): return hashlib.sha1(str(t).encode("utf8")).hexdigest()

def main(encoders=ENCODERS_TO_RUN, out=OUT_DIR, seed_dir=None, pairs_path="data/resume_job_pairs_v3.csv"):
    os.makedirs(out, exist_ok=True); sd = find_seed_dir(seed_dir); frames = [pd.read_csv(os.path.join(sd, f"pairs_seed{s}.csv")) for s in range(5)]
    if os.path.exists(pairs_path): frames.append(pd.read_csv(pairs_path))
    d = pd.concat(frames); res = set(d.resume_text.astype(str)); job = set(d.job_text.astype(str))
    try:        # also every resume / job text of the full pools (needed by rq2_cross_source.py with dense encoders)
        from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
        res |= set(build_deduped_resumes().text_clean.astype(str)); job |= set(build_deduped_job_pool().text_clean.astype(str))
    except Exception as e:
        print("[note] full pools not added:", type(e).__name__, e)
    res, job = sorted(res), sorted(job)
    print(f"{len(res)} unique resume texts, {len(job)} unique job texts", flush=True)
    for enc in load_encoders(encoders, chunk_long=True):
        keys = [sha(t) for t in res] + [sha(t) for t in job]
        vec = np.vstack([enc.encode(res, "resume"), enc.encode(job, "job")]).astype(np.float32)
        np.savez_compressed(f"{out}/{enc.key}.npz", keys=np.array(keys), vectors=vec)
        print(f"saved {enc.key}: {vec.shape}  | {enc.report()}", flush=True)

if __name__ == "__main__":
    main()
