import os, glob, sys, subprocess, traceback
try:
    import rank_bm25  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rank_bm25"])
import colab_export_embeddings, colab_v7b_robustness_sbert as rob

def _done(folder):
    return bool(glob.glob(os.path.join(folder, "*.csv")) or glob.glob(os.path.join(folder, "*.npy")) or glob.glob(os.path.join(folder, "*.npz")))

STEPS = (("embeddings", "results/embeddings", lambda: colab_export_embeddings.main()),
         ("robustness MiniLM+mpnet", "results/rq2_dense", lambda: rob.main(encoders="SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2", repeats=30, out="results/rq2_dense")),
         ("robustness other encoders", "results/rq2_dense_more", lambda: rob.main(encoders="e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2", repeats=10, out="results/rq2_dense_more")))
failed = []
for title, folder, fn in STEPS:
    if _done(folder):
        print(f"[skip] {title}: results already in {folder}", flush=True); continue
    print("=" * 20, title, flush=True)
    try: fn()
    except BaseException: traceback.print_exc(); failed.append(title)
print("FAILED STEPS:", failed or "none")
print("Now: !zip -r rq2_results.zip results/embeddings results/rq2_dense results/rq2_dense_more")
