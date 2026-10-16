"""Writes results/ENVIRONMENT.txt: library versions, GPU, and how JobBERT-v2 routes plain strings (review 3.6). Called at the start of run_everything_gpu.py.
Pinned environment used for every GPU number of the revised paper: see requirements_gpu_pinned.txt."""
import os, sys, platform, importlib


def report(out="results/ENVIRONMENT.txt", check_jobbert=True):
    L = [f"python {sys.version.split()[0]} on {platform.platform()}"]
    for m in ("torch", "transformers", "sentence_transformers", "sklearn", "scipy", "numpy", "pandas", "rank_bm25"):
        try: L.append(f"{m} {getattr(importlib.import_module(m), '__version__', 'installed (no version attribute)')}")
        except Exception as e: L.append(f"{m} NOT AVAILABLE ({type(e).__name__})")
    try:
        import torch
        L.append("CUDA " + (f"available: {torch.cuda.get_device_name(0)}, torch.version.cuda={torch.version.cuda}" if torch.cuda.is_available() else "NOT available (CPU only)"))
    except Exception: pass
    if check_jobbert:
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("TechWolf/JobBERT-v2"); L.append(f"JobBERT-v2 modules: {[type(x).__name__ for x in m]}; max_seq_length reported by the loaded model: {m.max_seq_length}")
            for x in m:
                if hasattr(x, "sub_modules") or hasattr(x, "default_route"):
                    L.append(f"JobBERT-v2 routing module {type(x).__name__}: routes={list(getattr(x, 'sub_modules', {}).keys())}, default_route={getattr(x, 'default_route', None)} "
                             "(plain strings, as used for BOTH resumes and jobs here, go to the default/first route)")
        except Exception as e: L.append(f"JobBERT-v2 check failed: {type(e).__name__}: {e}")
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True); return L


if __name__ == "__main__": report()
