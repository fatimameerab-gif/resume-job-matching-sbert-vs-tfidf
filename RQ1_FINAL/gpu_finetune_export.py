import os, hashlib, numpy as np
import colab_v7b_finetune_hybrid as ft
from colab_common import ENCODERS, Encoder
from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
ALL6 = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
sha = lambda t: hashlib.sha1(str(t).encode("utf8")).hexdigest()
_orig = ft._finetune


def _finetune_and_export(enc_key, tr, seed, epochs, batch, lr):
    model = _orig(enc_key, tr, seed, epochs, batch, lr)
    path = f"results/ft_embeddings/{enc_key}_seed{seed}.npz"
    if not os.path.exists(path):
        os.makedirs("results/ft_embeddings", exist_ok=True)
        R, J = build_deduped_resumes().text_clean.astype(str).tolist(), build_deduped_job_pool().text_clean.astype(str).tolist()
        e = Encoder(f"FT[{enc_key}]", ENCODERS[enc_key][0], ENCODERS[enc_key][1], ENCODERS[enc_key][2], chunk_long=True, model=model)
        V = np.vstack([e.encode(R, "resume"), e.encode(J, "job")]).astype(np.float16)
        np.savez_compressed(path + ".tmp.npz", keys=np.array([sha(t) for t in R] + [sha(t) for t in J]), vectors=V); os.replace(path + ".tmp.npz", path); print(f"[export] {path} {V.shape}", flush=True)
    return model


def main(encoders=ALL6, out="results/colab_v7b_finetune_pinned"):
    """The export hook is active ONLY while this stage runs. It writes results/ft_embeddings/<encoder>_seed<k>.npz, so any file left over from an
    earlier (partly failed) run is deleted first: other stages (leave-category-out, Design B, ...) also fine-tune with seeds 0..24 and must never
    share or pre-empt these file names."""
    import shutil
    shutil.rmtree("results/ft_embeddings", ignore_errors=True)
    ft._finetune = _finetune_and_export
    try:
        ft.main(encoders=encoders, seeds=5, finetune=True, out=out)
    finally:
        ft._finetune = _orig           # restore: later stages must not export under these names


if __name__ == "__main__":
    main()
