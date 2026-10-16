import sys, os
D = sys.argv[1] if len(sys.argv) > 1 else "."
def edit(fn, pairs):
    p = os.path.join(D, fn); s = open(p).read(); done = False
    for old, new in pairs:
        if new in s: continue                       # already patched
        assert old in s, f"{fn}: pattern not found:\n{old[:80]}"
        s = s.replace(old, new, 1); done = True
    open(p, "w").write(s); print(("patched " if done else "already patched ") + fn)

HELPER = '''

def load_st(model_id):
    """Load a SentenceTransformer and force float32 weights.
    Newer transformers versions load a checkpoint in its stored dtype; e.g. thenlper/gte-base is stored in float16, which breaks
    GradScaler ("Attempting to unscale FP16 gradients") and would make that one encoder differ in numeric precision from the others.
    Casting float16 -> float32 is exact; for checkpoints already stored in float32 this is a no-op."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    dt = {str(p.dtype) for p in m.parameters()}
    if dt != {"torch.float32"}:
        print(f"  [note] {model_id} was loaded with parameter dtype(s) {sorted(dt)}; cast to float32", flush=True)
        m = m.float()
    return m

'''
edit("colab_common.py", [
    ("\nclass Encoder:", HELPER + "\nclass Encoder:"),
    ("            from sentence_transformers import SentenceTransformer   # lazy: not needed for stub\n            self.model = SentenceTransformer(model_id)",
     "            self.model = load_st(model_id)   # lazy import inside; float32 enforced")])
edit("colab_v7b_finetune_hybrid.py", [
    ("TfidfBackend, ENCODERS, Encoder\n", "TfidfBackend, ENCODERS, Encoder, load_st\n"),
    ("    model = SentenceTransformer(ENCODERS[enc_key][0])\n", "    model = load_st(ENCODERS[enc_key][0])\n")])
edit("colab_v7b_taskA_chunked.py", [
    ("from colab_common import ENCODERS, Encoder, TfidfBackend, load_taskA", "from colab_common import ENCODERS, Encoder, TfidfBackend, load_taskA, load_st"),
    ("                        from sentence_transformers import SentenceTransformer\n                        enc_models[\"m\"] = SentenceTransformer(ENCODERS[key][0])",
     "                        enc_models[\"m\"] = load_st(ENCODERS[key][0])")])
p = os.path.join(D, "gpu_finetune_export.py"); s = open(p).read()
if "finally:" in s: print("already patched gpu_finetune_export.py")
else:
    a = s.index("def main("); b = s.index("if __name__")
    new = '''def main(encoders=ALL6, out="results/colab_v7b_finetune_pinned"):
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


'''
    open(p, "w").write(s[:a] + new + s[b:]); print("patched gpu_finetune_export.py")
