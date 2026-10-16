import gc
import robust_suite as RS
import paraphrase_lib as PL
from gpu_robustness_all import DenseModel, FROZEN, FINETUNED

SETTINGS = [("paraphrase", 1.0)]


def generate(**kw):
    D = PL.generate(**kw)
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return D


def evaluate(frozen=tuple(FROZEN), finetuned=FINETUNED, cosine_for=("SBERT-mpnet-base-v2",), out_root="results/paraphrase_robust"):
    RS.set_paraphrases(PL.load_map())
    jobs = [(k, False, True) for k in frozen] + [(k, True, True) for k in finetuned] + [(k, False, False) for k in cosine_for] + [(k, True, False) for k in cosine_for]
    for key, fine, head in jobs:
        name = f"{key} {'fine-tuned' if fine else 'frozen'} {'head' if head else 'cosine'}"; out = f"{out_root}/{name.replace(' ', '_')}"
        mk = lambda tr, seed, key=key, fine=fine, head=head: DenseModel(tr, seed, key, fine, head)
        RS.run_suite(mk, name, out, repeats=1, settings=SETTINGS); print("finished", name, flush=True)
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass


if __name__ == "__main__":
    generate(); evaluate()
