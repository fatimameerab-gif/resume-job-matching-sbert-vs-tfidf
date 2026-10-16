import os, gc, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import colab_v7b_finetune_hybrid as ft
from colab_common import load_encoders, ENCODERS, Encoder
from robust_suite import run_suite
from checklist_suite import run_checklist
FROZEN = {"SBERT-MiniLM-L6-v2": 10, "SBERT-mpnet-base-v2": 10, "JobBERT-v2": 10, "e5-base-v2": 5, "bge-base-en-v1.5": 5, "gte-base": 5}
FINETUNED = ("SBERT-mpnet-base-v2", "SBERT-MiniLM-L6-v2", "JobBERT-v2")
_ENC_CACHE = {}


def frozen_encoder(key):
    if key not in _ENC_CACHE: _ENC_CACHE[key] = load_encoders(key, chunk_long=True)[0]
    return _ENC_CACHE[key]


class DenseModel:
    def __init__(self, tr_full, seed, key, finetuned, head):
        i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr_full, groups=tr_full.resume_id)); tr, va = tr_full.iloc[i_tr].reset_index(drop=True), tr_full.iloc[i_va].reset_index(drop=True)
        self.head = head
        if finetuned:
            m = ft._finetune(key, tr, seed, ft.EPOCHS, ft.BATCH_SIZE, ft.LEARNING_RATE)
            self.enc = Encoder(f"FT[{key}]", ENCODERS[key][0], ENCODERS[key][1], ENCODERS[key][2], chunk_long=True, model=m)
        else: self.enc = frozen_encoder(key)
        X, _ = ft._pair_feats(self.enc, tr)
        if head: self.m = ft._head(X, tr.label.values, seed); self.thr = 0.5
        else:
            _, cv = ft._pair_feats(self.enc, va); self.thr = ft._best_threshold(cv, va.label.values)
    def score(self, df):
        X, c = ft._pair_feats(self.enc, df); return self.m.predict_proba(X)[:, 1] if self.head else c


def main(frozen=FROZEN, finetuned=FINETUNED, cosine_for=("SBERT-mpnet-base-v2",)):
    jobs = [(k, False, True, r) for k, r in frozen.items()] + [(k, True, True, 10) for k in finetuned]
    jobs += [(k, False, False, 10) for k in cosine_for] + [(k, True, False, 10) for k in cosine_for]            # cosine (zero-shot / fine-tuned cosine) for mpnet
    for key, fine, head, reps in jobs:
        name = f"{key} {'fine-tuned' if fine else 'frozen'} {'head' if head else 'cosine'}"; out = f"results/robust_stronger/{name.replace(' ', '_')}"
        mk = lambda tr, seed, key=key, fine=fine, head=head: DenseModel(tr, seed, key, fine, head)
        run_suite(mk, name, out, repeats=reps); run_checklist(mk, name, out); print("finished", name, flush=True)
        if fine:
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
        if not fine and len(_ENC_CACHE) > 3:                      # keep GPU memory in check: drop encoders we no longer need
            for k in list(_ENC_CACHE)[:-2]: _ENC_CACHE.pop(k, None)


if __name__ == "__main__":
    main()
