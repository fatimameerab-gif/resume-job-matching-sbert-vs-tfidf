"""Review 3.5, CPU part: the stronger robustness suite (ESCO synonyms, typos, reordering) for TF-IDF head and TF-IDF cosine on the five frozen seeds.
Dense models (frozen and fine-tuned) run the SAME suite on the GPU: gpu_robustness_all.py.  Output: results/robust_stronger/<model>/ .  Run: python rq2_robustness_stronger_cpu.py"""
import numpy as np, pandas as pd
import colab_v7b_finetune_hybrid as ft
from colab_common import TfidfBackend
from robust_suite import run_suite
from sklearn.model_selection import GroupShuffleSplit


class TfidfModel:
    def __init__(self, tr, seed, head):
        i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, groups=tr.resume_id)); a, v = tr.iloc[i_tr].reset_index(drop=True), tr.iloc[i_va].reset_index(drop=True)
        self.tf = TfidfBackend(pd.concat([tr.resume_text, tr.job_text])); self.head = head
        if head:
            X, _ = ft._pair_feats(self.tf, tr); self.m = ft._head(X, tr.label.values, seed); self.thr = 0.5
        else:
            _, cv = ft._pair_feats(self.tf, v); self.thr = ft._best_threshold(cv, v.label.values)
    def score(self, df):
        X, c = ft._pair_feats(self.tf, df); return self.m.predict_proba(X)[:, 1] if self.head else c


if __name__ == "__main__":
    for head, name in ((True, "TF-IDF head"), (False, "TF-IDF cosine")):
        S = run_suite(lambda tr, seed, h=head: TfidfModel(tr, seed, h), name, f"results/robust_stronger/{name.replace(' ', '_')}")
    pd.set_option("display.width", 220)
    print(S[S.side == "both"].to_string(index=False))
    from checklist_suite import run_checklist
    for head, name in ((True, "TF-IDF head"), (False, "TF-IDF cosine")):
        C = run_checklist(lambda tr, seed, h=head: TfidfModel(tr, seed, h), name, f"results/robust_stronger/{name.replace(' ', '_')}")
        print(C.to_string(index=False))
