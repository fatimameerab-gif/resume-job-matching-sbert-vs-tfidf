"""Paraphrase robustness for TF-IDF (head and cosine) on the CPU, same code path as the dense models (gpu_paraphrase.evaluate).
Needs results/paraphrases/paraphrased_texts.csv from gpu_paraphrase.generate().  Output: results/paraphrase_robust/TF-IDF_head, TF-IDF_cosine.  Run: python rq2_paraphrase_cpu.py"""
import pandas as pd
import robust_suite as RS
import paraphrase_lib as PL
from rq2_robustness_stronger_cpu import TfidfModel
from gpu_paraphrase import SETTINGS

if __name__ == "__main__":
    RS.set_paraphrases(PL.load_map())
    for head, name in ((True, "TF-IDF head"), (False, "TF-IDF cosine")):
        S = RS.run_suite(lambda tr, seed, h=head: TfidfModel(tr, seed, h), name, f"results/paraphrase_robust/{name.replace(' ', '_')}", repeats=1, settings=SETTINGS)
        pd.set_option("display.width", 220); print(S.to_string(index=False))
