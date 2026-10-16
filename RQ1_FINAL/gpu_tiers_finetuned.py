import os, time, gc
import numpy as np, pandas as pd
from scipy import sparse
import colab_v7b_tiers_sbert as T
import colab_v7b_finetune_hybrid as ft
from colab_common import load_encoders, DenseIndex, ENCODERS, Encoder
from data_protocol_v2 import build_deduped_resumes, build_deduped_job_pool
from eval_protocol_v7 import fit_hard_index
from tier_eval_v7b import build_tier_pairs, NEG_REUSE_CAP
MINERS = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,bge-base-en-v1.5"; FT_ENCODERS = ("SBERT-mpnet-base-v2", "SBERT-MiniLM-L6-v2", "JobBERT-v2")


def main(out="results/colab_v7b_tiers_finetuned", miners=MINERS, ft_encoders=FT_ENCODERS, seeds=5):
    os.makedirs(out, exist_ok=True); encs = load_encoders(miners, chunk_long=True); resumes, pool = build_deduped_resumes(), build_deduped_job_pool(); hv, hm = fit_hard_index(pool)
    jobs_txt, res_txt = pool["text_clean"].fillna("").tolist(), resumes["text_clean"].tolist(); idx = {"TF-IDF": (hv, hm)}; mats = []
    for e in encs:
        e.encode(res_txt, "resume"); m = sparse.csr_matrix(e.encode(jobs_txt, "job")); mats.append(m); idx[e.key] = (DenseIndex(e), m)
    sp = [T._spread(hv.transform(res_txt), hm)] + [T._spread(e.encode(res_txt, "resume"), m) for e, m in zip(encs, mats)]
    w = [1.0 / np.sqrt(s) for s in sp]; idx["consensus"] = (T.ConsensusIndex(hv, encs, w), sparse.hstack([w[0] * hm] + [w[i + 1] * mats[i] for i in range(len(encs))]).tocsr())
    fam = [0.5] + [0.5 / len(encs)] * len(encs); wf = [float(np.sqrt(f / s)) for f, s in zip(fam, sp)]
    idx["consensus_family"] = (T.ConsensusIndex(hv, encs, wf), sparse.hstack([wf[0] * hm] + [wf[i + 1] * mats[i] for i in range(len(encs))]).tocsr())
    path = f"{out}/tiers_ft_per_seed.csv"; rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []; done = {r["seed"] for r in rows}; t0 = time.time()
    for seed in range(seeds):
        if seed in done: continue
        tiers = build_tier_pairs(seed, resumes, pool, hv, hm, NEG_REUSE_CAP); tiers["hard[TF-IDF]"] = tiers.pop("hard_mined")
        for name, (v, m) in idx.items():
            if name != "TF-IDF": tiers[f"hard[{name}]"] = build_tier_pairs(seed, resumes, pool, v, m, NEG_REUSE_CAP)["hard_mined"]
        tr = tiers["random_other_category"]; tr = tr[tr.split == "train"]; heads = [T.TfidfHeads(tr)] + [T.EncHeads(e, tr) for e in encs]
        for k in ft_encoders:
            model = ft._finetune(k, tr, seed, ft.EPOCHS, ft.BATCH_SIZE, ft.LEARNING_RATE); e = Encoder(f"FT[{k}]", ENCODERS[k][0], ENCODERS[k][1], ENCODERS[k][2], chunk_long=True, model=model)
            e.key = f"{k} fine-tuned"; heads.append(T.EncHeads(e, tr))
            try:
                import torch; gc.collect(); torch.cuda.empty_cache()
            except Exception: pass
        for tier, df in tiers.items():
            te = df[df.split == "test"]
            for h in heads: rows.append(T._row(h.name, tier, seed, tr, te, *h.scores(te)))
        pd.DataFrame(rows).to_csv(path, index=False); print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows); S = T.summarise(per); S.round(4).to_csv(f"{out}/tiers_ft_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for m in ("zeroshot_cos_auc_mean", "full_auc_mean"): print(f"\n=== {m} ==="); print(S.pivot(index="tier", columns="model", values=m).round(3).to_string())


if __name__ == "__main__":
    main()
