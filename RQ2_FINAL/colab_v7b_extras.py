import os, time, types
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2,GloVe-avg-300d"
SEEDS = 5
NAME_SEEDS = 2             # name-swap probe only on the first N seeds (it re-encodes every resume once per name = slow)
SEED_DIR = None            # None = auto-find pairs_seed0..4.csv
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # None to skip
OUT_DIR = "results/colab_v7b_extras"
# Name groups for the counterfactual test (first + last name). Edit freely; report the lists in the paper.
NAME_GROUPS = {
    "male_A":   ["James Miller", "Michael Johnson", "Robert Smith", "David Brown"],
    "female_A": ["Emily Miller", "Sarah Johnson", "Jessica Smith", "Laura Brown"],
    "male_B":   ["Jamal Washington", "Darnell Jackson", "Tyrone Robinson", "Malik Jefferson"],
    "female_B": ["Lakisha Washington", "Latoya Jackson", "Tanisha Robinson", "Aaliyah Jefferson"],
    # CONTROL: a neutral, name-free prefix of similar length. Subtract its shift from the name groups: the control isolates the effect of merely adding a prefix.
    "neutral_control": ["Candidate Profile", "Applicant Resume", "Resume Document", "Candidate Summary"],
}
# ========================================================
from colab_common import load_encoders, find_seed_dir, load_seed_pairs, TfidfBackend

def _lr(X, y): return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)

def _feats(U, V, mode):
    cos = (U * V).sum(1, keepdims=True)
    if mode == "cos": return cos
    if mode == "abs+cos": return np.hstack([np.abs(U - V), cos])
    if mode == "uv_abs_prod": return np.hstack([U, V, np.abs(U - V), U * V])
    raise ValueError(mode)

MODES = ["cos", "abs+cos", "uv_abs_prod"]

def _mrr_hit(te, score):
    d = te[["resume_id", "label"]].copy(); d["s"] = np.asarray(score); hit, rr = [], []
    for _, g in d.groupby("resume_id"):
        if g.label.sum() != 1: continue
        s = g.s[g.label == 1].iloc[0]; oth = g.s[g.label == 0].values
        rank = 1 + (oth > s).sum() + 0.5 * (oth == s).sum()
        hit.append(float(rank == 1)); rr.append(1 / rank)
    return float(np.mean(hit)), float(np.mean(rr))

def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, out=OUT_DIR, seed_dir=SEED_DIR, cross=CROSS_ENCODER):
    os.makedirs(out, exist_ok=True)
    sd = find_seed_dir(seed_dir)
    encs = load_encoders(encoders, chunk_long=False)       # frozen, head truncation (as in the first tier runs)
    abl, eff, ce_rows, cf_rows = [], [], [], []
    for seed in range(seeds):
        tr, te = load_seed_pairs(sd, seed)
        trr = tr[tr.negative_reason.isin(["positive", "random_other_category"])] if "negative_reason" in tr else tr
        backs = [TfidfBackend(pd.concat([tr.resume_text, tr.job_text]))] + encs
        for b in backs:
            Utr, Vtr = b.encode(trr.resume_text, "resume"), b.encode(trr.job_text, "job")
            Ute, Vte = b.encode(te.resume_text, "resume"), b.encode(te.job_text, "job")
            for mode in MODES:
                clf = _lr(_feats(Utr, Vtr, mode), trr.label)
                sc = clf.predict_proba(_feats(Ute, Vte, mode))[:, 1]
                hit, mrr = _mrr_hit(te, sc)
                abl.append(dict(seed=seed, model=b.name, head=mode, acc=accuracy_score(te.label, sc > .5),
                                auc=roc_auc_score(te.label, sc), hit1=hit, mrr=mrr))
            # D) counterfactual name swap (uses the 'abs+cos' head, the pipeline's head)
            print(f'  seed {seed} / {b.name}: ablation done', flush=True)
            if seed < NAME_SEEDS:
                clf = _lr(_feats(Utr, Vtr, "abs+cos"), trr.label); base = clf.predict_proba(_feats(Ute, Vte, "abs+cos"))[:, 1]
                zs = (Ute * Vte).sum(1)
                for grp, names in NAME_GROUPS.items():
                    print(f'    name-swap {b.name} {grp}', flush=True)
                    sw, swz = [], []
                    for nm in names:
                        Un = b.encode([f"{nm}. " + t for t in te.resume_text], "resume")
                        sw.append(clf.predict_proba(_feats(Un, Vte, "abs+cos"))[:, 1]); swz.append((Un * Vte).sum(1))
                    sw, swz = np.mean(sw, 0), np.mean(swz, 0)
                    cf_rows.append(dict(seed=seed, model=b.name, name_group=grp,
                                        mean_head_score_shift=float((sw - base).mean()),
                                        mean_cosine_shift=float((swz - zs).mean()),
                                        decision_flip_rate=float(((sw > .5) != (base > .5)).mean())))
    A = pd.DataFrame(abl); A.to_csv(f"{out}/head_ablation_per_seed.csv", index=False)
    S = A.groupby(["model", "head"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), auc=("auc", "mean"),
                                         hit1=("hit1", "mean"), mrr=("mrr", "mean")).reset_index()
    S.to_csv(f"{out}/head_ablation_summary.csv", index=False)
    C = pd.DataFrame(cf_rows); C.to_csv(f"{out}/name_swap_per_seed.csv", index=False)
    CS = C.groupby(["model", "name_group"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    # gap between name groups (max - min of mean head-score shift) per model = the bias-style number to report
    named = CS[CS.name_group != "neutral_control"]
    gap = named.groupby("model").mean_head_score_shift.agg(lambda s: s.max() - s.min()).rename("group_gap_max_minus_min")
    ctl = CS[CS.name_group == "neutral_control"].set_index("model")[["mean_head_score_shift", "decision_flip_rate"]].add_prefix("control_")
    CS = CS.merge(ctl, left_on="model", right_index=True, how="left")
    CS["shift_minus_control"] = CS.mean_head_score_shift - CS.control_mean_head_score_shift
    CS = CS.merge(gap, on="model"); CS.to_csv(f"{out}/name_swap_summary.csv", index=False)

    # B) efficiency (seed-0 test texts)
    tr, te = load_seed_pairs(sd, 0)
    docs = list(dict.fromkeys(list(te.job_text) + list(te.resume_text)))[:200]
    for b in encs:
        try:
            import torch; cuda = torch.cuda.is_available()
            if cuda: torch.cuda.reset_peak_memory_stats()
        except Exception: cuda = False
        base_mb = torch.cuda.memory_allocated() / 1e6 if cuda else np.nan          # all loaded models are resident: report ACTIVATION peak above this base, plus the model's own size
        b.cache = {}; t0 = time.time(); E = b.encode(docs, "job"); dt = time.time() - t0
        jobs = b.encode(list(te.job_text)[:100], "job"); q = b.encode(list(te.resume_text)[:1], "resume")
        t1 = time.time(); [(jobs @ q[0]) for _ in range(200)]; lat = (time.time() - t1) / 200 * 1000
        eff.append(dict(model=b.name, n_docs=len(docs), sec_per_1k_docs=1000 * dt / len(docs), dim=int(E.shape[1]),
                        index_MB_per_100k_docs=float(E.shape[1] * 4 * 1e5 / 1e6), rank_100_jobs_ms=lat,
                        peak_activation_MB=float(torch.cuda.max_memory_allocated() / 1e6 - base_mb) if cuda else np.nan,
                        model_weights_MB=float(sum(p_.numel() * p_.element_size() for p_ in b.model.parameters()) / 1e6) if getattr(b, 'model', None) is not None and hasattr(b.model, 'parameters') else np.nan))
    pd.DataFrame(eff).to_csv(f"{out}/efficiency.csv", index=False)

    # C) cross-encoder zero-shot
    if cross:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(cross, max_length=512)
            for seed in range(seeds):
                _, te = load_seed_pairs(sd, seed); t0 = time.time()
                sc = ce.predict(list(zip(te.resume_text, te.job_text)), batch_size=16, show_progress_bar=False)
                hit, mrr = _mrr_hit(te, sc)
                ce_rows.append(dict(seed=seed, model=cross, auc=roc_auc_score(te.label, sc), hit1=hit, mrr=mrr,
                                    sec_per_1k_pairs=1000 * (time.time() - t0) / len(te)))
            pd.DataFrame(ce_rows).to_csv(f"{out}/cross_encoder.csv", index=False)
        except Exception as e:
            print(f"[SKIP] cross-encoder: {type(e).__name__}: {e}")
    open(f"{out}/NOTE_extras.txt", "w").write(
        "Head ablation: heads trained on random-negative train pairs, frozen encoders, truncation, no tuning.\n"
        "Cross-encoder is a zero-shot (MS MARCO) re-ranker, not trained on this task; it is a reference point only.\n"
        "Name swap: prepends a first+last name to each test resume; a synthetic probe, not a fairness audit. "
        "Report the name lists. group_gap is a score-shift gap between name groups; compare with seed noise.\n"
        "Efficiency numbers depend on the GPU/CPU used; state the hardware in the paper.\n")
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False)); print(CS.round(4).to_string(index=False))
    print(pd.DataFrame(eff).round(2).to_string(index=False))

if __name__ == "__main__":
    main()
