import os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")
# ======================= SETTINGS =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
SEEDS = 5
SEED_DIR = None
OUT_DIR = "results/colab_v7b_truncation"
STRATEGIES = ["head", "head_tail", "section_aware", "chunk_mean", "chunk_max"]
# ========================================================
from colab_common import load_encoders, find_seed_dir, load_seed_pairs
HEADINGS = r"\b(summary|objective|profile|skills?|technical skills|education(?: details)?|experience|work experience|projects?|certifications?|responsibilities|requirements|qualifications|about (?:the )?(?:role|us|you))\b"
STUB_LIMIT = 100

def _limit(enc): return STUB_LIMIT if enc.key == "stub" else int(enc.model.max_seq_length) - 8
def _ids(enc, text):
    return text.split() if enc.key == "stub" else enc.model.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
def _dec(enc, ids):
    return " ".join(ids) if enc.key == "stub" else enc.model.tokenizer.decode(ids, skip_special_tokens=True)

def pieces(enc, text, strategy, pref):
    L = _limit(enc) - (0 if enc.key == "stub" else len(enc.model.tokenizer(pref, add_special_tokens=False)["input_ids"])); ids = _ids(enc, text)
    if len(ids) <= L: return [pref + text]
    if strategy == "head": return [pref + _dec(enc, ids[:L])]
    if strategy == "head_tail":
        h = L * 2 // 3; return [pref + _dec(enc, ids[:h] + ids[-(L - h):])]
    if strategy in ("chunk_mean", "chunk_max"):
        return [pref + _dec(enc, ids[i:i + L]) for i in range(0, len(ids), L)]
    if strategy == "section_aware":
        cuts = [m.start() for m in re.finditer(HEADINGS, text)]
        if len(cuts) < 2: return [pref + _dec(enc, ids[:L])]          # no headings found: falls back to head
        cuts = sorted(set([0] + cuts)) + [len(text)]
        secs = [text[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if len(text[a:b].split()) >= 3]
        return [pref + _dec(enc, _ids(enc, s)[:L]) for s in secs] or [pref + _dec(enc, ids[:L])]
    raise ValueError(strategy)

def embed(enc, texts, kind, strategy):
    pref = enc.prefix[kind]; out = []
    for t in texts:
        ps = pieces(enc, str(t), strategy, pref); E = enc._embed(ps)
        v = E.max(0) if strategy == "chunk_max" else E.mean(0)
        out.append(v / max(np.linalg.norm(v), 1e-12))
    return np.vstack(out)

def _feat(U, V): return np.hstack([np.abs(U - V), (U * V).sum(1, keepdims=True)])
def _mrr_hit(te, s):
    d = te[["resume_id", "label"]].copy(); d["s"] = s; h, r = [], []
    for _, g in d.groupby("resume_id"):
        if g.label.sum() != 1: continue
        p = g.s[g.label == 1].iloc[0]; o = g.s[g.label == 0].values; k = 1 + (o > p).sum() + .5 * (o == p).sum(); h.append(float(k == 1)); r.append(1 / k)
    return np.mean(h), np.mean(r)

def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, out=OUT_DIR, seed_dir=SEED_DIR, strategies=STRATEGIES):
    os.makedirs(out, exist_ok=True); sd = find_seed_dir(seed_dir); rows = []
    for enc in load_encoders(encoders, chunk_long=False):
        for st in strategies:
            for seed in range(seeds):
                tr, te = load_seed_pairs(sd, seed)
                trr = tr[tr.negative_reason.isin(["positive", "random_other_category"])]
                Utr, Vtr = embed(enc, trr.resume_text, "resume", st), embed(enc, trr.job_text, "job", st)
                Ute, Vte = embed(enc, te.resume_text, "resume", st), embed(enc, te.job_text, "job", st)
                cos = (Ute * Vte).sum(1); hit, mrr = _mrr_hit(te, cos)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(_feat(Utr, Vtr), trr.label)
                sc = clf.predict_proba(_feat(Ute, Vte))[:, 1]
                rows.append(dict(model=enc.key, strategy=st, seed=seed, cos_auc=roc_auc_score(te.label, cos),
                                 cos_hit1=hit, cos_mrr=mrr, head_acc=accuracy_score(te.label, sc > .5), head_auc=roc_auc_score(te.label, sc)))
            print(f"done {enc.key} / {st}", flush=True)
    P = pd.DataFrame(rows); P.to_csv(f"{out}/truncation_per_seed.csv", index=False)
    S = P.groupby(["model", "strategy"]).agg(cos_auc=("cos_auc", "mean"), cos_auc_sd=("cos_auc", "std"), cos_hit1=("cos_hit1", "mean"),
                                             cos_mrr=("cos_mrr", "mean"), head_acc=("head_acc", "mean"), head_acc_sd=("head_acc", "std")).reset_index()
    S.to_csv(f"{out}/truncation_summary.csv", index=False)
    open(f"{out}/NOTE_truncation.txt", "w").write("Strategies fixed in advance, frozen encoders, no tuning. section_aware falls back to head when fewer than 2 headings are found (resume text is lower-cased). Differences of ~1-2 pp are within seed noise; report mean +- SD.\n")
    pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
