import os, sys, time, types, glob, zipfile
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings("ignore")

# ======================= SETTINGS (change these, then run) =======================
ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2"   # or "SBERT-MiniLM-L6-v2" for a quick check
SEEDS = 5                                # 5 seeds (seeds 0-4); quick check = 1
REPEATS = 30                             # review asks for >= 30 repeats; quick check = 2
CHUNK_LONG_TEXTS = True                  # True = long texts are split in chunks and averaged (no truncation)
SEED_DIR = None                          # folder holding pairs_seed0.csv ... pairs_seed4.csv; None = find it (or unzip it) automatically
OUT_DIR = "results/colab_v7b_chunked"    # different folder, so the earlier truncated results are not overwritten
# ==================================================================================

from representations import build_pair_features
from robustness import CORE_PAIRS, DIRECTIONS, build_mapping, compile_mapping, perturb_counted
from robustness_v2 import SIDES, RATES, rank_change
import re
from sklearn.feature_extraction.text import TfidfVectorizer
INV_TOL = 0.02
FILL = ' references available upon request. available for interview at your convenience. thank you for your consideration.'
from tier_eval_v7b import _lr
from colab_common import load_encoders

HEADS = (("full", "absdiff_cos"), ("cosLR", "cos_only"))


def find_seed_dir(explicit=None):
    """Folder that holds pairs_seed0.csv. Looks in SEED_DIR, data/eval_v7/frozen_seeds, anywhere below '.'; if the files
    only exist inside a zip (ANY zip that contains pairs_seed0.csv, whatever it is called), the zip is extracted."""
    for r in (explicit, "data/eval_v7/frozen_seeds", "frozen_seeds", "."):
        if r and os.path.isdir(r):
            for root, dirs, files in os.walk(r):
                dirs[:] = [d for d in dirs if not d.startswith("_archive")]       # never pick up archived (old/unlicensed) seeds
                if "pairs_seed0.csv" in files:
                    return root
    for z in sorted(set(glob.glob("**/*.zip", recursive=True) + glob.glob("/content/*.zip"))):
        try:
            names = zipfile.ZipFile(z).namelist()
        except Exception:
            continue
        if any(n.endswith("pairs_seed0.csv") for n in names):
            print(f"seed files are inside a zip ({z}) -> extracting to data/eval_v7_unzipped ...", flush=True)
            zipfile.ZipFile(z).extractall("data/eval_v7_unzipped")
            for root, dirs, files in os.walk("data/eval_v7_unzipped"):
                if "pairs_seed0.csv" in files:
                    return root
    return None


def _no_seed_files(explicit):
    zs = sorted(glob.glob("**/*.zip", recursive=True) + glob.glob("/content/*.zip"))
    msg = ("pairs_seed0.csv ... pairs_seed4.csv NOT FOUND.\n"
           f"  working folder: {os.getcwd()}\n  contents: {sorted(os.listdir('.'))[:40]}\n  zip files seen: {zs or 'NONE'}\n")
    for z in zs[:3]:
        try:
            msg += f"  first entries of {z}: {zipfile.ZipFile(z).namelist()[:12]}\n"
        except Exception as e:
            msg += f"  could not open {z}: {e}\n"
    msg += "  Fix: unzip the frozen seed files, then set SEED_DIR = '<folder with pairs_seed0.csv>' in SETTINGS."
    if explicit:
        msg += f"\n  (SEED_DIR was {explicit!r}, exists: {os.path.isdir(explicit)})"
    raise FileNotFoundError(msg)


def summarise(raw):
    clean = raw[raw.direction == "none"].groupby(["encoder", "seed", "head"]).acc.mean().rename("clean_acc").reset_index()
    p = raw[raw.direction != "none"].merge(clean, on=["encoder", "seed", "head"])
    p["dacc"] = p.acc - p.clean_acc
    lvl = p.groupby(["encoder", "head", "direction", "side", "rate", "seed"]).agg(
        dacc=("dacc", "mean"), flip_rate=("flip_rate", "mean"), mean_abs_dscore=("mean_abs_dscore", "mean"),
        kendall_tau=("kendall_tau", "mean"), top1_change=("top1_change", "mean"),
        pct_texts_changed=("pct_texts_changed", "mean")).reset_index()
    out = []
    for keys, g in lvl.groupby(["encoder", "head", "direction", "side", "rate"]):
        n = len(g); r = dict(zip(["encoder", "head", "direction", "side", "rate"], keys)); r["n_seeds"] = n
        for m in ("dacc", "flip_rate", "mean_abs_dscore", "kendall_tau", "top1_change", "pct_texts_changed"):
            mu = g[m].mean(); r[m] = mu
            if n > 1:
                se = g[m].std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
                r[m + "_lo"], r[m + "_hi"] = mu - t * se, mu + t * se
        out.append(r)
    return pd.DataFrame(out)


def main(encoders=ENCODERS_TO_RUN, seeds=SEEDS, repeats=REPEATS, out=OUT_DIR, chunk_long=CHUNK_LONG_TEXTS, seed_dir=SEED_DIR):
    a = types.SimpleNamespace(encoders=encoders, seeds=seeds, repeats=repeats, out=out, chunk_long=chunk_long, seed_dir=seed_dir)
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/NOTE_lexicon_unvalidated.txt", "w").write(
        "The perturbation lexicon (robustness.py, CORE_PAIRS) is curated but NOT human-validated.\n"
        "Report these results as 'curated, unvalidated lexicon' until annotation_kit/"
        "robustness_lexicon_validation_sheet.csv has been rated by two people.\n")

    seeds_dir = find_seed_dir(a.seed_dir) or _no_seed_files(a.seed_dir)
    print("frozen seed files:", seeds_dir, flush=True)
    encs, rows, t0 = load_encoders(a.encoders, chunk_long=a.chunk_long), [], time.time()
    cl_rows = []
    for enc in encs:
        for seed in range(a.seeds):
            df = pd.read_csv(os.path.join(seeds_dir, f"pairs_seed{seed}.csv"))
            tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
            Utr, Vtr = enc.encode(tr.resume_text, "resume"), enc.encode(tr.job_text, "job")
            R0, J0 = enc.encode(te.resume_text, "resume"), enc.encode(te.job_text, "job")
            heads = {}
            for hname, mode in HEADS:
                clf = _lr(build_pair_features(Utr, Vtr, mode)[0], tr.label)
                bp = clf.predict_proba(build_pair_features(R0, J0, mode)[0])[:, 1]
                heads[hname] = (clf, mode, bp, (bp >= 0.5).astype(int))
                rows.append(dict(encoder=enc.key, seed=seed, head=hname, direction="none", side="none", rate=0.0,
                                 repeat=0, acc=accuracy_score(te.label, heads[hname][3]), flip_rate=0.0,
                                 mean_abs_dscore=0.0, kendall_tau=1.0, top1_change=0.0, pct_texts_changed=0.0))
            for direction in DIRECTIONS:
                mp = build_mapping(CORE_PAIRS, direction); rx = compile_mapping(mp)
                for side in SIDES:
                    for rate in RATES:
                        for rep in range(a.repeats):
                            rng = np.random.default_rng([seed, rep, int(rate * 100), DIRECTIONS.index(direction), SIDES.index(side)])
                            R, J = R0.copy(), J0.copy(); n_changed = 0
                            for do_side, col, M, kind in (("resume_only", te.resume_text, R, "resume"),
                                                          ("job_only", te.job_text, J, "job")):
                                if side not in (do_side, "both"):
                                    continue
                                res_out = [perturb_counted(t, mp, rx, rate, rng) for t in col]
                                idx = [k for k, o in enumerate(res_out) if o[2] > 0]
                                if idx:
                                    M[idx] = enc.encode([res_out[k][0] for k in idx], kind)
                                n_changed += len(idx)
                            for hname, (clf, mode, bp, bpred) in heads.items():
                                p = clf.predict_proba(build_pair_features(R, J, mode)[0])[:, 1]; pred = (p >= 0.5).astype(int)
                                _kt, _t1 = rank_change(te, bp, p)
                                rows.append(dict(encoder=enc.key, seed=seed, head=hname, direction=direction, side=side,
                                                 rate=rate, repeat=rep, acc=accuracy_score(te.label, pred),
                                                 flip_rate=float((pred != bpred).mean()),
                                                 mean_abs_dscore=float(np.abs(p - bp).mean()), kendall_tau=_kt, top1_change=_t1,
                                                 pct_texts_changed=100 * n_changed / (len(te) * (2 if side == "both" else 1))))
            # ---- CheckList INV/DIR tests (same definitions as rq2_checklist_tests.py) ----
            rngc = np.random.default_rng(seed)
            vv = TfidfVectorizer(max_features=3000, stop_words="english").fit(pd.concat([tr.resume_text, tr.job_text])); vocab = np.array(vv.get_feature_names_out())
            def _top(job, k=10):
                w = vv.transform([job]).toarray()[0]; return " ".join(vocab[np.argsort(-w)[:k]])
            by_cat = te.groupby("resume_category").resume_text.apply(list).to_dict()
            def _foreign(cat):
                oc = [c for c in by_cat if c != cat]; c = oc[rngc.integers(len(oc))]; t = by_cat[c][rngc.integers(len(by_cat[c]))]; return " ".join(t.split()[:30])
            variants = {"INV_filler": (te.resume_text + FILL, None),
                        "INV_punct": (te.resume_text.map(lambda x: re.sub(r"\s+", "  ", re.sub(r"[^\w\s]", "", x.upper()))), None),
                        "DIR_up_copy_job": (te.resume_text + " " + te.job_text.map(_top), "up"),
                        "DIR_down_foreign": (te.resume_text + " " + te.resume_category.map(_foreign), "down")}
            for vn, (newres, direction) in variants.items():
                Rn = enc.encode(list(newres), "resume")
                for hname, (clf, mode, bp, bpred) in heads.items():
                    p = clf.predict_proba(build_pair_features(Rn, J0, mode)[0])[:, 1]; d = p - bp
                    sel = np.ones(len(te), bool) if vn != "DIR_down_foreign" else (te.label.values == 1)
                    ok = np.abs(d[sel]) < INV_TOL if direction is None else (d[sel] > 1e-9 if direction == "up" else d[sel] < -1e-9)
                    kt, t1 = rank_change(te, bp, p)
                    cl_rows.append(dict(encoder=enc.key, seed=seed, head=hname, test=vn, pass_rate=float(ok.mean()), mean_dscore=float(d[sel].mean()),
                                        flip_rate=float(((p >= .5) != (bp >= .5))[sel].mean()), kendall_tau=kt, top1_change=t1))
            print(f"{enc.key} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(f"{a.out}/robustness_sbert_raw.csv", index=False)
    summ = summarise(raw); summ.round(4).to_csv(f"{a.out}/robustness_sbert_summary.csv", index=False)
    CL = pd.DataFrame(cl_rows); CL.to_csv(f"{a.out}/checklist_sbert_per_seed.csv", index=False)
    CL.groupby(["encoder", "head", "test"]).mean(numeric_only=True).drop(columns="seed").round(4).reset_index().to_csv(f"{a.out}/checklist_sbert_summary.csv", index=False)
    for e in encs:
        print(e.report())
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== rate = 100 % ===")
    print(summ[summ.rate == 1.0][["encoder", "head", "direction", "side", "dacc", "flip_rate", "mean_abs_dscore",
                                   "pct_texts_changed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
