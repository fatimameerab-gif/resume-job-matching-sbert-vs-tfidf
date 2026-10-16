import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from representations import build_pair_features
from robustness import (CORE_PAIRS, ALL_PAIRS, DIRECTIONS, build_mapping, compile_mapping,
                        perturb_counted)

OUT = "results/v7"
RATES = (0.25, 0.5, 0.75, 1.0)
SIDES = ("resume_only", "job_only", "both")
N_REPEATS = 30            # review: >= 30 repeats per cell
SEEDS = range(5)


def _vec_fit(train):
    v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    v.fit(pd.concat([train.resume_text, train.job_text], ignore_index=True))
    return v


def _feats(v, res, job, mode):
    return build_pair_features(v.transform(res), v.transform(job), mode=mode)[0]


def rank_change(te, base, new):
    """Score-level, ranking view (review 4.6): per test resume the 3 candidate jobs are ranked by score.
    Returns (mean Kendall tau between original and perturbed ranking, share of resumes whose top-1 job changes)."""
    taus, top = [], []
    for idx in _GROUPS(te):
        b, n = base[idx], new[idx]
        if len(idx) < 3:
            continue
        top.append(float(np.argmax(b) != np.argmax(n)))
        if np.ptp(b) == 0 or np.ptp(n) == 0:
            taus.append(1.0 if np.array_equal(np.argsort(b), np.argsort(n)) else 0.0)
        else:
            taus.append(stats.kendalltau(b, n)[0])
    return float(np.nanmean(taus)), float(np.mean(top))


_GCACHE = {}
def _GROUPS(te):
    key = id(te)
    if key not in _GCACHE:
        _GCACHE.clear(); _GCACHE[key] = [np.array(ix) for ix in te.groupby("resume_id").indices.values()]
    return _GCACHE[key]


def coverage_report(df, pairs, name):
    rows = []
    for direction in DIRECTIONS:
        mp = build_mapping(pairs, direction); rx = compile_mapping(mp)
        for side, col in (("resume", "resume_text"), ("job", "job_text")):
            texts = df[col].drop_duplicates()
            res = [perturb_counted(t, mp, rx) for t in texts]
            n_hit = sum(r[1] > 0 for r in res)
            rows.append(dict(lexicon=name, direction=direction, side=side, n_unique_texts=len(texts),
                             n_texts_with_match=n_hit, pct_texts_with_match=round(100 * n_hit / len(texts), 1),
                             total_matches=sum(r[1] for r in res),
                             mean_matches_per_text=round(np.mean([r[1] for r in res]), 2)))
    return pd.DataFrame(rows)


def entry_hits(df, pairs):
    from collections import Counter
    c = Counter()
    for direction in DIRECTIONS:
        mp = build_mapping(pairs, direction); rx = compile_mapping(mp)
        for col in ("resume_text", "job_text"):
            for t in df[col].drop_duplicates():
                for h in rx.findall(str(t)):
                    c[(direction, h.lower())] += 1
    return pd.DataFrame([dict(direction=d, term=t, n=n) for (d, t), n in c.most_common()])


def validation_sheet(df, path, per_entry=2, seed=0):
    """Contexts of real substitutions for human rating of meaning preservation."""
    rng = np.random.default_rng(seed); rows = []
    texts = pd.concat([df.resume_text, df.job_text]).drop_duplicates().tolist()
    for direction in DIRECTIONS:
        mp = build_mapping(ALL_PAIRS, direction); rx = compile_mapping(mp)
        buckets = {}
        for t in texts:
            words = str(t).split()
            for m in rx.finditer(str(t)):
                pre = str(t)[:m.start()].split()[-10:]; post = str(t)[m.end():].split()[:10]
                buckets.setdefault(m.group(0).lower(), []).append((" ".join(pre), m.group(0), " ".join(post)))
        for term, ctxs in buckets.items():
            for i in rng.permutation(len(ctxs))[:per_entry]:
                pre, hit, post = ctxs[i]
                rows.append(dict(direction=direction, original=hit, replacement=mp[hit.lower()],
                                 context_before=pre, context_after=post,
                                 annotator1_meaning_preserved_0_1="", annotator2_meaning_preserved_0_1="",
                                 notes=""))
    sheet = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "item_id", [f"R{i+1:03d}" for i in range(len(sheet))])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.to_csv(path, index=False)
    return len(sheet)


def run(n_repeats=N_REPEATS, out_dir=OUT, pairs=CORE_PAIRS, lexicon_name="CORE"):
    os.makedirs(out_dir, exist_ok=True)
    all_test = pd.concat([pd.read_csv(f"data/eval_v7/frozen_seeds/pairs_seed{s}.csv").query("split=='test'")
                          for s in SEEDS], ignore_index=True)
    cov = pd.concat([coverage_report(all_test, pairs, lexicon_name),
                     coverage_report(all_test, ALL_PAIRS, "CORE+RISKY")], ignore_index=True)
    cov.to_csv(f"{out_dir}/robustness_v2_coverage.csv", index=False)
    entry_hits(all_test, ALL_PAIRS).to_csv(f"{out_dir}/robustness_v2_entry_hits.csv", index=False)
    n_items = validation_sheet(all_test, "annotation_kit/robustness_lexicon_validation_sheet.csv")

    rows = []
    for seed in SEEDS:
        df = pd.read_csv(f"data/eval_v7/frozen_seeds/pairs_seed{seed}.csv")
        tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
        v = _vec_fit(tr)
        models = {}
        for name, mode in (("tfidf_full", "absdiff_cos"), ("tfidf_cosLR", "cos_only")):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
            clf.fit(_feats(v, tr.resume_text, tr.job_text, mode), tr.label)
            base_p = clf.predict_proba(_feats(v, te.resume_text, te.job_text, mode))[:, 1]
            models[name] = (clf, mode, base_p, (base_p >= 0.5).astype(int))
        for name, (_, _, bp, bpred) in models.items():
            rows.append(dict(seed=seed, model=name, direction="none", side="none", rate=0.0, repeat=0,
                             acc=accuracy_score(te.label, bpred), flip_rate=0.0, mean_abs_dscore=0.0, kendall_tau=1.0, top1_change=0.0,
                             pct_texts_changed=0.0, mean_replacements_per_changed_text=0.0))
        R0 = v.transform(te.resume_text).toarray(); J0 = v.transform(te.job_text).toarray()
        for direction in DIRECTIONS:
            mp = build_mapping(pairs, direction); rx = compile_mapping(mp)
            for side in SIDES:
                for rate in RATES:
                    for rep in range(n_repeats):
                        rng = np.random.default_rng([seed, rep, int(rate * 100),
                                                     DIRECTIONS.index(direction), SIDES.index(side)])
                        R, J = R0.copy(), J0.copy()
                        n_changed, n_repl = 0, 0
                        for do_side, col, M in (("resume_only", te.resume_text, R), ("job_only", te.job_text, J)):
                            if side not in (do_side, "both"):
                                continue
                            out = [perturb_counted(t, mp, rx, rate, rng) for t in col]
                            idx = [k for k, o in enumerate(out) if o[2] > 0]
                            if idx:
                                M[idx] = v.transform([out[k][0] for k in idx]).toarray()
                            n_changed += len(idx); n_repl += sum(o[2] for o in out)
                        for name, (clf, mode, bp, bpred) in models.items():
                            p = clf.predict_proba(build_pair_features(R, J, mode=mode)[0])[:, 1]
                            pred = (p >= 0.5).astype(int)
                            _kt, _t1 = rank_change(te, bp, p)
                            rows.append(dict(seed=seed, model=name, direction=direction, side=side, rate=rate,
                                             repeat=rep, acc=accuracy_score(te.label, pred),
                                             flip_rate=float((pred != bpred).mean()),
                                             mean_abs_dscore=float(np.abs(p - bp).mean()),
                                             kendall_tau=_kt, top1_change=_t1,
                                             pct_texts_changed=100 * n_changed / (len(te) * (2 if side == "both" else 1)),
                                             mean_replacements_per_changed_text=(n_repl / n_changed) if n_changed else 0.0))
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{out_dir}/robustness_v2_raw.csv", index=False)

    clean = raw[raw.direction == "none"].groupby(["seed", "model"]).acc.mean().rename("clean_acc").reset_index()
    perturbed = raw[raw.direction != "none"].merge(clean, on=["seed", "model"])
    perturbed["dacc"] = perturbed.acc - perturbed.clean_acc
    seed_lvl = perturbed.groupby(["model", "direction", "side", "rate", "seed"]).agg(
        dacc=("dacc", "mean"), flip_rate=("flip_rate", "mean"), mean_abs_dscore=("mean_abs_dscore", "mean"),
        kendall_tau=("kendall_tau", "mean"), top1_change=("top1_change", "mean"),
        pct_texts_changed=("pct_texts_changed", "mean")).reset_index()
    summ = []
    for keys, g in seed_lvl.groupby(["model", "direction", "side", "rate"]):
        n = len(g); t = stats.t.ppf(0.975, n - 1)
        r = dict(zip(["model", "direction", "side", "rate"], keys)); r["n_seeds"] = n
        for m in ("dacc", "flip_rate", "mean_abs_dscore", "kendall_tau", "top1_change", "pct_texts_changed"):
            mu, se = g[m].mean(), g[m].std(ddof=1) / np.sqrt(n)
            r[m] = mu; r[m + "_lo"] = mu - t * se; r[m + "_hi"] = mu + t * se
        summ.append(r)
    summ = pd.DataFrame(summ)
    # dose-response slope of flip_rate on rate (through the 0% clean point), per seed then averaged
    slopes = []
    for (model, direction, side), g in seed_lvl.groupby(["model", "direction", "side"]):
        per = []
        for seed, gs in g.groupby("seed"):
            x = np.r_[0.0, gs.sort_values("rate").rate.values]; y = np.r_[0.0, gs.sort_values("rate").flip_rate.values]
            per.append(np.polyfit(x, y, 1)[0])
        slopes.append(dict(model=model, direction=direction, side=side,
                           flip_rate_slope_per_100pct=np.mean(per), slope_sd_over_seeds=np.std(per, ddof=1)))
    summ = summ.merge(pd.DataFrame(slopes), on=["model", "direction", "side"], how="left")
    summ.round(4).to_csv(f"{out_dir}/robustness_v2_summary.csv", index=False)
    return summ, cov, n_items


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    summ, cov, n_items = run()
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(cov.to_string(index=False)); print(f"\nvalidation sheet items: {n_items}")
    print(summ[(summ.rate == 1.0)][["model", "direction", "side", "dacc", "dacc_lo", "dacc_hi", "flip_rate",
                                    "mean_abs_dscore", "pct_texts_changed"]].round(3).to_string(index=False))
