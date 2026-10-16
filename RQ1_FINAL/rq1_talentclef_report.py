"""Review 3.8: TalentCLEF 2026 Task A (English dev set, 10 queries, 472 corpus resumes, expert relevance judgements) for the MAIN text: zero-shot ranking quality with
95% bootstrap intervals over the 10 queries, and paired differences against TF-IDF (from taskA_paired_vs_tfidf.csv). Pair 'accuracy' is NOT reported (472 of 4,720 pairs are
positive, so the majority rate is 0.90). Output: results/colab_v7b_taskA/taskA_main_table.csv. Run: python rq1_talentclef_report.py"""
import numpy as np, pandas as pd
D = "results/colab_v7b_taskA"; q = pd.read_csv(f"{D}/taskA_per_query.csv"); pv = pd.read_csv(f"{D}/taskA_paired_vs_tfidf.csv"); rng = np.random.default_rng(0); rows = []
for (m, mode), g in q[q.kind == "zeroshot"].groupby(["model", "mode"]):
    r = dict(model=m, mode=mode, n_queries=g.query_id.nunique())
    for met in ("map", "ndcg@100", "recall@10", "r_precision"):
        v = g.groupby("query_id")[met].mean().values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        r[met] = round(float(v.mean()), 3); r[met + "_ci"] = f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
    p = pv[(pv.model == m) & (pv["mode"] == mode) & (pv.kind == "zeroshot") & (pv.metric == "map")]
    if len(p): r["map_diff_vs_tfidf"] = f"{p.mean_diff_vs_tfidf.iloc[0]:+.3f} [{p.ci_lo.iloc[0]:+.3f}, {p.ci_hi.iloc[0]:+.3f}]"; r["queries_better_than_tfidf"] = f"{int(p.n_queries_better.iloc[0])}/{int(p.n_queries.iloc[0])}"
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{D}/taskA_main_table.csv", index=False); pd.set_option("display.width", 250); print(T[["model", "mode", "map", "map_ci", "ndcg@100", "map_diff_vs_tfidf", "queries_better_than_tfidf"]].to_string(index=False))
