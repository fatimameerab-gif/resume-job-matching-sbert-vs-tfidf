# Resume-job matching: representation comparison (RQ1) and robustness (RQ2)

Current data version: **v3** (all job postings have a stated licence; see `NOTICE.md` and `DATA_STATEMENT.md`). Labels are WEAK: a pair is positive when the resume category equals the job category. Nothing here is a human relevance judgement until `annotation_kit/` has been annotated.

## Data
| File | What |
|---|---|
| `data/UpdatedResumeDataSet.csv` | resumes (Kaggle) -> 166 unique after de-duplication, 25 categories |
| `data/data_science_job_postings_672.csv` | Kaggle postings (only 32 pairs use them) |
| `data/seek_selected_postings.csv` | 563 SEEK Australia postings (CC BY 4.0), title/description/category only; built by `seek_adapter.py` |
| `data/esco_*.csv`, `occupation_skills_mapping.csv` | ESCO occupations/skills |
| `data/resume_job_pairs_v3.csv` | frozen pairs: 498 = 348 train / 75 val / 75 test, 166 resumes, 25 categories; job source SEEK 338 / ESCO 128 / Kaggle 32 |
| `data/eval_v7/` | frozen seed pairs (pairs_seed0-4), negative-tier and job-grouped pair files |
| `data/taskA_en/` | TalentCLEF 2026 Task A dev set (10 queries x 472 resumes, synthetic corpus, CC BY 4.0) |
The old unlicensed 157-row file is not part of this package. Some script/result file names still contain "v2" (e.g. `data_protocol_v2.py`, `synthetic_v2_local_*.csv`); they now run on the v3 data.

## Run (CPU, ~20 min in total; pinned versions in `requirements.txt`, torch is NOT needed for these)
```
pip install -r requirements.txt     # or only the CPU packages (pandas, numpy, scikit-learn, scipy, statsmodels, rank-bm25, gensim, matplotlib, scikit-posthocs)
python data_protocol_v2.py          # rebuilds data/resume_job_pairs_v3.csv only if it is missing (never overwrites)
python full_comparison.py           # TF-IDF / LSA / BM25 / word2vec on the frozen split, clustered CIs, tests
python rq1_feature_ablation_table.py
python eval_protocol_v7.py          # frozen seeds, tier pairs, job-grouped pairs
python run_v7_local.py              # tiers, repeated grouped splits, TaskA RRF, power, truncation, robustness (TF-IDF)
python rq1_taskA_devset.py          # TaskA baselines (leave-one-query-out)
python error_analysis.py ; python power_mcnemar.py
python -m pytest tests -q           # 30 passed, 1 skipped
```
GPU part (SBERT and other encoders, fine-tuning, hybrid): **`GPU_RUNBOOK.md`** (how to run) and **`GPU_RESULTS.md`** (what came out, Colab T4). Review-report status: **`FEEDBACK_STATUS.md`**.

## Verified CPU results on v3 (what is in `results/`)
| Result | Value |
|---|---|
| Frozen split, test 75 pairs (TF-IDF / LSA / BM25 / word2vec) | accuracy 0.840 [0.747, 0.920] / 0.760 / 0.627 / 0.587 (resume-clustered CIs in `results/full_comparison.csv`) |
| Diagnostics | majority 0.667; job-only 0.480; cosine-only 0.827 (threshold from val) |
| 20 repeated resume-grouped splits (about 69 test pairs each) | TF-IDF 0.842 (pooled cluster CI 0.817-0.866), LSA 0.667, BM25 0.590 |
| Negative tiers, TF-IDF (10 seeds) | random 0.837 acc / 0.893 AUC; same-family 0.819 / 0.873; TF-IDF-mined hard 0.727 / 0.778 but zero-shot AUC 0.495 and job-prior AUC 0.691 (part of the hard-tier accuracy is job identity) |
| Robustness, TF-IDF, abbreviation->expansion at 100%, both sides | accuracy change +0.3 pp [-1.6, +2.2] (full head), -0.7 pp [-1.5, +0.2] (cosine head); flip rate 2.1% / 1.5% (unvalidated lexicon) |
| TaskA (10 queries, accuracy of the classifier framing) | TF-IDF 0.945; BM25 0.876; word2vec 0.772; LSA 0.147 (degenerate: predicts almost everything positive; use the ranking metrics instead) |
| GPU headline (150 test pairs per seed, 5 seeds) | TF-IDF 0.808; frozen SBERT 0.73-0.77; fine-tuned MiniLM 0.87-0.89; fine-tuned mpnet 0.93 (+12 pp vs TF-IDF in all 5 seeds); see `GPU_RESULTS.md` |
| Power (McNemar) | 625.5 pairs needed for a 5 pp gap at 20% discordance; minimum detectable gap at n=75 is 14.2 pp |
Notes: (1) `results/v7/token_length_truncation.csv` uses a whitespace approximation; true word-piece counts come from GPU script 3. (2) TaskA numbers do not depend on the v3 data; the word2vec TaskA accuracy differs from an older run (0.820), so treat it as run-to-run variable. (3) TaskA TF-IDF numbers from script 4 use their own fold protocol (about 0.939) and must not be mixed with `rq1_taskA_devset.py` numbers (0.945) in one table. (4) SBERT / fine-tuned / hybrid / TaskA-dense / dense-robustness results are in `results/colab_v7b*/` and summarised, with caveats, in **`GPU_RESULTS.md`**; re-aggregate the fine-tuning results with `python analyze_finetune_results.py`.

## Still open (human)
The 240 pairs and the lexicon sheet are annotated (`rq1_human_annotation_report.py`, `results/human_annotation/`); dataset details in `NOTICE.md` / `DATA_STATEMENT.md` are filled in (PII hits were inspected, not removed); manuscript wording (weak labels, domain mismatch, SEEK attribution, ethics paragraph).
