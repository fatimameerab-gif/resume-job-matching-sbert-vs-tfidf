# RQ1: Where does the benefit of sentence encoders come from? (resume-job matching, controlled comparison with TF-IDF)

This folder holds everything for RQ1: data, code, results and tests (the paper's RQ2 and RQ3 have their own folders, `RQ2_FINAL` and `RQ3_FINAL`; some robustness and pipeline scripts live here because they use the same data). The skill-gap exploration (`skill_gap_v2.py`) is also here. This file is the story of what we did and what came out. Technical details, file by file, are in `TECHNICAL_NOTES.md`; the GPU runs are in `GPU_RUNBOOK.md`; all current numbers are in `GPU_RESULTS.md` and `STATUS_PASS2.md`. If this file and the paper ever disagree, the paper tables (generated from `results/`) are right.

## 1. Where we started
The first version compared TF-IDF, LSA and SBERT (MiniLM) on resume-job pairs and found SBERT slightly ahead. A review raised problems we took seriously: the labels only said "resume category equals job category", the negatives were random and easy, the test set had 75 pairs from 25 resumes, the SBERT-TF-IDF gap was not significant, the manuscript promised a fine-tuned model that did not exist, one job file was not clearly licensed, and the software environment was mixed. A second review asked for resume-level statistics, a category-classifier baseline, leave-category-out, full-pool ranking, stronger robustness tests and one pinned environment. We fixed what code can fix, report the results as they came out, and did not tune anything to make SBERT win.

## 2. Data
- The unlicensed job file was replaced by SEEK Australia postings (563 postings, CC BY 4.0; only title, description and category are kept). The old 157-row file is archived outside every release.
- After removing duplicate resumes there are 166 unique resumes in 25 categories. Job texts come from SEEK (338 pairs), ESCO descriptions (128) and Kaggle postings (32).
- The frozen pair file has 498 pairs (348 train, 75 validation, 75 test). Besides it there are five frozen resume-grouped seed files (150 test pairs each, 135 unique test resumes in total), 20 repeated grouped splits, and Design B, which groups jobs as well as resumes.
- Resume and job texts are PII-scrubbed by pattern (e-mail, phone, URLs, ...); the marker file `data/PII_SCRUBBED` records this. Names are NOT removed. See `DATA_STATEMENT.md`.
- Labels are weak (category equality). Say so everywhere.

## 3. What we ran
1. **CPU baselines** (TF-IDF, LSA, BM25, word2vec trained on the pair corpus) on the first split with resume-clustered confidence intervals.
2. **Negative tiers** (random, same family, model-mined) with shortcut checks (job-only, job-length). Mined tiers partly measure job identity, so a family-balanced consensus tier is used as the shared hard tier.
3. **Frozen encoders** (MiniLM, mpnet, e5, bge, gte, JobBERT-v2) on five seeds, with chunk pooling; head ablation and truncation alternatives.
4. **Fine-tuning** of all six encoders on the training pairs of each seed (cosine loss, 3 epochs, 256-token training texts, mixed precision) and hybrids with TF-IDF.
5. **Category-classifier baseline**, **resume-level paired tests** (135 resumes, bootstrap, Holm), **leave-category-out** (including fine-tuned models), **full-pool ranking**, **Design B** with Nadeau-Bengio, Friedman/Nemenyi and Bayesian correlated t-test, **near-duplicate** analysis.
6. **Learning curve** (12, 29, 58, 116 training resumes; five seeds; all six encoders).
7. **Robustness** (run here for RQ2): ESCO synonyms, typos, reordering, abbreviation lexicon, CheckList with scaled tolerance, LLM paraphrase (Phi-3.5-mini-instruct) with a human check of 60 rewrites by three raters and an LLM judge.
8. **TalentCLEF Task A** (descriptive only), efficiency, name-swap probe, zero-shot cross-encoder.
9. **Paired pipeline test** (`rq3_pipeline_paired.py`) and **skill-gap** exploration (ESCO linking; gold-set kit prepared).

## 4. What we found (numbers are in `GPU_RESULTS.md` and the paper)
- **Frozen general-purpose encoders did not beat TF-IDF**: head accuracy 0.685-0.765 against 0.805 (five seeds, 150 test pairs each). JobBERT-v2 is higher (0.827; AUC 0.907 against 0.875) but not significantly so after correction. The honest statement is "no advantage", not "TF-IDF is better".
- **Fine-tuning helps a lot**: 0.885-0.931 accuracy; fine-tuned mpnet is +13.5 points over TF-IDF at the resume level (CI 10.0-17.0, Holm p<0.001).
- **The label is category equality**: two category classifiers that never see the pair reach 0.829 (TF-IDF) to 0.895 (JobBERT-v2 features). Fine-tuned JobBERT-v2 does not exceed its own baseline (2.6 points, CI -0.2 to 5.4).
- **Leave-category-out**: the fine-tuning advantage did NOT shrink when all test categories were unseen (AUC 0.91-0.95 against 0.70 for the TF-IDF head), so it is a transferable same-topic similarity rather than memorised categories. We still do not read it as better skill matching.
- **Design B** (resume- and job-grouped): TF-IDF stays ahead of every general-purpose frozen encoder by 10-19 points; fine-tuned JobBERT-v2 and mpnet are about 7.5 points above TF-IDF (Holm p=0.029).
- **Learning curve**: with 58 or more training resumes the fine-tuned head beats TF-IDF and its own frozen head for all six encoders; with 12 and 29 resumes this fails for e5 and gte.
- **Robustness**: frozen dense encoders were less stable than TF-IDF under typos (up to -9.1 points at 20% of words) and reordering (9-11% of decisions flipped); fine-tuning removed most of this. The abbreviation lexicon had no measurable effect for any model, it is author-curated; two annotators judged 123 of 131 sampled substitutions (94%) as meaning-preserving (`results/human_annotation/lexicon_*.csv`).
- **Paraphrase test**: humans disagreed on whether the rewrites keep the meaning (kappa 0.04), so it is reported as a meaning-altering rewrite stress test.
- **Pipeline**: a tuned gradient-boosting classifier on TF-IDF features was only 1.8 MRR points above the plain cosine threshold (not significant).
- **Skill-gap**: linking raises coverage (exact 2.4% against 21% at tau 0.5), but the validity AUC only moves from 0.678 to 0.710. No precision or recall claim without a gold set.

## 5. What we did not do, and what is open
- **Human annotation (done):** two annotators scored 240 pairs (0/1/2; quadratic-weighted kappa 0.91, binary kappa 0.87, exact agreement 91.7%). The weak label agreed with human fit for 90.0% of pairs (9 of 80 weak positives rated no fit, 15 of 160 weak negatives rated fit). Zero-shot cosine scores against human fit: JobBERT-v2 AUC 0.956, TF-IDF 0.868, gte 0.854, bge 0.830, MiniLM 0.816, mpnet 0.808, e5 0.751 (same ordering as against the weak label). Everything is in `results/human_annotation/` (anonymised, no texts, no free-text notes); regenerate with `python rq1_human_annotation_report.py` (needs the private sheets). The sample uses random negatives, so it audits label validity, not hard negatives; the fine-tuned models are scored on it by cross-fitting (`colab_human_label_finetune.py`, results in `results/human_label_finetune/`): AUC vs human fit 0.974 (JobBERT-v2), 0.961 (mpnet), 0.945 (MiniLM) against 0.879 for the TF-IDF head; frozen mpnet and MiniLM heads (0.828, 0.847) were not better than TF-IDF; one fold seed and one training seed.
- **Lexicon validation (done):** 131 substitutions, raw agreement 95.4%, Cohen's kappa 0.38 (CI -0.02 to 0.74; unstable because 95-97% of items were rated preserved), PABAK 0.91. 8 items were flagged by at least one annotator (4 of them from the main lexicon: js, aws, kpi); the robustness runs were not repeated without them.
- Still open: the skill-gap gold set (`annotation_kit_skillgap/`) needs two people. No conclusion about skill matching is drawn without it.
- The five seeds reuse the same resumes, so they are not independent replications; we report resume-level paired tests and clustered intervals.
- Everything is in-domain with weak labels and three data sources; we make no claim about real hiring decisions.
- Rater A of the paraphrase check is excluded from the main analysis because independence from the LLM judge could not be confirmed; the LLM judge is disclosed in the paper.
- Two Kaggle sets are CC0 as declared by their uploaders, but one is web scraped and the other has no documented origin; their texts are not redistributed (see `DATA_STATEMENT.md`, `NOTICE.md`).

## 6. How to repeat it
- `pip install -r requirements.txt`, then `python check_environment.py`, then `python -m pytest tests -q` (30 passed, 1 skipped in the last run). The CPU steps are listed in `TECHNICAL_NOTES.md`.
- GPU numbers come from one pinned Kaggle run (`requirements_gpu_pinned.txt`, `results/ENVIRONMENT.txt`); the saved outputs and embeddings are in `results/`. The GPU steps are in `GPU_RUNBOOK.md`.
- Human paraphrase ratings: `paraphrase_agreement.py` reads `data_private_ratings/*.xlsx`. These sheets contain resume text and are NOT part of any release; the public repository can contain only the anonymised item-level ratings (rater labels A, B, C and LLM judge; no texts, no notes) and only with the raters' written consent.
- Before sharing this folder publicly, run `python make_public_release.py ../RQ1_PUBLIC` (removes the Kaggle resume and job texts; `restore_texts.py` puts them back from the Kaggle files). Never include `data_private_ratings/`, the annotation KEY files, or `data/_archive_unlicensed`.

- **Decision layers on the full pool (review point 15, CPU):** `python rq3_fullpool_pipelines.py` (needs `rq3_grid_lib.py`, `results/ft_embeddings`) writes `results/fullpool_pipelines/{per_resume,summary,paired}.csv`. Fine-tuned JobBERT-v2 / mpnet beat TF-IDF with a cosine score or tuned gradient boosting (per-resume MRR 0.887 / 0.836 vs 0.756 / 0.713); a tuned LR or GB on fine-tuned features did not beat their cosine score.
- **Fine-tuned encoders on TalentCLEF (review point 16a):** Task A has no training set, so the encoders (fine-tuned on our weak labels) were tested zero-shot on its 10 development queries: no gain over frozen encoders or TF-IDF (MAP 0.844-0.874 vs 0.860). `colab_taskA_finetuned_transfer.py`, `results/taskA_finetuned/`.
