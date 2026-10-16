# Review status (current, after the pinned rerun, "pass 2")

Legend: DONE = code run and result saved in `results/` (pinned environment); HUMAN = needs a person; TEXT = manuscript wording; YOURS = for the first author. Numbers are in `GPU_RESULTS.md`, `STATUS_PASS2.md` and the paper tables (generated from `results/`). The first review (13 issues) is in part A, the second review (16 items) in part B.

## A. First review (13 headline issues)

| # | Issue | Status | What exists / what is left |
|---|---|---|---|
| 1 | Weak labels + easy negatives | DONE + HUMAN | Three negative tiers (random, same-family, model-mined) plus a family-balanced consensus tier (`results/v7/negative_tier_v7b_*.csv`, `results/colab_v7b_tiers_finetuned/`); job-prior and job-length shortcut checks. The category-classifier baseline (`results/category_baseline/`) shows how much of the pair task is category recognition. The 240-pair annotation sheets (`annotation_kit/`) are ready; the annotations and `compute_agreement.py` are HUMAN work. Labels stay weak: the paper says so. |
| 2 | Tiny test set (75 pairs, 25 resumes) | DONE as mitigation | Five frozen seeds (150 test pairs each, 135 unique test resumes), resume-level paired bootstrap with Holm (`results/resume_level/`), 20 repeated grouped splits and Design B (`results/design_b_dense/`). The main first split is unchanged (75 pairs); power: `results/v7/synthetic_mcnemar_power.csv`. |
| 3 | SBERT vs TF-IDF not significant | DONE | Frozen general-purpose encoders: 0.685-0.765 head accuracy against 0.805 for TF-IDF (no advantage); JobBERT-v2 0.827, not significant after Holm (+3.0 points, CI -0.8 to +6.8, p=0.50). The paper must not claim that off-the-shelf SBERT beats TF-IDF. |
| 4 | Manuscript claims a fine-tuned SBERT, code used frozen encoders | DONE | All six encoders are fine-tuned on each of the five seeds (`results/colab_v7b_finetune/`, `colab_v7b_newenc/`): 0.885-0.931. Fine-tuned mpnet: +13.5 points over TF-IDF at the resume level (CI 10.0-17.0, Holm p<0.001). Training details (256 tokens, 3 epochs, batch 8, lr 2e-5) are in the methods and in `NOTE_finetune.txt`. The gain is adaptation to the weak label; the category baseline and leave-category-out results are reported next to it. |
| 5 | Sparse job pool, job leakage | DONE | 563 licensed SEEK postings; only 2 categories have <= 4 job texts; 12.0% of test positives reuse a training-positive job (was 68%). Near-duplicate analysis (`results/duplicates_categories/`): 44% of test job slots have a near-duplicate training job, but accuracy on flagged and unflagged pairs differs little. Design B groups jobs as well as resumes (20 splits). |
| 6 | RQ2 perturbations invalid / symmetric / sparse / single draw | DONE + HUMAN | One-sided and two-sided perturbations, rates 25-100%, five seeds. Abbreviation lexicon: no measurable effect for any model. ESCO synonyms, typos, reordering, CheckList with scaled tolerance, LLM paraphrase (`results/robust_stronger/`, `results/paraphrase_robust/`). Human check of 60 paraphrases by three raters and an LLM judge (`results/paraphrases/human_check/`): low agreement (kappa 0.04), so the paraphrase test is reported as a "meaning-altering rewrite stress test". The abbreviation lexicon is author-curated; 131 sampled substitutions were rated by two annotators (94% preserved; `results/human_annotation/lexicon_*.csv`). |
| 7 | Cross-source experiment | DONE (RQ2 folder) | Now in `RQ2_FINAL/results/rq2/cross_source_*.csv` (SEEK against ESCO, 19 shared categories, 20 splits). Dense models lost up to 0.045 AUC when moved across sources; only JobBERT-v2 was consistently high. |
| 8 | RQ3 classifier grid without tuning/validation | DONE (RQ3 folder) | Now in `RQ3_FINAL/results/rq3/` and `rq3_dense/`. The paired pipeline test is `rq3_pipeline_paired.py` in this folder (`results/pipeline_paired/`). |
| 9 | Error analysis on regenerated pairs | DONE | `error_analysis.py` and `results/v7/error_analysis_*.csv`; disagreement tables for SBERT against TF-IDF in `results/colab_v7b_finetune/disagreement_*.csv`. A qualitative taxonomy by hand is yours. |
| 10 | Skill-gap demo = exact string matching, unevaluated | PARTLY | ESCO skill linking replaces exact matching (coverage 2.4% to 21% at tau 0.5; validity AUC 0.678 to 0.710, `results/skill_gap/`). No precision or recall claim without the gold set: `annotation_kit_skillgap/` is ready and needs two people. The paper presents it as a side analysis only. |
| 11 | Same datasets as Ajjam & Al-Raweshidy (2025) | TEXT | Novelty is argued in the paper (licensed SEEK/TalentCLEF data, tiers, shortcut controls, perturbation design, pinned environment). |
| 12 | Reproducibility | DONE | All GPU numbers from one pinned Kaggle environment (`requirements_gpu_pinned.txt`, `results/ENVIRONMENT.txt`); fp32 cast; CPU analyses rerun on the PII-scrubbed texts (`data/PII_SCRUBBED`); frozen seed files; 30 tests passed, 1 skipped. |
| 13 | Literature review gaps | TEXT / YOURS | Related work extended; verify references [18], [24], [33] and the 2025-26 DOIs yourself. |

## B. Second review (16 items)

| Item | Status | Where |
|---|---|---|
| Category-classifier baseline | DONE | `rq1_category_baseline.py`, `results/category_baseline/` |
| Leave-category-out incl. fine-tuned | DONE | `rq1_leave_category_out.py`, `results/leave_category_out/` (fine-tuning advantage does NOT shrink on unseen categories) |
| Fine-tuning on all tiers | DONE | `gpu_tiers_finetuned.py`, `results/colab_v7b_tiers_finetuned/` |
| Candidate-set size next to every ranking metric; full-pool ranking | DONE | `rq1_fullpool_ranking.py`, `results/fullpool_ranking/` |
| Learning curve: zero-shot lines, five seeds, all encoders | DONE | `rq1_learning_curve_cpu.py`, `colab_v7b_learning_curve.py`; the text states the exceptions (e5, gte at 12 and 29 resumes) |
| Head ablation; one pre-specified head | DONE | `rq1_feature_ablation_table.py` |
| Resume-level stats, Design B incl. fine-tuned, near-duplicates | DONE | `rq1_resume_level_tests.py`, `rq1_design_b_dense.py`, `rq1_duplicates_and_categories.py` |
| ESCO / typos / reorder for frozen and fine-tuned encoders; CheckList scaled tolerance; LLM paraphrase | DONE | `robust_suite.py`, `gpu_robustness_all.py`, `gpu_paraphrase.py`; paraphrase meaning not guaranteed (human check: kappa 0.04) |
| Reproducibility rewrite, pinned environment, JobBERT 64 vs 512 | DONE | Methods text; only the 512-token JobBERT-v2 is claimed (the 64-token embeddings exist but are not a main model) |
| Paired pipeline test, gain definition | DONE | `rq3_pipeline_paired.py`, `results/pipeline_paired/` (claim softened) |
| Table 2 incl. fine-tuned e5/bge/gte, RRF, hybrid weight, TalentCLEF, efficiency incl. TF-IDF | DONE | `hybrid_rrf.py`, `rq1_talentclef_report.py`, `rq1_efficiency_tfidf.py` |
| Title/scope, PII | DONE | new title; `pii_scrub.py`; names are not removed (stated in `DATA_STATEMENT.md`) |
| Section 2 shortened, "proposed" removed, novelty | DONE | paper text |
| Minor items (settings, negatives, majority lines, macro accuracy, CIs, Data Science share) | DONE | paper text |
| 240-pair human annotation | DONE | two annotators; kappa (quadratic) 0.91; weak label agrees with human fit for 90%; models vs human labels (`results/human_annotation/`); paper Sections 3.8 and 4.4 |
| Lexicon validation | DONE | 131 items, raw agreement 95.4%, kappa 0.38 (prevalence), PABAK 0.91; 8 items flagged; robustness not rerun without them |
| Funding, code URL, AI-use statement | YOURS | placeholders in red in the Declarations |

## C. Corrections to older statements
- "LLM judge: NOT done (privacy)" is no longer true. An LLM judge rated the 60 paraphrase chunk pairs; this is disclosed in the paper and in the AI-use statement, and the judge is reported separately from the human raters.
- Rater A of the paraphrase check is excluded from the main analysis (independence from the LLM judge not confirmed); the raters' written consent is needed before any rating is released.
- "The fine-tuning gain is largely category recognition" was withdrawn: the category baseline matches much of the gain on the pair benchmark, but the gain does not shrink on unseen categories.
- The learning-curve claim "fine-tuned above frozen at every size" was wrong; the exceptions are e5 and gte at 12 and 29 resumes.
- Word2vec trained on the pair corpus: first-split accuracy 0.467 (an earlier text said 0.587).
- Open items for people: skill-gap gold set, OSF pre-registration (draft in `PREREGISTRATION_TEMPLATE.md`), code URL (after the repository is public), final proofreading. Done since the last version: 240-pair annotation, lexicon validation, funding, AI-use statement, references check, raters' consent.

## D. Third review (16 points) - status

| # | Point | Status | Where |
|---|---|---|---|
| 1 | Design B defined twice, "no job text shared" vs "does not remove overlap" | DONE | one definition (20 splits, resumes and job texts grouped; identical job texts never shared, near-duplicates can remain), Sections 3.1, 4.1, Limitations |
| 2 | MRR of fine-tuned mpnet 0.979 vs 0.973 | DONE | Table 11 now reports per-resume MRR (0.979) |
| 3 | MRR gain is not MRR minus 0.890 | DONE | cause: table MRR was pooled over pair-seeds, gain was per resume; table now per resume (reference 0.884), caption explains; `rq3_pipeline_comparison.py` adds `mrr_per_resume`, `hit1_per_resume` |
| 4 | MiniLM p 0.074 vs 0.073 | DONE | recomputed: 0.0735 (0.073496) -> 0.073 everywhere |
| 5 | Fig. 2 legend "3 seeds", "all encoders" | DONE | legend and axis fixed in `make_figures_v2.py` (5 seeds), text says the figure shows mpnet and JobBERT-v2 |
| 6 | Leave-category-out "did not shrink" too strong | DONE | abstract, Section 4.1, discussion, conclusion rewritten (fine-tuned AUC 0.95-0.97 -> 0.91-0.95; frozen JobBERT-v2 0.911, fine-tuning adds +0.036 there; accuracy gap inflated by the TF-IDF threshold) |
| 7 | Fig. 4 caption "mainly false positives" | DONE | FP 76/112/19 and FN 70/64/33 (TF-IDF / frozen mpnet / fine-tuned mpnet); described as operating point |
| 8 | Fig. 3 ROC pools overlapping test sets | DONE | caption says so |
| 9 | TF-IDF min_df, sublinear, "tuned" | DONE | settings stated (min_df=1, sublinear_tf=False, defaults, not tuned); "tuned TF-IDF baseline" replaced by "TF-IDF baseline" |
| 10 | Code availability broken, no URL | PARTLY | line is now a plain heading; URL to add after the repository is public |
| 11 | References 2025-2026 | DONE earlier | all checked against the pages (TalentCLEF, ConFit v2, JobBERT 2025, etc.) |
| 12 | Environment, tagged commit, main script, tests | PARTLY | abstract explains GPU + CPU environments; unit-test sentence made accurate (fine-tuning only smoke-tested). Still to do by the author: tag the commit (e.g. v1.0) after upload; `run_everything_gpu.py` and `run_v7_local.py` are the entry points |
| 13 | Robustness: ranking metric, CIs in Table 10, name-swap CIs | DONE | new MRR-change table, CheckList table with CIs, name-swap gaps with intervals (2 seeds) |
| 14 | Fine-tuned models on all hard tiers | DONE | new table (AUC for every tier, from `colab_v7b_tiers_finetuned`) |
| 15 | RQ3 full-pool paired test; tuned classifier on fine-tuned features | DONE (CPU) | `rq3_fullpool_pipelines.py` (+ `rq3_grid_lib.py`), results in `results/fullpool_pipelines/`. Full pool, 124 resumes: fine-tuned JobBERT-v2 (MRR 0.887) and mpnet (0.836) above TF-IDF cosine (0.756) and TF-IDF + tuned GB (0.713), Holm p=0.02; mpnet vs TF-IDF + LR not significant (Holm 0.12); MiniLM not separated. Tuned LR/GB on fine-tuned features did not beat their cosine. Only LR and GB; classifiers trained on pairs the encoder saw. Paper Table 16, Methods 3.4, abstract, discussion, limitations, conclusion |
| 16 | Fine-tuned models on human labels / TalentCLEF | DONE | (b) done: `colab_human_label_finetune.py` (cross-fitting by resume, no resume or job text of a test pair in training; Kaggle T4, 5 folds, one fold seed, one training seed). AUC vs human fit: JobBERT-v2 fine-tuned head 0.974, mpnet 0.961, MiniLM 0.945 vs TF-IDF head 0.879 (paired differences +0.096 [0.055, 0.142], +0.082 [0.041, 0.127], +0.066 [0.020, 0.112]); frozen heads: JobBERT-v2 0.961, MiniLM 0.847, mpnet 0.828 (the last two not better than TF-IDF). Results in `results/human_label_finetune/`; paper Section 3.5, Table 18, Limitations. (a) TalentCLEF Task A has no training set, so fine-tuning on it is impossible; done as a transfer test instead (`colab_taskA_finetuned_transfer.py`, `results/taskA_finetuned/`, paper Table 9): fine-tuned encoders trained on weak labels were NOT better on the 10 TalentCLEF dev queries than frozen encoders or TF-IDF (MAP 0.874 / 0.844 / 0.849 for JobBERT-v2 / mpnet / MiniLM against 0.860 for TF-IDF; all paired intervals include zero; MRR of fine-tuned mpnet and MiniLM below frozen, n.s.); the feasible substitute is a transfer test, script `colab_taskA_finetuned_transfer.py` (see GPU_RUNBOOK, script 9); random negatives only; same pinned package versions confirmed in the Kaggle session |
