# Review report (13 headline issues): status after the v3 data switch

Legend: DONE = code run on v3 and result saved in `results/`; GPU = script ready, needs the GPU run (see `GPU_RUNBOOK.md`); HUMAN = needs a person; TEXT = manuscript wording; NOT IN PACKAGE = the script the review refers to is not part of this folder, so nothing here addresses it.

| # | Issue | Status | What exists / what is left |
|---|---|---|---|
| 1 | Weak labels + easy negatives | DONE (partly) + HUMAN | Three negative tiers (random, same-family, hard-mined) in `results/v7/negative_tier_v7b_*.csv`; job-prior and job-length shortcut checks. Hard tier (TF-IDF-mined): zero-shot AUC 0.495 but trained accuracy 0.727 with job-prior AUC 0.691, so part of that accuracy is job identity, not matching: do not present it as "TF-IDF solves hard negatives". 240-pair annotation kit rebuilt on v3 (`annotation_kit/`); the 240 annotations (and `compute_agreement.py`) are HUMAN work. Labels stay weak: say so in the paper. |
| 2 | Tiny test set (75 pairs, 25 resumes) | DONE as mitigation | The main split is unchanged (75 test pairs / 25 resumes / 18 categories). Added: 5 frozen seeds (150 test pairs each) and 20 repeated grouped splits (pooled 1380 pair-evaluations; TF-IDF accuracy 0.842, resume-clustered 95% CI 0.817-0.866). Power: 625.5 pairs are needed to detect a 5 pp gap at 20% discordance; at n=75 the minimum detectable gap is 14.2 pp (`results/v7/synthetic_mcnemar_power.csv`). |
| 3 | SBERT vs TF-IDF not significant | DONE | Result (`GPU_RESULTS.md`): frozen SBERT is 4-8 pp BELOW TF-IDF on the 5 frozen seeds (per-seed McNemar mostly not significant at n=150); on TaskA SBERT is about equal to TF-IDF (only JobBERT-v2 trained MAP +0.041, one of 36 uncorrected comparisons, 10 queries). So the manuscript must not claim that off-the-shelf SBERT beats TF-IDF; metric choice and task decide the ranking. |
| 4 | Manuscript claims a fine-tuned SBERT, code uses frozen encoders | DONE + TEXT | Script 5 fine-tuned MiniLM and mpnet on each of the 5 seeds (cosine loss, 3 epochs, 256-token training texts, mixed precision, T4). mpnet: 0.928 vs TF-IDF 0.808 (+12 pp, all 5 seeds; also +11.3 pp on test pairs with unseen jobs); MiniLM +6 to +8 pp. The gain is adaptation to the weak category-equality labels; say that, and describe the training details (256 tokens, 3 epochs) in the methods. |
| 5 | Sparse job pool, job leakage | DONE | v3 pool: 563 licensed SEEK postings added. Only 2 categories have <= 4 job texts (v2: 13); 12.0% of test positives reuse a job that is a training positive (review, v2: 68%). A resume+job-grouped split (no job text in two splits) is evaluated over 20 seeds. |
| 6 | RQ2 perturbations invalid / symmetric / sparse / single draw | DONE + HUMAN | One-sided (resume-only, job-only, both), abbreviation<->expansion lexicon, rates 25-100%, 5 seeds, flip rate and mean |delta score| (`results/v7/robustness_v2_*.csv`). TF-IDF at 100%, both sides: accuracy change +0.3 pp [-1.6, +2.2] (full head) and -0.7 pp [-1.5, +0.2] (cosine head) for abbreviation->expansion: no measurable effect. Dense encoders (MiniLM, mpnet, 5 seeds x 20 repeats): accuracy change between -1.1 and +0.8 pp at 100%, all intervals include 0, so no evidence that dense encoders are more robust than TF-IDF to this perturbation. The lexicon is curated but NOT human-validated: have two people rate `annotation_kit/robustness_lexicon_validation_sheet.csv`. |
| 7 | Cross-source experiment | NOT IN PACKAGE | `cross_source_generalization.py` is not in this folder; nothing here addresses it. Either drop that experiment from the paper or rerun it on the licensed data. |
| 8 | RQ3 classifier grid without tuning/validation | NOT IN PACKAGE | `rq3_classifier_grid.py` is not in this folder. Script 5 does use a validation split for thresholds/weights, but it is not an RQ3 grid. |
| 9 | Error analysis on regenerated pairs | DONE | `error_analysis.py` reads the saved `data/resume_job_pairs_v3.csv`; outputs in `results/v7/error_analysis_*.csv` (TF-IDF/LSA/BM25; SBERT/fine-tuned/hybrid per-pair predictions are in `results/colab_v7b_finetune/pair_predictions.csv`, not yet fed into `error_analysis.py`). |
| 10 | Skill-gap demo = exact string matching, unevaluated | NOT ADDRESSED | `skill_gap_demo.py` is unchanged. Present it as a demonstration only, or evaluate it against a small human-labelled sample. |
| 11 | Same datasets as Ajjam & Al-Raweshidy (2025) | TEXT | Novelty must be argued in the manuscript (licensed SEEK/TalentCLEF data, tiers, shortcut controls, perturbation design). |
| 12 | Reproducibility | DONE | `requirements.txt` pinned; `check_environment.py` output in `results/v7/environment_check.txt`; frozen seed files; fixed seeds; 24 tests. CPU results were produced with the pinned versions WITHOUT torch/transformers (those are only needed for the GPU scripts). |
| 13 | Literature review gaps | TEXT | Add ConFit v2, TalentCLEF, PJB, JobBERT, bias studies; cite the sources in `NOTICE.md`. |

Also needed for the paper: the data statement (`DATA_STATEMENT.md`) has all dataset fields filled in (4 Oct 2026); the PII hits were inspected by hand and are documented there (not removed).


## Update (v10)
- 3.2 ranking framing: DONE (ranking_holm_errors.py, Hit@1/MRR on the 3-candidate pairs, GPU_RESULTS.md).
- 4.9 Holm correction: DONE for accuracy and ranking comparisons.
- 4.10 error analysis: SBERT vs TF-IDF disagreement tables DONE; qualitative taxonomy by hand is yours.
- 4.5 new encoders, head ablation, cross-encoder, efficiency, GloVe; 4.11 name swap: script 6 written and stub-tested, NOT yet run on GPU.

## Update (v13)
- Remaining Colab items all received and folded in (tiers, hybrid, robustness for new encoders; RQ2 dense; RQ3 embeddings; fine-tuned new encoders).
- RQ3 dense grid: all six encoders (design A).
- Corrected t-test + Bayesian correlated t-test: DONE for RQ3 design B (stats_extra.py). Bayesian signed-rank not applicable (single dataset family). OSF pre-registration: draft in PREREGISTRATION_TEMPLATE.md (to fill and upload by you). LLM judge: NOT done (sending resumes to an external API is a privacy issue).
- Learning curve: script ready (colab_v7b_learning_curve.py, run_learning_curve.py), needs GPU; results pending.
