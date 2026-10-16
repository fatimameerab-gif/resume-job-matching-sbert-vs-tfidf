# GPU results (pinned environment, v3 data, chunk pooling ON, five frozen seeds)

All numbers on this page come from the CSVs in `results/colab_v7b*/`, `results/leave_category_out/`, `results/fullpool_ranking/`, `results/resume_level/` and `results/category_baseline/`. They were produced in ONE pinned Kaggle environment (T4; Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, sentence-transformers 5.7.0; fp32 cast; see `results/ENVIRONMENT.txt`) and replace the numbers of the first, mixed-environment version of this file. The paper tables are generated from the same CSVs. Labels are weak (resume category == job category). The five seeds share resumes, so they are NOT independent replications; use the resume-level tests (below) for inference.

## 1. Frozen vs fine-tuned encoders vs TF-IDF (150 test pairs per seed, three candidates per resume; mean over five seeds)
| Model | Accuracy | AUC | Paired diff vs TF-IDF head, pp (per-seed range) | Seeds with McNemar p<0.05 |
|---|---|---|---|---|
| TF-IDF head | 0.805 | 0.875 | reference | - |
| TF-IDF cosine (zero-shot) | 0.824 | 0.867 | +1.9 (-0.7 to +5.3) | 0 of 5 |
| MiniLM frozen head | 0.747 | 0.809 | -5.9 (-9.3 to -2.7) | 1 of 5 |
| mpnet frozen head | 0.765 | 0.834 | -4.0 (-8.0 to +0.7) | 0 of 5 |
| bge frozen head | 0.761 | 0.809 | -4.4 (-8.7 to +1.3) | 1 of 5 |
| gte frozen head | 0.759 | 0.826 | -4.7 (-9.3 to -1.3) | 1 of 5 |
| e5 frozen head | 0.685 | 0.736 | -12.0 (-15.3 to -9.3) | 3 of 5 |
| JobBERT-v2 frozen head | 0.827 | 0.907 | +2.1 (-4.0 to +6.7) | 0 of 5 |
| MiniLM fine-tuned head | 0.885 | 0.946 | +8.0 (+2.0 to +10.7) | 4 of 5 |
| mpnet fine-tuned head | 0.931 | 0.971 | +12.5 (+10.0 to +14.7) | 5 of 5 |
| bge fine-tuned head | 0.911 | 0.966 | +10.5 (+5.3 to +13.3) | 4 of 5 |
| gte fine-tuned head | 0.903 | 0.955 | +9.7 (+4.7 to +13.3) | 3 of 5 |
| e5 fine-tuned head | 0.899 | 0.960 | +9.3 (+5.3 to +11.3) | 4 of 5 |
| JobBERT-v2 fine-tuned head | 0.919 | 0.968 | +11.3 (+8.0 to +13.3) | 5 of 5 |
| Hybrid TF-IDF + mpnet fine-tuned (weighted) | 0.937 | 0.975 | +13.2 (+10.0 to +16.0) | 5 of 5 |
| Hybrid TF-IDF + JobBERT-v2 frozen (weighted) | 0.860 | 0.929 | +5.5 (+2.0 to +11.3) | 1 of 5 |
| Hybrid TF-IDF + JobBERT-v2 fine-tuned (weighted) | 0.920 | 0.970 | +11.5 (+8.0 to +14.7) | 5 of 5 |

Source: `results/colab_v7b_finetune/finetune_summary_clean.csv` and `paired_vs_tfidf_summary.csv`. RRF hybrids are AUC-only (the validation threshold does not transfer).
Reading:
- Frozen general-purpose encoders did NOT beat the TF-IDF head (0.685-0.765 against 0.805). Per-seed tests are mostly not significant at n=150, so the safe statement is "no advantage", not "TF-IDF is better".
- The domain encoder JobBERT-v2 has the higher AUC (0.907 against 0.875) and higher accuracy (0.827), but the accuracy difference is not significant at the resume level (+3.0 points, CI -0.8 to +6.8, Holm p=0.50).
- Fine-tuning raised every encoder to 0.885-0.931. At the resume level (135 unique test resumes) fine-tuned mpnet is +13.5 points above TF-IDF (CI 10.0-17.0, Holm p<0.001).
- Hybrids with TF-IDF add little for the fine-tuned encoders (mpnet 0.937 against 0.931).
- Fine-tuning texts were cut at 256 tokens, 3 epochs, batch 8, lr 2e-5, mixed precision; inference uses chunk pooling. Thresholds and fusion weights were chosen on a validation split of the training pairs, never on test.

## 2. Resume-level tests and the category-classifier baseline
Files: `results/resume_level/` and `results/category_baseline/` (scripts `rq1_resume_level_tests.py`, `rq1_category_baseline.py`).
- The weak label is category equality, so two category classifiers that never see the pair already reach 0.829 (TF-IDF features), 0.852 (frozen mpnet) and 0.895 (frozen JobBERT-v2). Fine-tuned mpnet exceeds its own baseline by 7.8 points (CI 5.1-10.5, Holm p<0.001); fine-tuned JobBERT-v2 does not exceed its own baseline (2.6 points, CI -0.2 to 5.4, Holm p=0.46).
- So part of the fine-tuning gain on the pair benchmark can be reproduced by a model that only recognises categories. This does not decide what the fine-tuned models learn; see section 3.

## 3. Leave-category-out (`results/leave_category_out/summary.csv`; five folds of five unseen categories, three partitions, 15 runs)
| Model | AUC | Accuracy |
|---|---|---|
| JobBERT-v2 fine-tuned head | 0.947 | 0.847 |
| mpnet fine-tuned head | 0.928 | 0.825 |
| MiniLM fine-tuned head | 0.913 | 0.810 |
| JobBERT-v2 frozen head | 0.911 | 0.805 |
| mpnet frozen head | 0.810 | 0.719 |
| TF-IDF cosine | 0.727 | 0.667 |
| TF-IDF head | 0.696 | 0.504 |
| Category classifier (TF-IDF), cannot predict unseen categories | 0.682 | 0.676 |
The fine-tuning advantage did NOT shrink when the test categories were unseen, so fine-tuning is not only memorising training categories. The label is still category equality, so this is not evidence of better skill matching. The earlier statement "the fine-tuning gain is largely category recognition" was withdrawn because of this result.

## 4. Full-pool ranking (`results/fullpool_ranking/`)
Each test resume is ranked against 856-868 job texts (relevant = same category). Random ranking gives MRR 0.13 and nDCG@10 0.06. nDCG@10: fine-tuned JobBERT-v2 0.820, mpnet 0.804, e5 0.784, bge 0.782; category baseline with JobBERT-v2 features 0.761; frozen JobBERT-v2 0.678, TF-IDF 0.532, frozen mpnet 0.487. About 40% of the pool is Data Science.

## 5. Ranking on the three-candidate pairs (`results/colab_v7b_finetune/ranking_summary.csv`; 250 resume x seed units)
Hit@1 / MRR: hybrid TF-IDF + fine-tuned mpnet 0.956 / 0.977; mpnet fine-tuned head 0.952 / 0.973; JobBERT-v2 fine-tuned head 0.932 / 0.963; MiniLM fine-tuned head 0.888 / 0.939; TF-IDF cosine (zero-shot) 0.858 / 0.919; frozen JobBERT-v2 head 0.848 / 0.920. Random Hit@1 is 0.333 and random MRR 0.611. Paired and Holm-corrected tests: `ranking_paired_holm.csv`, `paired_accuracy_holm.csv`. Disagreement tables: `disagreement_*.csv`.

## 6. Learning curve (`results/colab_v7b_learning_curve/`, five seeds, all encoders; accuracy of the head)
| Encoder | Model | 12 resumes | 29 | 58 | 116 |
|---|---|---|---|---|---|
| (any) | TF-IDF head | 0.665 | 0.745 | 0.760 | 0.805 |
| MiniLM | frozen head | 0.705 | 0.708 | 0.728 | 0.747 |
| MiniLM | fine-tuned head | 0.767 | 0.813 | 0.845 | 0.885 |
| mpnet | frozen head | 0.720 | 0.731 | 0.732 | 0.765 |
| mpnet | fine-tuned head | 0.817 | 0.873 | 0.881 | 0.931 |
| bge | frozen head | 0.708 | 0.748 | 0.731 | 0.761 |
| bge | fine-tuned head | 0.745 | 0.804 | 0.827 | 0.911 |
| gte | frozen head | 0.744 | 0.755 | 0.748 | 0.759 |
| gte | fine-tuned head | 0.692 | 0.717 | 0.784 | 0.903 |
| e5 | frozen head | 0.632 | 0.644 | 0.664 | 0.685 |
| e5 | fine-tuned head | 0.601 | 0.731 | 0.805 | 0.899 |
| JobBERT-v2 | frozen head | 0.816 | 0.827 | 0.825 | 0.827 |
| JobBERT-v2 | fine-tuned head | 0.853 | 0.897 | 0.901 | 0.917 |
- At 12 training resumes frozen mpnet (0.720) and frozen JobBERT-v2 (0.816) are above the TF-IDF head (0.665); by 116 resumes TF-IDF (0.805) has caught up with frozen mpnet but not with JobBERT-v2.
- With 58 or more training resumes the fine-tuned head is above both TF-IDF and its own frozen head for all six encoders. With 12 and 29 resumes this holds for MiniLM, mpnet, bge and JobBERT-v2, but NOT for e5 (0.601 / 0.731) or gte (0.692 / 0.717; its frozen head is 0.744 / 0.755). Do not write "fine-tuned above frozen at every size".
- Seeds share resumes; SDs are descriptive only.

## 7. Negative tiers, truncation, cross-encoder, TalentCLEF
- On the family-balanced consensus tier (hard negatives mined from frozen-model similarity) every frozen model is near chance (AUC 0.47-0.60; JobBERT-v2 0.604, TF-IDF 0.588); fine-tuned encoders are clearly above chance (JobBERT-v2 0.838, mpnet 0.821, MiniLM 0.647) but were not part of the mining, so this is not a fair difficulty measure. Files: `results/v7/negative_tier_v7b_*.csv`, `results/colab_v7b_tiers_finetuned/`.
- No truncation strategy reliably beats the head baseline; section-aware encoding is worst for most encoders (`results/colab_v7b_truncation/`, `colab_v7b_chunked/`).
- Zero-shot MS MARCO cross-encoder: AUC 0.62-0.69, below TF-IDF; reference only (`results/colab_v7b_extras/cross_encoder.csv`).
- TalentCLEF 2026 Task A, ten queries: TF-IDF MAP 0.860 [0.766, 0.947]; JobBERT-v2 0.876-0.880; mpnet 0.859-0.869; MiniLM 0.842-0.850; every difference to TF-IDF has an interval that includes 0 (`results/colab_v7b_taskA/`).
- Name swap: the neutral prefix itself shifts dense scores, so only the shift minus control is read; the largest gap between name groups is 0.011 (mpnet, bge); synthetic probe, not a fairness audit.

## 8. Robustness (RQ2 part, run from this folder)
Files: `results/robust_stronger/`, `results/paraphrase_robust/`, `results/paraphrases/`, `results/colab_v7b_newenc/robustness/`.
- Abbreviation/expansion (author-curated; 94% of 131 sampled substitutions rated meaning-preserving by two annotators): TF-IDF -0.7 to +0.4 points at 100%; MiniLM and mpnet -1.1 to +0.8; all intervals include 0.
- ESCO synonyms: at most 3.0 points for any model. Typos at 20% of words: TF-IDF about 0; frozen dense heads lose 2.3 (JobBERT-v2) to 9.1 points (MiniLM); fine-tuned encoders -0.5 to -1.2. Sentence reordering: frozen dense encoders flip 9-11% of decisions (JobBERT-v2 1.6%); fine-tuned 1.1-3.4%.
- LLM paraphrase (Phi-3.5-mini-instruct, accepted rewrites 60.8%): reported as a "meaning-altering rewrite stress test" because the human check (60 items) found low agreement on meaning preservation (Cohen's kappa B-C 0.04). Details: `results/paraphrases/human_check/`.

## 9. Paired pipeline test
`rq3_pipeline_paired.py` -> `results/pipeline_paired/pipeline_paired.csv`: complete pipelines compared on per-resume reciprocal rank over the three candidates (averaged over the five seeds for each of the 135 test resumes; bootstrap over resumes, sign-flip p, Holm). The gradient-boosting predictions come from `../RQ3_FINAL/results/rq3/predictions_A.csv`. A tuned gradient-boosting classifier on TF-IDF features was only 1.8 MRR points above the plain cosine threshold (not significant); among the fine-tuned encoders only mpnet clearly beat it.

## Added in the third review round

- **Fine-tuned models against human labels (review point 16b)** - `colab_human_label_finetune.py`, Kaggle T4, 5 folds by resume, one fold seed and one training seed, same pinned environment (Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, sentence-transformers 5.7.0, scikit-learn 1.6.1). AUC against human fit on the 240 annotated pairs: JobBERT-v2 fine-tuned head 0.974, mpnet 0.961, MiniLM 0.945, TF-IDF head 0.879; frozen heads JobBERT-v2 0.961, MiniLM 0.847, mpnet 0.828. Files: `results/human_label_finetune/`.
- **Full-pool decision layers (review point 15, CPU)** - `rq3_fullpool_pipelines.py`. Per-resume MRR, 124 resumes: JobBERT-v2 fine-tuned cosine 0.887, mpnet fine-tuned 0.836, MiniLM fine-tuned 0.775, TF-IDF cosine 0.756, TF-IDF + tuned LR 0.770, TF-IDF + tuned GB 0.713. Files: `results/fullpool_pipelines/`.
- **Fine-tuned encoders on TalentCLEF 2026 Task A dev (transfer test, review point 16a)** - `colab_taskA_finetuned_transfer.py`, Kaggle T4, 3 training seeds (0-2). MAP: JobBERT-v2 fine-tuned 0.874 (frozen 0.876), mpnet 0.844 (0.859), MiniLM 0.849 (0.850), TF-IDF 0.860; MRR fine-tuned 0.920 / 0.943 / 0.920 (frozen 0.920 / 1.000 / 1.000). All paired differences (10 queries) include zero. TalentCLEF Task A has no training set. Files: `results/taskA_finetuned/`.
