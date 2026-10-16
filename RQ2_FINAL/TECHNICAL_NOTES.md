# RQ2: robustness to lexical variation, rebuilt on the licensed v3 data

Status: all results are final (TF-IDF on CPU; dense encoders on Colab/Kaggle GPU; dense cross-source on CPU). Everything here uses `data/resume_job_pairs_v3.csv` and the 5 frozen resume-grouped seeds shared with RQ1/RQ3. The unlicensed 157-row file is not part of this package.

## What the review asked (section 4.6) and what this package does
| Review point | Status | Where |
|---|---|---|
| Perturbations were not synonyms (cisco -> "Network Magic", "less" -> LESS language) | Replaced by a small curated abbreviation/expansion lexicon (word-boundary, both directions). Human validity rating done by two annotators (131 items; `results/rq2/lexicon_validation/`) | `robustness.py` |
| Perturbation applied to both texts | One-sided (resume only / job only) and both-sides reported separately | `robustness_v2.py` |
| Sparse, uncontrolled perturbation | Rate-controlled 25/50/75/100 % of matched terms; coverage reported (share of texts containing a term, matches per text) | `results/v7/robustness_v2_coverage.csv` |
| Single draw, shared RNG | 30 repeats per cell, independent RNG per cell, 5 seeds, mean and 95% CI over seeds, dose-response slope | `results/v7/robustness_v2_summary.csv` |
| Truncation only at test time | Matched design (train and test truncated to the same length) next to the old shifted design | `rq2_truncation_matched.py` |
| Only decision-level metrics | Added mean abs score change, Kendall tau of the per-resume ranking, top-1 change rate, CheckList INV/DIR tests | `robustness_v2.py`, `rq2_checklist_tests.py` |
| Cross-source numbers uninterpretable | Rebuilt: same 19 shared categories in both sources, balanced, 20 grouped splits, AUC/Hit@1/MRR, paired cross-vs-within gap | `rq2_cross_source.py` |

## Results (TF-IDF, CPU; seeds 0-4)
**Lexical robustness (100 % of matched terms replaced).** Accuracy change is between -0.7 and +0.4 percentage points in all 12 cells (model x direction x side); 11 of 12 confidence intervals include 0 (the exception, TF-IDF full head, resume side, expansion -> abbreviation, is -0.5 pp [-0.9, -0.2]). Decision flip rate is at most 2.1 %, Kendall tau of the candidate ranking is at least 0.968, top-1 change at most 2.0 %. 27-62 % of texts contain at least one lexicon term, so the perturbation touches a substantial part of the data. Conclusion: with this lexicon, TF-IDF is not measurably hurt; this is a null result, not evidence that the lexicon is representative.
**Granularity (first L words of both texts).** Matched training does not remove the drop: cosine-only AUC 0.647 (20 words), 0.746 (50), 0.826 (100), 0.862 (200), 0.868 (full). The full head is worse than the cosine head on short texts when trained matched (AUC 0.541 at 20 words), because 3000 features cannot be learned from 20-word texts with 348 training pairs. So most of the drop is information loss, not only train/test shift.
**CheckList probes.** Invariance (neutral filler sentence; punctuation/case): 97-99 % of pairs change by less than 0.02 in probability. Directional: copying the job's 10 top TF-IDF words into the resume raises the score in 99.9 % of pairs; appending 30 words of a resume from another category lowers the true-match score in 72 % (cosine head) to 77 % (full head) of positive pairs, so TF-IDF is only partly sensitive to dilution by foreign content.
**Cross-source (SEEK Australia vs ESCO descriptions, 19 shared categories, 114 test pairs per source per split, 20 splits).** ROC-AUC, trained on SEEK: tested on SEEK 0.886 (full head) / 0.902 (cosine head), tested on ESCO 0.827 / 0.810. Trained on ESCO: tested on ESCO 0.765 / 0.800, tested on SEEK 0.894 / 0.886. Compared with a model trained on the test source itself (paired, same test resumes), the cross-source model changes AUC by: SEEK test, cosine head -0.016 [-0.021, -0.012]; SEEK test, full head +0.008 [-0.006, +0.023] (not significant); ESCO test, cosine head +0.010; ESCO test, full head +0.062 [+0.044, +0.078]. The gain on ESCO is an artefact of the small, repetitive ESCO training pool (2-22 texts per category), not a real cross-source benefit; ESCO is also the harder test source for every model. Recall of 1.0 no longer occurs (thresholds are chosen on training data). Interpretation: with TF-IDF, moving between job-posting and occupation-description text costs at most about 0.02 AUC relative to in-source training; the label-set shift of the old experiment is removed by design.

## How to run
CPU: `python run_rq2_cpu.py` (20-30 min). Colab: `%run run_rq2_colab.py`. Tests: `python -m pytest tests -q`.

## Still yours to do (cannot be done by code)
1. Done: two annotators rated the lexicon sheet (validity 94%, raw agreement 95.4%, kappa 0.38, PABAK 0.91). Optional: rerun without the 8 flagged items.
2. Update the research-question wording to resume-to-posting vs resume-to-ESCO (README of the old RQ2 package already says this).
3. Cite Ribeiro et al. (2020, CheckList) for the INV/DIR design.

## Limits to state in the paper
Abbreviation/expansion substitution is one kind of lexical variation; paraphrase and typo robustness are not tested. Test sets are 150 pairs per seed; cross-source categories have few job texts. DIR_down uses a synthetic dilution, not a real resume edit. The name-swap probe (a synthetic gender/name test, not a fairness audit) is in `results/colab_v7b_extras/name_swap_*.csv`.


## Dense encoders (Colab T4 + CPU cross-source) — final

**CheckList (cosine+LR head, 5 seeds; INV tolerance 0.02; lexicon: 94% of sampled substitutions rated meaning-preserving)**

| test | TF-IDF | MiniLM | mpnet | JobBERT | bge | e5 | gte |
|---|---|---|---|---|---|---|---|
| INV_filler pass | 0.98 | 0.89 | 0.76 | 0.88 | 0.90 | 1.00 | 1.00 |
| INV_punct pass | 0.99 | 0.71 | 0.65 | 0.63 | 0.90 | 1.00 | 1.00 |
| DIR_up_copy_job pass | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
| DIR_down_foreign pass | 0.72 | 0.30 | 0.40 | 0.47 | 0.39 | 0.32 | 0.39 |

- Dense encoders are not more robust than TF-IDF; on DIR_down_foreign they are clearly worse (adding foreign-domain text rarely lowers the score).
- e5/gte INV = 1.00 is partly a scale artefact: their cosine scores are compressed, so a 0.02 tolerance is loose. Compare flip_rate / Kendall tau instead.
- e5/bge/gte/JobBERT use 10 repeats (cost), MiniLM/mpnet 30.

**Cross-source (SEEK Australia vs ESCO, 19 shared categories, 20 grouped splits)**
- Frozen-score (cosLR) models do not change with train source (cross-minus-within = 0 by construction); only the fitted "full" heads are meaningful.
- Dense "full" heads lose AUC when trained on one source and tested on the other (about -0.01 to -0.05). TF-IDF full: SEEK +0.008 (CI includes 0), ESCO +0.062 (higher, but its within-source AUC 0.766 is poor).
- Best cross-source AUC: JobBERT 0.91-0.92; TF-IDF 0.83-0.89 depending on direction. Do not write "SBERT generalises better"; only JobBERT is consistently higher, and this is weak-label data.
