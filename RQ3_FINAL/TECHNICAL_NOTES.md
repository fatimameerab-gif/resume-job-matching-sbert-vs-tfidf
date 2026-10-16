# RQ3: technical notes (pinned rerun)

Status: results are final (TF-IDF/LSA with designs A and B; six dense encoders with design A; pipeline comparison including fine-tuned encoders and the cross-encoder). Everything uses `data/resume_job_pairs_v3.csv` (PII-scrubbed) and the five frozen resume-grouped seeds shared with RQ1 and RQ2. The unlicensed 157-row job file is not part of this package.

## What the review asked and where it is
| Review point | What was done | Where |
|---|---|---|
| No hyper-parameter search; validation never used | Inner resume-grouped 3-fold CV on the training pairs only; test pairs never used for a choice | `rq3_classifier_grid_v3.py` |
| RF recall 0.26 = threshold artefact | Threshold chosen on calibrated inner out-of-fold scores; threshold-free metrics (AUC, Hit@1, MRR) first; default-threshold columns kept | `results/rq3/grid_summary_*.csv` |
| Only three classifiers | Cosine threshold, LR, linear SVM, RBF SVM, Random Forest, gradient boosting (HistGradientBoosting), kNN, MLP | same |
| No fine-tuned bi-encoder / cross-encoder | Fine-tuned encoders (RQ1 runs) and a zero-shot cross-encoder on the same frozen test sets; paired resume-level tests | `rq3_pipeline_comparison.py`; `../RQ1_FINAL/rq3_pipeline_paired.py` |
| No calibration | Platt scaling on out-of-fold scores, Brier, ECE, reliability plot | `results/rq3/fig_reliability_designA.png` |
| Demsar protocol | Friedman + Nemenyi + Holm-corrected Wilcoxon over 20 resume-grouped splits (design B); five frozen splits (design A) are descriptive; Nadeau-Bengio corrected t-test and Bayesian correlated t-test (ROPE +-1 point) | `friedman_*`, `pairwise_holm_*`, `bayes_nb_designB.csv`, `stats_extra.py`, `rq3_bayes_nb.py`, `results/rq3/fig_cd_designB_auc.png` |
| Dense encoders in the classifier grid | Six encoders, design A, frozen embeddings | `rq3_dense_per_rep.py` -> `results/rq3_dense/<encoder>/` |
| Motivation from Couronne et al. (2018, biomedical tabular) | Manuscript task for the author: cite text/embedding classification evidence or soften | paper |

Design A = the five frozen seeds (same 150 test pairs per seed as RQ1). Design B = 20 repeated resume-grouped 70/30 splits (job texts may repeat across train and test, so the blocks are not independent population samples: use B to rank classifiers, not for population estimates).

## Results
All numbers, with the paper's tables, are in `README.md` section 3 and in the CSVs: `results/rq3/grid_summary_A.csv` and `grid_summary_B.csv` (TF-IDF, LSA), `results/rq3_dense/*/grid_summary_A.csv` and `results/rq3_dense/grid_summary_A.csv` (dense), `results/rq3/friedman_*.csv`, `pairwise_holm_*.csv`, `bayes_nb_designB.csv`, `results/rq3/pipeline_comparison.csv` (per-seed means over all pipelines, including the RQ1 runs) and `results/rq3/pipeline_vs_tfidf_head.csv` (MRR gain against the untuned TF-IDF head with resume-clustered CI). Paired per-resume pipeline tests: `../RQ1_FINAL/results/pipeline_paired/pipeline_paired.csv` (paper Table of pipeline comparisons; per-resume averages, so MRR values differ slightly from the per-seed means).

## How to run
`python rq3_classifier_grid_v3.py cpu` (about 35 min on two cores) -> `results/rq3`; `python rq3_pipeline_comparison.py`; `python rq3_plots.py`; `python rq3_bayes_nb.py`. Dense: Colab/Kaggle `%run run_rq3_colab.py`, copy `results/embeddings` back, then `python rq3_classifier_grid_v3.py dense` -> `results/rq3_dense`. Tests: `python -m pytest tests -q`. Figure 5: `python make_fig5.py`.

## Limits to state
Gradient boosting is scikit-learn's HistGradientBoosting (LightGBM/XGBoost not in the pinned environment). Grids are small by design and identical for every representation. 150 test pairs per split; design B blocks are not independent; the dense designs reuse the same five resume-sharing seeds. No claim is made about large-data behaviour or real hiring decisions.
