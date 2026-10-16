# RQ3: How much does the decision layer matter on top of a fixed representation?

This folder holds the RQ3 code, data, tests and results: the classifier grid for TF-IDF and LSA (designs A and B), the dense-encoder grid (design A), the statistics, and the pipeline comparison. All numbers come from the pinned environment (`../RQ1_FINAL/results/ENVIRONMENT.txt`) and the PII-scrubbed texts (`data/PII_SCRUBBED`). If this file and the paper disagree, the paper tables (generated from the result CSVs) are right.

## 1. Where we started
The first RQ3 compared three classifiers (logistic regression, Random Forest, linear SVM) on three representations with default settings, one split and no threshold choice. Random Forest looked bad (recall 0.26), but that came from the default 0.5 threshold. A validation split existed but was never used and nothing was calibrated.

## 2. What we changed
- **Tuning without peeking at the test set:** each classifier is tuned by an inner resume-grouped 3-fold cross-validation on the training pairs; the threshold is chosen on calibrated out-of-fold scores. Test pairs are never used for a choice.
- **Eight classifiers:** cosine threshold, logistic regression, linear SVM, RBF SVM, Random Forest, gradient boosting (scikit-learn's histogram version), kNN and a small MLP, with the same grids for every representation.
- **Two designs:** (A) the five frozen seed splits shared with RQ1 (150 test pairs each) and (B) 20 repeated resume-grouped 70/30 splits for rank tests.
- **Statistics:** Friedman test with Nemenyi critical differences, Holm-corrected Wilcoxon tests, and for design B a Nadeau-Bengio corrected t-test and a Bayesian correlated t-test (region of practical equivalence of +-1 point). Benavoli's Bayesian signed-rank test needs several datasets and was not run.
- **Calibration:** Platt scaling, Brier score, ECE, reliability plot.
- **Pipelines, not only heads:** tuned heads on TF-IDF and LSA were compared with the frozen and fine-tuned encoders (RQ1 runs) and a zero-shot cross-encoder on the same test pairs, with paired resume-level tests.
- **Representations:** TF-IDF and LSA for both designs; six dense encoders (JobBERT-v2, mpnet, MiniLM, gte, bge, e5) for design A, from frozen embeddings exported once on a GPU.

## 3. What we found (details and tables in the paper, Section RQ3)
- **The "Random Forest is bad" result was a threshold artefact:** with a tuned threshold its recall on TF-IDF rose from 0.26 to 0.70 (accuracy 0.733 at the default threshold, 0.720 tuned, balanced accuracy 0.716). It stayed worse in AUC (0.80 against 0.89 for logistic regression).
- **Classifier ranking (design B, 20 splits):** Friedman p < 1e-17 for TF-IDF and LSA; Nemenyi critical difference 2.35 ranks. On TF-IDF gradient boosting (mean rank 1.65), RBF-SVM (2.65), the cosine threshold (3.15), logistic regression (3.35) and linear SVM (4.20) were not all separated; Random Forest (6.0), kNN (7.0) and the MLP (8.0) were worse. Gradient boosting had the highest mean AUC (0.917) and beat the cosine threshold in a paired Holm-corrected Wilcoxon test (+0.023 AUC, p_Holm = 0.002), but not by the Nemenyi criterion or the Nadeau-Bengio corrected t-test (p_Holm = 1.0).
- **Cosine threshold against LR / linear SVM / RBF-SVM:** no difference (corrected t-test p >= 0.70), but the Bayesian test put only 42-46% of its mass in the +-1 point region, so equivalence is plausible, not established.
- **The MLP fails on sparse TF-IDF features** (AUC about 0.5 with 3,000 features and 348 training pairs) but reaches 0.83 on 100-dimensional LSA features: a small-data limit, not a general statement.
- **Dense features (design A):** with LR, SVM or gradient boosting TF-IDF reached AUC 0.89-0.90 and the general-purpose encoders were lower (mpnet 0.77-0.85, gte 0.79-0.84, bge 0.77-0.82, MiniLM 0.78-0.81, e5 0.74-0.76). They were ahead only where TF-IDF is weak: MLP, kNN and marginally Random Forest. JobBERT-v2 was above TF-IDF with linear and SVM heads (0.93 against 0.89) and equal with gradient boosting (0.90). The seeds share resumes, so these differences are descriptive.
- **Pipelines (paired per-resume MRR, 135 resumes, Holm over 12 comparisons):** tuned gradient boosting was +0.051 above the untuned TF-IDF head (p_Holm = 0.018) but only +0.018 above the TF-IDF cosine (CI -0.002 to 0.038, p_Holm = 0.23). Fine-tuned mpnet was +0.045 above tuned gradient boosting (CI 0.021-0.071, p_Holm = 0.003); fine-tuned JobBERT-v2 (+0.028, p_Holm = 0.23) and MiniLM (+0.010, p_Holm = 0.51) were not distinguishable from it. So changing the pipeline mattered more than changing the classifier only for the strongest fine-tuned encoder; "fine-tuning beats classifier tuning" in general is not supported.
- **Cross-encoder (zero-shot MS MARCO):** AUC 0.64, below TF-IDF; it was not trained on this task.
- **Calibration:** ECE 0.05-0.10 for the better classifiers; with 150 test pairs and five bins it is noisy.

## 4. Open points
- Design B was run only for TF-IDF and LSA; the dense encoders use design A (five seeds that share resumes).
- Gradient boosting is scikit-learn's version (LightGBM and XGBoost are not in the pinned environment); grids are small on purpose.
- The Couronne et al. (2018) motivation is about biomedical tabular data: cite text/embedding classification evidence or soften it.
- Labels are weak category-equality labels and everything is in-domain.

## 5. How to repeat it
`pip install -r requirements.txt`, `python -m pytest tests -q`, `python rq3_classifier_grid_v3.py cpu` (about 35 minutes on two cores), `python rq3_pipeline_comparison.py`, `python rq3_plots.py`, `python rq3_bayes_nb.py`. Dense encoders: `python rq3_dense_per_rep.py <encoder> 0` (embeddings are in `results/embeddings`). Figure 5: `python make_fig5.py` -> `figures/Figure_5.pdf`. The paired pipeline test is `../RQ1_FINAL/rq3_pipeline_paired.py` (`../RQ1_FINAL/results/pipeline_paired/pipeline_paired.csv`; it averages per resume over seeds, so its MRR values (for example fine-tuned mpnet 0.979) differ slightly from the per-seed means in `results/rq3/pipeline_comparison.csv` (0.973)). Before sharing publicly run `python make_public_release.py ../RQ3_PUBLIC`; `restore_texts.py` puts the Kaggle texts back. Never include `data/_archive_unlicensed`.

**Files not duplicated here (identical copies are in `../RQ1_FINAL`):** `results/embeddings/JobBERT-v2-64.npz` (= `../RQ1_FINAL/results/embeddings_pinned/JobBERT-v2-64.npz`; not used by any RQ3 table) and `data/eval_v7/negative_tiers/`, `negative_tiers_v7b/`, `pairs_resume_and_job_grouped.csv` (= `../RQ1_FINAL/data/eval_v7/...`; RQ3 does not read them).
