# Pre-registration draft (OSF-style) — fill in the [BRACKETS], then upload to OSF BEFORE the final experiments
Honest note: parts of this study were already run (exploratory phase). Register the remaining/confirmatory analyses and state clearly which results were seen before registration.

1. Title: [paper title]
2. Research questions: RQ1 representation comparison (TF-IDF/BM25/LSA/GloVe/MiniLM/mpnet/e5/bge/gte/JobBERT, fine-tuned); RQ2 robustness (one-sided lexical variation, CheckList INV/DIR, cross-source); RQ3 decision layer (LR/SVM/RBF-SVM/RF/GBM/kNN/MLP/cosine threshold), calibration.
3. Data: SEEK Australia (CC BY 4.0), Kaggle resume set (CC0 as declared by the uploader; origin undocumented), Kaggle Glassdoor job postings (CC0 as declared by the uploader; web scraped), TalentCLEF 2026 Task A (CC BY 4.0), ESCO. The one source with no licence at all was removed.
4. Hypotheses (directional, stated before confirmatory runs): H1 frozen general-purpose SBERT does NOT outperform tuned TF-IDF on hard-negative tiers; H2 fine-tuning improves over TF-IDF on weak labels; H3 dense encoders are not more robust than TF-IDF to lexical variation; H4 the decision layer changes AUC by less than the representation choice for dense features.
5. Design: frozen seed files pairs_seed0-4 (resume-grouped), 20 repeated grouped splits; nested tuning; thresholds on validation only.
6. Primary metrics: accuracy/balanced accuracy, ROC-AUC, Hit@1/MRR; secondary: Brier/ECE, flip rate, Kendall tau.
7. Tests: exact McNemar (paired), resume-clustered bootstrap CI (10,000 for headline numbers), Nadeau-Bengio corrected t-test and Bayesian correlated t-test (ROPE +-1 pp) for repeated splits, Friedman+Nemenyi/Holm; Holm across the family of RQ comparisons; Cliff's delta.
8. Exclusions/stopping: no tuning on test; no encoder is dropped after seeing results; all runs reported, including negative ones.
9. Deviations log: [fill in]. Known deviations: RQ3 dense run on 2 encoders first and design A only; Design B for dense: [state if run].
10. Ethics: public/licensed data only; name-swap probe is synthetic, not a fairness audit; no personal data released.
