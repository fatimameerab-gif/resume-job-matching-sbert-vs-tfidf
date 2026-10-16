# Human annotation results (anonymised)

Made by `rq1_human_annotation_report.py` from private annotator sheets. These files contain **no resume or job text and no free-text notes**; raters are only "rater 1 / rater 2".

| File | Content |
|---|---|
| `item_scores.csv` | 240 annotated pairs: ids, categories, weak label, split, job source, the two scores (0/1/2), mean, human fit (mean >= 1) |
| `annotation_agreement.csv` | Cohen's kappa (unweighted, quadratic, binary fit, good fit) with bootstrap 95% CI, exact agreement, number of disagreements |
| `weak_label_audit.csv` | agreement of the weak label with human fit, kappa, counts of weak positives rated no-fit and weak negatives rated fit |
| `model_vs_human.csv` | zero-shot cosine (TF-IDF and six frozen encoders) ROC AUC against human fit and against the weak label, resume-clustered bootstrap CIs, paired difference from TF-IDF, Spearman with the mean human score |
| `lexicon_items.csv` | 131 lexicon substitutions (term, replacement, in main lexicon or extra entry, the two judgements) |
| `lexicon_agreement.csv` | raw agreement, kappa with CI, PABAK, share rated preserved, counts of flagged items |

Notes
- The 240 pairs reuse the benchmark's random negatives: they audit label validity, not hard negatives.
- 170 of the 240 pairs belong to the training part of the first frozen split, so only zero-shot scores are compared; fine-tuned models are not evaluated here.
- Lexicon kappa is low (0.38) because 95-97% of items were rated "preserved" (prevalence); read it with raw agreement (95.4%) and PABAK (0.91).
- The model comparison is exploratory (added after the annotations were available); no multiplicity correction.
