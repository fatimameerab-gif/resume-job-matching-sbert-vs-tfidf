# RQ2: How robust are TF-IDF and SBERT to lexical variation and to a change of source?

This folder holds everything for RQ2: data, code, results and tests. It is self-contained. Technical details by file are in `TECHNICAL_NOTES.md`.

## 1. Where we started
The first robustness test had four problems. Its "synonyms" were often not synonyms (for example "cisco" became "Network Magic" and "less" became "LESS language"). The perturbation was applied to both texts at once, so shared words stayed shared. It was applied sparsely and drawn only once. The cross-source experiment used tiny test sets, and its two directions covered different category sets, so source shift and label shift were mixed together.

## 2. What we changed
- **Lexicon:** a small curated list of abbreviations and expansions (word boundaries respected, both directions) instead of dictionary synonyms. Two annotators rated 131 sampled substitutions: 123 (94%) were judged meaning-preserving by both, 8 were flagged by at least one (results in `results/rq2/lexicon_validation/`).
- **One-sided perturbation:** we perturb the resume only, the job only, or both, and report them separately.
- **Controlled rate and repetition:** 25, 50, 75 and 100 percent of the matching terms, 30 repeats per cell with its own random stream, five seeds, and we report how many texts contain a lexicon term at all.
- **More than accuracy:** mean absolute score change, flip rate, Kendall tau and top-1 change of the ranking per resume, plus CheckList-style invariance and directional tests (neutral filler, punctuation, copying job words in, adding foreign text).
- **Truncation:** trained and tested at the same length (matched), next to the old train-full/test-short design.
- **Cross-source:** SEEK postings against ESCO descriptions, restricted to the 19 categories both sources share, balanced, 20 grouped splits, with a paired comparison of cross-source against within-source models.
- **Dense encoders:** the same protocol was run on a GPU for MiniLM and mpnet (30 repeats) and for e5, bge, gte and JobBERT-v2 (10 repeats, to save time).

## 3. What we found
- **Lexical substitution barely moves TF-IDF.** At 100 percent, accuracy changes by between -0.7 and +0.4 points; 11 of 12 intervals include zero. 27-62 percent of texts contain a lexicon term, so the test does touch the data. Flip rate is at most 2.1 percent.
- **Dense encoders are not more robust.** Accuracy changes stay within about one point and all intervals include zero, so we found no evidence that SBERT handles this kind of variation better than TF-IDF.
- **CheckList (pass rates, cosine head):**

  | test | TF-IDF | MiniLM | mpnet | JobBERT | bge | e5 | gte |
  |---|---|---|---|---|---|---|---|
  | filler sentence (score unchanged) | 0.98 | 0.89 | 0.76 | 0.88 | 0.90 | 1.00 | 1.00 |
  | punctuation and case (unchanged) | 0.99 | 0.71 | 0.65 | 0.63 | 0.90 | 1.00 | 1.00 |
  | copy job words into resume (score up) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
  | add foreign text (score down) | 0.72 | 0.30 | 0.40 | 0.47 | 0.39 | 0.32 | 0.39 |

  Dense encoders react more to harmless edits and much less to irrelevant text added to a resume. For e5 and gte the 1.00 pass rate is partly a scale effect (their cosine scores are compressed, so a 0.02 tolerance is loose); there we compare flip rate and Kendall tau instead.
- **Truncation:** most of the drop for short texts is lost information, not only a train/test mismatch (cosine AUC 0.647 at 20 words, 0.868 on full text).
- **Cross-source:** dense models trained on one source lose about 0.01 to 0.05 AUC on the other. TF-IDF is mixed (SEEK +0.008, not significant; ESCO +0.062, but its within-source AUC is poor, 0.766). Only JobBERT-v2 is consistently higher (0.91-0.92 against 0.83-0.89). Because the labels are weak, we do not claim that SBERT generalises better.
- **Name swap:** a synthetic probe, not a fairness audit; results are in `results/colab_v7b_extras/name_swap_*.csv` and are interpreted as shift minus control.

## 4. Open points
- The lexicon validation is done (two annotators; raw agreement 95.4%, kappa 0.38 because 95-97% of items were rated preserved, PABAK 0.91). The robustness runs were not repeated without the 8 flagged items.
- We only tested abbreviation and expansion; paraphrase and typo robustness were not tested.
- The "add foreign text" test uses a synthetic dilution, not a real resume edit.
- Test sets are 150 pairs per seed, and some categories have few job texts in the cross-source setting.

## 5. How to repeat it
`pip install -r requirements.txt`, `python -m pytest tests -q`, then `python run_rq2_cpu.py` (20-30 minutes on CPU). The GPU parts (`run_rq2_colab.py`, `colab_v7b_robustness_sbert.py`) are already run; their outputs are in `results/rq2_dense`, `results/rq2_dense_more` and `results/embeddings`.
