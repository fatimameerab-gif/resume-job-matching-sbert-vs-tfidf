# Status after the pinned rerun (pass 2)

All GPU numbers come from one pinned Kaggle environment (see `results/ENVIRONMENT.txt`); all CPU analyses were rerun on the PII-scrubbed texts (`data/PII_SCRUBBED` marker). Numbers in the older README/GPU_RESULTS.md are from the first, mixed-environment version and are superseded by the paper tables.

| Item | Status |
|---|---|
| Frozen / fine-tuned encoders, five seeds | DONE (pinned). Frozen general-purpose encoders 0.685-0.765 vs TF-IDF head 0.805; fine-tuned 0.885-0.931 |
| Category baseline, resume-level tests (15 comparisons) | DONE (`results/category_baseline`, `results/resume_level`) |
| Leave-category-out incl. fine-tuned | DONE: fine-tuned advantage does NOT shrink on unseen categories |
| Full-pool ranking incl. fine-tuned, Design B incl. fine-tuned | DONE |
| Learning curve, five seeds, all encoders | DONE |
| Robustness: ESCO / typos / reorder for frozen and fine-tuned encoders, CheckList (scaled tolerance) | DONE (`results/robust_stronger`) |
| LLM paraphrase test | DONE (`results/paraphrases`, `results/paraphrase_robust`); meaning preservation rated by people: low agreement, reported as "meaning-altering rewrite stress test" |
| Human check of paraphrases | DONE with 3 raters + LLM judge (`paraphrase_agreement.py`, `results/paraphrases/human_check`). Rater A excluded from the main analysis (independence from the LLM judge not confirmed). Raw rating sheets contain resume text and are NOT part of any public release |
| RQ2 truncation / cross-source, RQ3 grid + dense grid + pipelines | DONE on pinned embeddings |
| LLM judge | Done in this session; NOTE the earlier decision "no external API for resumes (privacy)" no longer holds for these 60 chunk pairs: disclosed in the paper |
| 240-pair human annotation, lexicon validation | DONE (two annotators each; `results/human_annotation/`; paper Sections 3.8 and 4.4) |
| Skill-gap gold set | STILL OPEN (needs people) |
| Not run: JobBERT-v2-64 variant as a main model (only embeddings exist) | Not claimed in the paper |
