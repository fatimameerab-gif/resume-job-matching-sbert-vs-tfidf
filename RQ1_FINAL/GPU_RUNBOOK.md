# GPU runbook (Colab, Kaggle or any machine with a GPU)

Five scripts need a GPU or a model download. Their default SETTINGS (top of each file) are already the FINAL values (5 seeds, chunking on, 20 repeats), so you only change SETTINGS for a quick check. The CPU part of the project (TF-IDF/BM25/LSA/word2vec, tiers, robustness, power, annotation kit) is already run and saved in `results/`.

## What to upload to Colab
1. The whole project folder as ONE zip (this package). In Colab: upload it, then
```
!unzip -q RQ1_v9_FINAL.zip && cd RQ1_v9_FINAL && pip install -q rank-bm25 gensim sentence-transformers scikit-learn==1.4.2 scipy==1.13.0 statsmodels
```
   (keep Colab's own torch/transformers; do not reinstall them). Everything the scripts import is in the zip: `colab_common.py, representations.py, tier_eval_v7b.py, eval_protocol_v7.py, data_protocol_v2.py, data_loading.py, seek_adapter.py, robustness.py, robustness_v2.py, taskA_loader.py, rq1_colab_taskA_sbert.py, rq1_taskA_devset.py` and the data folders `data/eval_v7/frozen_seeds/` and `data/taskA_en/`.
2. Nothing else: the 563 selected SEEK postings are shipped as `data/seek_selected_postings.csv` (title, description, category only), so the raw 30,000-row Zenodo file is NOT needed.

## Run in this order (each: `!python <file>`; results are written to the folder shown)
| # | File | What it gives | Output folder | Needs |
|---|---|---|---|---|
| 1 | `colab_v7b_tiers_sbert.py` | dense encoders on the negative tiers, incl. dense-mined and family-balanced consensus tier | `results/colab_v7b_chunked/` | shipped data |
| 2 | `colab_v7b_robustness_sbert.py` | dense encoders under abbreviation perturbation (RQ2) | `results/colab_v7b_chunked/` | seed files |
| 3 | `colab_v7b_token_counts.py` | true word-piece lengths / truncation per encoder (CPU is enough) | `results/colab_v7b/` | pairs v3 file, TaskA |
| 4 | `colab_v7b_taskA_chunked.py` | TalentCLEF TaskA, truncate vs chunk, MiniLM / mpnet / JobBERT-v2 | `results/colab_v7b_taskA/` | TaskA |
| 5 | `colab_v7b_finetune_hybrid.py` | fine-tuned SBERT vs TF-IDF vs frozen vs hybrid (weighted + RRF), paired tests, per-pair predictions | `results/colab_v7b_finetune/` | seed files |

Quick check before the long run: set `SEEDS = 1` (and `REPEATS = 2` in script 2, `FINETUNE = False` in script 5), run, then restore. Script 5 is the slowest (it fine-tunes per seed and per encoder); if time is short set `ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2"`. The results of the first full run on a Colab T4 are in `GPU_RESULTS.md`. Script 5 uses batch 8, 256-token training texts and mixed precision (batch 16 ran out of memory on the 14.5 GB T4); after an out-of-memory error use Runtime -> Restart session before rerunning. Total runtimes were not recorded.
Optional extra encoders for scripts 1/4: `"all"` adds e5, bge, gte.

## What to send back
Zip the folders `results/colab_v7b*/` (CSV + NOTE_*.txt files) and upload them; the numbers in the paper must come from those CSVs, not from memory.

## Rules the scripts already follow
Thresholds and fusion weights are chosen on a validation split, never on test; nothing is tuned to favour SBERT; long texts are chunk-pooled (no silent truncation); seeds overlap in resumes, so use the paired per-seed tables, not seed-level t-intervals; the perturbation lexicon is unvalidated until two people rate `annotation_kit/robustness_lexicon_validation_sheet.csv`.

## Script 6 (extras) and widening RQ1 to more encoders
- `colab_v7b_extras.py`: head ablation (cos / |u-v|+cos / [u,v,|u-v|,u*v]), efficiency, cross-encoder, name-swap probe. Output `results/colab_v7b_extras/`. Set ENCODERS_TO_RUN (default includes e5, bge, gte, JobBERT, GloVe-avg).
- New encoders on tiers / fine-tune / hybrid: no new script. In script 1 (`colab_v7b_tiers_sbert.py`) and script 5 (`colab_v7b_finetune_hybrid.py`) set ENCODERS_TO_RUN = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2,GloVe-avg-300d" (for script 5 start with FINETUNE=False for the frozen/hybrid rows; fine-tune only the models you will discuss, T4 memory is tight).
- After script 5 reruns: `python analyze_finetune_results.py` then `python ranking_holm_errors.py`.

## ONE cell for all remaining GPU work
Upload this package folder to Colab (GPU T4), `pip install -q sentence-transformers`, then run `%run run_all_colab_extras.py`. It runs script 6 (extras), script 7 (truncation alternatives: head, head+tail, section-aware, chunk mean/max), new encoders on tiers / frozen+hybrid / robustness, and prints FAILED STEPS at the end. Zip `results/colab_v7b_extras`, `results/colab_v7b_truncation`, `results/colab_v7b_newenc` and send them back.

## Main script per job (no separate "remaining" scripts any more)
| Job | ONE script to run | Resumable? | Output |
|---|---|---|---|
| RQ1 extras, truncation, new encoders (tiers/hybrid/robustness) | `run_all_colab_extras.py` (replaces `run_remaining_colab.py`) | yes: finished steps are skipped | `results/colab_v7b_extras`, `colab_v7b_truncation`, `colab_v7b_newenc` |
| RQ1 learning curve (mpnet + JobBERT-v2, 3 seeds) | `run_learning_curve.py` | yes: per (seed, encoder, fraction) | `results/colab_v7b_learning_curve` |
| RQ2 dense (embeddings, MiniLM+mpnet 30 repeats, other encoders 10 repeats) | `run_rq2_colab.py` in RQ2 (replaces `run_rq2_remaining.py`) | yes: finished steps are skipped | `results/embeddings`, `rq2_dense`, `rq2_dense_more` |

## Fine-tuned models on the human labels (review 3, point 16b)
| Job | ONE script to run | Output |
|---|---|---|
| Fine-tune MiniLM, mpnet and JobBERT-v2 with cross-fitting by resume (no annotated resume or job text in training) and score the 240 human-annotated pairs | `colab_human_label_finetune.py` (needs `results/human_annotation/item_scores.csv`, `data/resume_job_pairs_v3.csv`, `colab_common.py`, `colab_v7b_finetune_hybrid.py`, `representations.py`) | `results/human_label_finetune/{predictions.csv, human_label_finetune.csv, NOTE.txt}` |
Quick check first: `python colab_human_label_finetune.py --encoders SBERT-MiniLM-L6-v2 --folds 5` (about 5 minutes on a T4). CPU smoke test without downloads: `--encoders stub`.

How the results were actually produced (honest log): Colab T4 and Kaggle T4 were used. The mpnet and JobBERT learning curves ran on Kaggle with `sentence-transformers==2.7.0` (mpnet) and, for JobBERT-v2, a manual training loop because `model.fit()` raised a `router_mapping` error in a newer sentence-transformers version; the loop (MSE on the cosine, AdamW, 10% warm-up, AMP, grad-clip 1.0) is now built into `colab_v7b_finetune_hybrid.py` as an automatic fallback (it was applied as a notebook patch in the original run). The first run of script 5 hit an out-of-memory error and was rerun with batch 8, 256-token training texts and AMP (see above). The code changes of this section were smoke-tested on CPU only (resume logic); the fallback loop itself has not been re-run on a GPU since it was moved into the file.

## Script 9: fine-tuned encoders on TalentCLEF 2026 Task A (transfer test, review point 16a)

`colab_taskA_finetuned_transfer.py` (GPU, about 20-25 min on a T4). TalentCLEF Task A has no training set (overview paper, Table 1: development 10 queries, test 40 queries), so a model cannot be fine-tuned on TalentCLEF data. The script fine-tunes MiniLM, mpnet and JobBERT-v2 on the weak-label training pairs of frozen seeds 0, 1 and 2 (script 5 recipe) and ranks the 472 resumes of each of the 10 development queries by cosine (chunk pooling); frozen encoders and TF-IDF are the references. No TalentCLEF text or label is used in training. The unit is the query (n = 10), so intervals are descriptive.

Needs, in the working folder: `colab_common.py`, `colab_v7b_finetune_hybrid.py`, `representations.py`, `rq1_colab_taskA_sbert.py`, `taskA_loader.py`, `data/taskA_en/`, `data/eval_v7/frozen_seeds/pairs_seed{0,1,2}.csv`.
Quick check: `python colab_taskA_finetuned_transfer.py --encoders SBERT-MiniLM-L6-v2 --seeds 0 --out results/taskA_quick`. Full run: `python colab_taskA_finetuned_transfer.py`.
Output: `results/taskA_finetuned/{per_query.csv, summary.csv, paired.csv, NOTE.txt}`. CPU check: `--encoders stub`.
