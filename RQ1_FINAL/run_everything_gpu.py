import sys, time, argparse, traceback, importlib
ALL6 = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
NEW4 = "e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"


def _m(name): return importlib.import_module(name)


STAGES = [
    ("env", "versions, GPU, JobBERT routing", lambda: _m("gpu_env").report()),
    ("embeddings", "frozen embeddings of all texts (6 encoders + JobBERT-v2 with 64-token chunks)", lambda: _m("colab_export_embeddings").main(encoders=ALL6 + ",JobBERT-v2-64", out="results/embeddings_pinned")),
    ("finetune", "fine-tuned vs frozen vs TF-IDF vs hybrids, 6 encoders x 5 seeds (+ exports fine-tuned embeddings)", lambda: _m("gpu_finetune_export").main(out="results/colab_v7b_finetune_pinned")),
    ("leave_category_out", "leave-category-out fine-tuning (3 partitions x 5 folds)", lambda: _m("rq1_leave_category_out").main(n_partitions=3, finetune=True, encoders="SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2")),
    ("design_b", "Design B (20 job-text-grouped splits) with fine-tuned encoders", lambda: _m("rq1_design_b_dense").main(finetune=True, encoders="SBERT-mpnet-base-v2,SBERT-MiniLM-L6-v2,JobBERT-v2")),
    ("robust_stronger", "ESCO synonyms / typos / reordering + scaled CheckList: frozen AND fine-tuned encoders", lambda: _m("gpu_robustness_all").main()),
    ("learning_curve", "learning curve, 6 encoders x 5 seeds, fine-tuned rows", lambda: _m("colab_v7b_learning_curve").main(encoders=ALL6, seeds=5, out="results/colab_v7b_learning_curve_pinned")),
    ("tiers_ft", "FINE-TUNED encoders on every negative tier", lambda: _m("gpu_tiers_finetuned").main(out="results/colab_v7b_tiers_finetuned")),
    ("tiers_frozen", "frozen encoders on the negative tiers", lambda: _m("colab_v7b_tiers_sbert").main(encoders="SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,bge-base-en-v1.5", out="results/colab_v7b_chunked_pinned")),
    ("extras", "head ablation, efficiency, cross-encoder, name swap WITH chunk pooling", lambda: _m("colab_v7b_extras").main(chunk_long=True, out="results/colab_v7b_extras_chunked")),
    ("newenc", "new encoders: tiers, frozen+hybrid, robustness", lambda: (
        _m("colab_v7b_tiers_sbert").main(encoders=ALL6 + ",GloVe-avg-300d", out="results/colab_v7b_newenc_pinned/tiers"),
        _m("colab_v7b_finetune_hybrid").main(encoders=NEW4, finetune=False, out="results/colab_v7b_newenc_pinned/hybrid"),
        _m("colab_v7b_robustness_sbert").main(encoders=NEW4, repeats=10, out="results/colab_v7b_newenc_pinned/robustness"))),
    ("robust_abbrev", "abbreviation/expansion robustness (original lexicon), MiniLM + mpnet", lambda: _m("colab_v7b_robustness_sbert").main(encoders="SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2", out="results/colab_v7b_chunked_pinned")),
    ("taskA", "TalentCLEF Task A (chunk vs truncate)", lambda: _m("colab_v7b_taskA_chunked").main(out="results/colab_v7b_taskA_pinned")),
    ("token_counts", "word-piece lengths", lambda: _m("colab_v7b_token_counts").main(encoders="SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2,JobBERT-v2-64", out="results/colab_v7b_pinned_tokens")),
    ("paraphrase_gen", "LLM paraphrases of the test resumes / jobs with the faithfulness filter (Phi-3.5-mini-instruct, MIT)", lambda: _m("gpu_paraphrase").generate()),
    ("paraphrase_eval", "paraphrase robustness: frozen + fine-tuned dense models (same suite as RQ2)", lambda: _m("gpu_paraphrase").evaluate()),
]


def _sync(src, dst):
    """copy src -> dst (rsync if present, else copytree); used to keep results on Google Drive between Colab sessions."""
    import os, subprocess, shutil
    if not os.path.isdir(src): return
    os.makedirs(dst, exist_ok=True)
    try: subprocess.run(["rsync", "-a", src.rstrip("/") + "/", dst.rstrip("/") + "/"], check=True)
    except Exception: shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sync", default=""); ap.add_argument("--only", default=""); ap.add_argument("--skip", default=""); ap.add_argument("--list", action="store_true"); a, _ = ap.parse_known_args()
    if a.sync: _sync(a.sync, "results"); print("restored results from", a.sync, flush=True)       # earlier sessions' results back into ./results
    if a.list:
        for n, d, _f in STAGES: print(f"{n:20s} {d}")
        return
    if a.sync:                          # background copy every 10 minutes, so a Colab disconnect in the middle of a long stage loses at most 10 minutes of work
        import threading
        def _loop():
            while True:
                time.sleep(600)
                try: _sync("results", a.sync)
                except Exception as e: print("[sync warning]", e, flush=True)
        threading.Thread(target=_loop, daemon=True).start()
    only = [x for x in a.only.split(",") if x]; skip = set(x for x in a.skip.split(",") if x); failed = []; t0 = time.time()
    for name, desc, fn in STAGES:
        if (only and name not in only) or name in skip: continue
        print("\n" + "=" * 24, name, "-", desc, f"[{(time.time() - t0) / 60:.0f} min elapsed]", flush=True)
        try: fn()
        except KeyboardInterrupt:
            raise
        except BaseException as e:      # SystemExit (e.g. 'No encoder could be loaded') is recorded as a failed stage, the other stages still run
            traceback.print_exc(); failed.append((name, type(e).__name__))
            try:
                import torch, gc; gc.collect(); torch.cuda.empty_cache()
            except Exception: pass
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception: pass
        if a.sync: _sync("results", a.sync); print("synced results ->", a.sync, flush=True)
    print("\nFAILED STAGES:", failed or "none", "| total minutes:", round((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
