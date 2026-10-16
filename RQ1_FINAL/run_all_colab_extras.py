import traceback, importlib
NEW = "SBERT-MiniLM-L6-v2,SBERT-mpnet-base-v2,e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2"
steps = [
    ("6 extras (head ablation, efficiency, cross-encoder, name swap)", "colab_v7b_extras", {}),
    ("7 truncation alternatives", "colab_v7b_truncation", {}),
    ("1 tiers, all encoders (+GloVe)", "colab_v7b_tiers_sbert",
     dict(encoders=NEW + ",GloVe-avg-300d", out="results/colab_v7b_newenc/tiers")),
    ("5 frozen + hybrid, new encoders (no fine-tuning)", "colab_v7b_finetune_hybrid",
     dict(encoders="e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2", finetune=False, out="results/colab_v7b_newenc/hybrid")),
    ("2 robustness, new encoders", "colab_v7b_robustness_sbert",
     dict(encoders="e5-base-v2,bge-base-en-v1.5,gte-base,JobBERT-v2", out="results/colab_v7b_newenc/robustness")),
]
failed = []
for title, mod, kw in steps:
    print("\n" + "=" * 20, title, "=" * 20, flush=True)
    try:
        importlib.import_module(mod).main(**kw)
    except BaseException as e:
        traceback.print_exc(); failed.append((title, type(e).__name__)); 
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
print("\nFAILED STEPS:", failed or "none")
print("Now zip and send: !zip -r new_results.zip results/colab_v7b_extras results/colab_v7b_truncation results/colab_v7b_newenc")
