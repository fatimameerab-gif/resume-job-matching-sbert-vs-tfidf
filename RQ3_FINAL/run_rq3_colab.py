"""RQ3 needs frozen SBERT features: run ONCE on Colab (GPU), then send results/embeddings back.
    %run run_rq3_colab.py
(Fine-tuned bi-encoder and cross-encoder rows come from the RQ1 Colab runs: colab_v7b_finetune_hybrid.py and colab_v7b_extras.py.)"""
import colab_export_embeddings
colab_export_embeddings.main()
