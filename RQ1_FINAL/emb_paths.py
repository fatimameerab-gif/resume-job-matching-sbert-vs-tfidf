"""Where the frozen embeddings live. Order: $EMB_DIR, results/embeddings_pinned (single pinned GPU run), results/embeddings, ../RQ2_FINAL/results/embeddings (original mixed-environment export)."""
import os
EMB_DIR = os.environ.get("EMB_DIR") or next((p for p in ("results/embeddings_pinned", "results/embeddings", "../RQ2_FINAL/results/embeddings") if os.path.isdir(p)), "results/embeddings")
