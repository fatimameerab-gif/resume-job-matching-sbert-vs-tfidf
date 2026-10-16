"""Reciprocal Rank Fusion of two rankers over the same query groups."""
import numpy as np
import pandas as pd


def rrf_fuse(rank_a, rank_b, k=60):
    """rank_* are 1-based ranks (smaller is better), aligned arrays."""
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


def ranks_from_scores(scores, higher_is_better=True):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if higher_is_better else scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks
