import os
import numpy as np
import pandas as pd

from hybrid_rrf import rrf_fuse, ranks_from_scores
from stats_utils import paired_permutation_pvalue, benjamini_hochberg, min_detectable_effect_paired
from text_processing import head_only, head_tail, apply_strategy
from robustness import perturb_text, SAFE_SYNONYMS
from split_diagnostics import diagnose_synthetic_pairs, assert_resume_disjoint
from eval_protocol_v7 import family_of, _job_text_key


def test_rrf_prefers_docs_ranked_well_by_both():
    scores_a = np.array([3.0, 2.0, 1.0])
    scores_b = np.array([1.0, 3.0, 2.0])
    fused = rrf_fuse(ranks_from_scores(scores_a), ranks_from_scores(scores_b), k=60)
    assert fused.argmax() in (0, 1)


def test_permutation_identical_scores_not_small_p():
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    delta, p = paired_permutation_pvalue(a, a, n_perm=200, seed=0)
    assert abs(delta) < 1e-12
    assert p > 0.5


def test_mde_increases_when_n_falls():
    assert min_detectable_effect_paired(10, 0.05) > min_detectable_effect_paired(40, 0.05)


def test_bh_monotonic():
    adj, _ = benjamini_hochberg([0.001, 0.02, 0.8])
    assert adj[0] <= adj[1] <= adj[2]


def test_head_tail_preserves_ends():
    text = " ".join(f"w{i}" for i in range(20))
    out = head_tail(text, n_tokens=6).split()
    assert out[:3] == ["w0", "w1", "w2"]
    assert out[-3:] == ["w17", "w18", "w19"]
    assert len(head_only(text, 4).split()) == 4
    assert apply_strategy(text, "full") == text


def test_synonyms_do_not_touch_labels_and_avoid_category_tokens():
    assert "data science" not in {k.lower() for k in SAFE_SYNONYMS}
    src = "used tensorflow and postgresql"
    out = perturb_text(src)
    assert "tf" in out or "tensorflow" in out
    assert src != out


def test_leakage_assertions_on_tiny_pairs():
    pairs = pd.DataFrame({
        "split": ["train", "train", "test", "test"],
        "resume_id": [0, 0, 1, 1],
        "job_id": [10, 11, 10, 12],
        "resume_text": ["a", "a", "b", "b"],
        "job_text": ["j0", "j1", "j0", "j2"],
        "label": [1, 0, 1, 0],
    })
    assert_resume_disjoint(pairs)
    diag = diagnose_synthetic_pairs(pairs)
    # diagnose_synthetic_pairs uses itertools.combinations which yields pairs in
    # sorted order (alphabetical by split name), so the cross-split row is
    # split_a="test", split_b="train" (not "train"/"test").  Both orderings
    # represent the same overlap; we query for whichever ordering is present.
    cross = diag[
        ((diag.split_a == "train") & (diag.split_b == "test")) |
        ((diag.split_a == "test")  & (diag.split_b == "train"))
    ]
    assert len(cross) == 1, f"Expected exactly one cross-split row; got:\n{diag}"
    overlap = cross.iloc[0]
    assert overlap["resume_id_overlap"] == 0
    assert overlap["job_id_overlap"] == 1
    assert overlap["job_text_overlap"] == 1


def test_job_text_key_collapses_duplicates():
    assert _job_text_key("abc") == _job_text_key("abc")
    assert _job_text_key("abc") != _job_text_key("abd")
    assert family_of("Java Developer") == "software"
    assert family_of("Data Science") == "data"


def test_data_protocol_refuses_to_require_overwrite_logic():
    src = open(os.path.join(os.path.dirname(__file__), "..", "data_protocol_v2.py"), encoding="utf-8").read()
    assert "Refusing to overwrite" in src
