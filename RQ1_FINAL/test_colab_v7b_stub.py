"""Smoke tests: the three Colab scripts run end-to-end with the offline 'stub' encoder.
These do NOT test SBERT quality -- only that paths, columns and shapes work before you spend GPU time."""
import os, subprocess, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(module, call):
    code = f"import warnings; warnings.filterwarnings('ignore'); import {module} as m; m.main({call})"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr[-2000:]


def test_tiers_script(tmp_path):
    _run("colab_v7b_tiers_sbert", f"encoders='stub', seeds=1, out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "tiers_sbert_summary.csv")
    assert set(s.model) == {"TF-IDF", "stub"}
    assert {"random_other_category", "same_family", "hard[TF-IDF]", "hard[stub]", "hard[consensus]", "hard[consensus_family]"} == set(s.tier)
    assert s[(s.tier == "hard[stub]") & (s.model == "stub")].self_mined.all()


def test_robustness_script(tmp_path):
    _run("colab_v7b_robustness_sbert", f"encoders='stub', seeds=1, repeats=1, out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "robustness_sbert_summary.csv")
    assert {"full", "cosLR"} == set(s["head"]) and len(s) == 2 * 2 * 3 * 4
    assert (tmp_path / "NOTE_lexicon_unvalidated.txt").exists()


def test_token_count_script(tmp_path):
    _run("colab_v7b_token_counts", f"encoders='stub', out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "token_counts_summary.csv")
    assert {"v3_resume", "v3_job"} <= set(s.source)


def test_finetune_hybrid_script(tmp_path):
    _run("colab_v7b_finetune_hybrid", f"encoders='stub', seeds=1, finetune=False, n_boot=20, out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "finetune_summary.csv")
    assert any("Hybrid" in m and "RRF" in m for m in s.model) and any("weighted" in m for m in s.model)
    p = pd.read_csv(tmp_path / "pair_predictions.csv")
    assert {"resume_id", "job_id", "label", "score", "pred", "correct", "model"} <= set(p.columns)
    pv = pd.read_csv(tmp_path / "paired_vs_tfidf.csv")
    assert {"mcnemar_p", "ci_lo", "ci_hi"} <= set(pv.columns)


def test_fusion_helpers():
    import numpy as np
    sys.path.insert(0, HERE)
    import colab_v7b_finetune_hybrid as m
    a = np.array([0.9, 0.1, 0.8, 0.2]); b = np.array([0.7, 0.3, 0.2, 0.9])
    r = m._rrf(a, b)
    assert len(r) == 4 and r[0] > r[1]                       # item 0 is ranked high by both models
    y = np.array([1, 0, 1, 0])
    assert 0.0 <= m._best_threshold(a, y) <= 1.0


def test_taskA_chunked_script(tmp_path):
    _run("colab_v7b_taskA_chunked", f"encoders='stub', out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "taskA_summary.csv")
    assert {"truncate", "chunk"} <= set(s["mode"]) and (s.accuracy.between(0, 1)).all()


def test_extras_script(tmp_path):
    _run("colab_v7b_extras", f"encoders='stub', seeds=1, out=r'{tmp_path}', cross=None")
    s = pd.read_csv(tmp_path / "head_ablation_summary.csv")
    assert {"cos", "abs+cos", "uv_abs_prod"} == set(s["head"]) and {"TF-IDF", "stub"} == set(s.model)
    assert (tmp_path / "name_swap_summary.csv").exists() and (tmp_path / "efficiency.csv").exists()


def test_truncation_script(tmp_path):
    _run("colab_v7b_truncation", f"encoders='stub', seeds=1, out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "truncation_summary.csv")
    assert {"head", "head_tail", "section_aware", "chunk_mean", "chunk_max"} == set(s.strategy)


def test_export_embeddings_and_rq3_dense(tmp_path):
    import importlib.util, pytest
    if importlib.util.find_spec('rq3_classifier_grid_v3') is None:
        pytest.skip('RQ3 code lives in RQ3_FINAL')
    _run("colab_export_embeddings", f"encoders='stub', out=r'{tmp_path}/emb', pairs_path='data/resume_job_pairs_v3.csv'")
    assert (tmp_path / "emb" / "stub.npz").exists()
    code = (f"import warnings; warnings.filterwarnings('ignore'); import rq3_classifier_grid_v3 as m; m.EMB_DIR=r'{tmp_path}/emb'; "
            f"m.main(out=r'{tmp_path}/rq3', n_b=3, designs=('A','B'), classifiers=['CosineThreshold','LR','kNN'], reps=[r for r in m.make_reps() if r.name=='stub'])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE); assert r.returncode == 0, r.stderr[-2000:]
    s = pd.read_csv(tmp_path / "rq3" / "grid_summary_A.csv"); assert set(s.classifier) == {"CosineThreshold", "LR", "kNN"} and (s.representation == "stub").all()


def test_learning_curve_script(tmp_path):
    _run("colab_v7b_learning_curve", f"encoders='stub', seeds=1, fractions=(0.25, 1.0), out=r'{tmp_path}'")
    s = pd.read_csv(tmp_path / "learning_curve_summary.csv")
    assert {"TF-IDF head", "frozen head", "fine-tuned head", "fine-tuned cosine", "frozen cosine"} == set(s.model) and set(s.fraction) == {0.25, 1.0}
    _run("colab_v7b_learning_curve", f"encoders='stub', seeds=1, fractions=(0.25, 1.0), out=r'{tmp_path}'")   # re-run resumes from the checkpoint
    assert len(pd.read_csv(tmp_path / "learning_curve_per_seed.csv")) == 10
