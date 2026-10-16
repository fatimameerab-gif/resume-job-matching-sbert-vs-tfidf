"""Smoke / integrity tests for the RQ2 and RQ3 scripts (CPU, small)."""
import os, subprocess, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _run(code):
    r = subprocess.run([sys.executable, "-c", "import warnings; warnings.filterwarnings('ignore');" + code], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr[-2000:]

def test_rq3_ece_and_holm():
    sys.path.insert(0, HERE)
    import rq3_classifier_grid_v3 as m
    y = np.array([0, 0, 1, 1]); assert m.ece(y, np.array([0., 0., 1., 1.])) == 0.0
    assert np.allclose(m.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

def test_rq3_cpu_mini(tmp_path):
    _run(f"import rq3_classifier_grid_v3 as m; m.main(out=r'{tmp_path}', n_b=2, designs=('A','B'), classifiers=['CosineThreshold','LR','RBF-SVM','kNN'], reps=[m.Rep('TF-IDF', m.tfidf_rep)])")
    s = pd.read_csv(tmp_path / "grid_summary_B.csv"); assert len(s) == 4 and s.auc.between(0.5, 1).all()

def test_export_embeddings_and_rq3_dense(tmp_path):
    _run(f"import colab_export_embeddings as m; m.main(encoders='stub', out=r'{tmp_path}/emb')")
    assert (tmp_path / "emb" / "stub.npz").exists()
    _run(f"import rq3_classifier_grid_v3 as m; m.EMB_DIR=r'{tmp_path}/emb'; m.main(out=r'{tmp_path}/rq3', n_b=3, designs=('A','B'), classifiers=['CosineThreshold','LR','kNN'], reps=[r for r in m.make_reps() if r.name=='stub'])")
    s = pd.read_csv(tmp_path / "rq3" / "grid_summary_A.csv"); assert set(s.classifier) == {"CosineThreshold", "LR", "kNN"} and (s.representation == "stub").all()
