"""Smoke / integrity tests for the RQ2 and RQ3 scripts (CPU, small)."""
import os, subprocess, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _run(code):
    r = subprocess.run([sys.executable, "-c", "import warnings; warnings.filterwarnings('ignore');" + code], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stderr[-2000:]

def test_rank_change_identity_and_reversal():
    sys.path.insert(0, HERE)
    from robustness_v2 import rank_change
    te = pd.DataFrame({"resume_id": [1, 1, 1, 2, 2, 2]}); a = np.array([.9, .5, .1, .2, .3, .8])
    assert rank_change(te, a, a) == (1.0, 0.0)
    tau, top = rank_change(te, a, -a); assert tau == -1.0 and top == 1.0

def test_truncation_matched_runs(tmp_path):
    _run(f"import rq2_truncation_matched as m; m.main(out=r'{tmp_path}', seeds=range(1))")
    s = pd.read_csv(tmp_path / "truncation_summary.csv"); assert {"matched", "shifted"} == set(s.design)
    full = s[s.words.astype(str) == "full"]; assert np.allclose(full[full.design == "matched"].auc.values, full[full.design == "shifted"].auc.values)   # same at full length

def test_checklist_runs(tmp_path):
    _run(f"import rq2_checklist_tests as m; m.main(out=r'{tmp_path}', seeds=range(1))")
    s = pd.read_csv(tmp_path / "checklist_summary.csv"); assert {"INV_filler", "INV_punct", "DIR_up_copy_job", "DIR_down_foreign"} == set(s.test)
    assert s[s.test == "DIR_up_copy_job"].pass_rate.min() > 0.9

def test_cross_source_shared_categories_and_balance(tmp_path):
    _run(f"import rq2_cross_source as m; m.main(n_splits=2, out=r'{tmp_path}')")
    s = pd.read_csv(tmp_path / "cross_source_summary.csv"); assert set(s.kind) == {"within", "cross"} and s.n_test_pairs.nunique() == 1   # equal test size in both sources
    assert "Blockchain" not in open(tmp_path / "NOTE_cross_source.txt").read().split("\n")[0]       # categories missing from SEEK are not in the shared set
