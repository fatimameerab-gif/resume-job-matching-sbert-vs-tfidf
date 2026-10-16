import os, subprocess, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _run(code):
    r = subprocess.run([sys.executable, "-c", "import warnings; warnings.filterwarnings('ignore');" + code], capture_output=True, text=True, cwd=HERE); assert r.returncode == 0, r.stderr[-2000:]

def test_exact_and_linking_find_synonym():
    sys.path.insert(0, HERE); import skill_gap_v2 as sg
    skills = [("u1", ["machine learning", "ml"]), ("u2", ["welding"])]
    assert sg.exact_present("i use ml daily", skills).tolist() == [1.0, 0.0]
    sc = sg.TfidfCharScorer(["machine learning", "ml", "welding", "teaching"]); s = sc.score("experience in machine learnin and statistics", skills)   # typo still links
    assert s[0] > 0.5 > s[1]

def test_validity_runs(tmp_path):
    _run(f"import skill_gap_v2 as m; m.main(out=r'{tmp_path}', max_resumes=40)")
    t = pd.read_csv(tmp_path / "validity_auc_tfidf_char.csv"); assert t.auc_own_vs_other.between(0.5, 1).all()

def test_gold_kit_and_metrics_roundtrip(tmp_path):
    _run(f"import make_gap_gold_kit as k; k.main(n_resumes=8, k=6, out=r'{tmp_path}/kit')")
    sheet = pd.read_csv(tmp_path / "kit" / "annotation-sheet-skillgap.csv"); key = pd.read_csv(tmp_path / "kit" / "KEY-system-scores-do-not-share.csv"); assert len(sheet) == len(key) and (sheet.present.isna() | (sheet.present == "")).all()
    rng = np.random.default_rng(0); sc = sheet.merge(key[["item_id", "tfidf_char_score"]], on="item_id").tfidf_char_score.values
    for n, noise in (("a", 0.1), ("b", 0.15)):
        s = sheet.copy(); s["present"] = np.where(sc + rng.normal(0, noise, len(s)) > 0.45, "yes", "no"); s.to_csv(tmp_path / f"{n}.csv", index=False)
    _run(f"import compute_gap_metrics as m; m.main(r'{tmp_path}/a.csv', r'{tmp_path}/b.csv', r'{tmp_path}/kit/KEY-system-scores-do-not-share.csv', out=r'{tmp_path}/res', n_boot=20)")
    g = pd.read_csv(tmp_path / "res" / "gold_system_metrics.csv"); assert {"exact_string", "linking_tfidf_tau0.5"} <= set(g.system)
    assert pd.read_csv(tmp_path / "res" / "gold_agreement.csv").kappa_binary.iloc[0] > 0
