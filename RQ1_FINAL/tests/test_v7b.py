import numpy as np
import pandas as pd

from eval_protocol_v7 import build_pairs_v7, fit_hard_index, _job_text_key
from robustness import (perturb_text, perturb_counted, build_mapping, compile_mapping,
                        CORE_PAIRS, SAFE_SYNONYMS)


def _tiny():
    rows, jid = [], 0
    for cat in ["Alpha Dev", "Beta Dev", "Gamma Ops", "Delta Ops"]:
        for k in range(6):
            n = 8 if k % 2 == 0 else 60          # short and long jobs in every category
            txt = " ".join([f"{cat.split()[0].lower()}skill{(k*7+i) % 9}" for i in range(n)] + ["shared", "words"])
            rows.append(dict(job_id=jid, text_clean=txt, mapped_category=cat)); jid += 1
    pool = pd.DataFrame(rows)
    res = pd.DataFrame([dict(resume_id=i, text_clean=f"alphaskill{i%5} betaskill1 experience",
                             category=["Alpha Dev", "Beta Dev", "Gamma Ops", "Delta Ops"][i % 4]) for i in range(16)])
    return res, pool


def test_synonyms_no_longer_cancel_out():
    for abbr, full in CORE_PAIRS:
        assert perturb_text(f"used {abbr} daily", build_mapping(direction="abbr_to_full")) != f"used {abbr} daily"
        assert perturb_text(f"used {full} daily", build_mapping(direction="full_to_abbr")) != f"used {full} daily"
    assert perturb_text("used javascript daily") == "used js daily"          # legacy default direction
    assert "hr" not in {k for k in SAFE_SYNONYMS} and "hr" not in {a for a, _ in CORE_PAIRS}


def test_perturb_rate_zero_and_one():
    mp = build_mapping(direction="abbr_to_full"); rx = compile_mapping(mp)
    txt = "aws and api and qa"
    out0, m0, r0, _ = perturb_counted(txt, mp, rx, rate=0.0, rng=np.random.default_rng(0))
    assert out0 == txt and m0 == 3 and r0 == 0
    out1, m1, r1, _ = perturb_counted(txt, mp, rx, rate=1.0)
    assert m1 == r1 == 3 and "amazon web services" in out1


def test_ms_excel_longest_match_first():
    mp = build_mapping(direction="abbr_to_full")
    assert perturb_text("ms excel expert", mp) == "microsoft excel expert"


def test_negative_reuse_cap_and_shared_positives_across_tiers():
    res, pool = _tiny(); hv, hm = fit_hard_index(pool)
    pos = {}
    for tier in ("random_other_category", "same_family", "hard_mined"):
        kw = dict(hard_vectorizer=hv, hard_job_matrix=hm) if tier == "hard_mined" else {}
        df = build_pairs_v7(res, pool, "test", np.random.RandomState(5), tier,
                            pos_rng=np.random.RandomState(9), neg_reuse_cap=2, **kw)
        assert df[df.label == 0].job_id.value_counts().max() <= 2
        pos[tier] = df[df.label == 1].sort_values("resume_id").job_id.tolist()
    assert pos["random_other_category"] == pos["same_family"] == pos["hard_mined"]


def test_length_match_limits_negative_length_ratio():
    res, pool = _tiny(); hv, hm = fit_hard_index(pool)
    df = build_pairs_v7(res, pool, "test", np.random.RandomState(1), "hard_mined", hard_vectorizer=hv,
                        hard_job_matrix=hm, pos_rng=np.random.RandomState(2), length_match_ratio=2.0,
                        drop_contradictory=True)
    pl = df[df.label == 1].set_index("resume_id").job_text.str.split().str.len()
    for _, r in df[df.label == 0].iterrows():
        ratio = len(r.job_text.split()) / pl[r.resume_id]
        assert 0.5 <= ratio <= 2.0, ratio


def test_drop_contradictory_removes_texts_shared_with_own_category():
    res, pool = _tiny()
    dup = pool.iloc[[0]].copy(); dup["job_id"] = 999; dup["mapped_category"] = "Beta Dev"   # same text as an Alpha job
    pool = pd.concat([pool, dup], ignore_index=True)
    df = build_pairs_v7(res[res.category == "Alpha Dev"], pool, "test", np.random.RandomState(3),
                        "random_other_category", pos_rng=np.random.RandomState(4), drop_contradictory=True)
    bad_key = _job_text_key(pool.iloc[0].text_clean)
    assert not ((df.label == 0) & (df.job_text_key == bad_key)).any()


def test_compute_agreement_perfect_and_disagreement(tmp_path):
    import subprocess, sys, os
    sheet = pd.DataFrame({"item_id": [f"A{i:03d}" for i in range(12)], "resume_text": "r", "job_text": "j",
                          "annotator1_score_0_1_2": [0, 1, 2] * 4, "annotator1_notes": "",
                          "annotator2_score_0_1_2": [0, 1, 2] * 4, "annotator2_notes": ""})
    sheet.loc[3, "annotator2_score_0_1_2"] = 2                      # one disagreement
    p = tmp_path / "sheet.csv"; sheet.to_csv(p, index=False)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(here, "compute_agreement.py"), str(p)],
                       capture_output=True, text=True, cwd=here)
    assert r.returncode == 0, r.stderr
    dis = pd.read_csv(tmp_path / "disagreements_to_adjudicate.csv")
    assert len(dis) == 1 and "adjudicated_score" in dis.columns
