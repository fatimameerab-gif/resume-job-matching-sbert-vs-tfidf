import sys, math
import numpy as np, pandas as pd
OUT = "results/paraphrases"


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def make(n_per_role=30, seed=0):
    C = pd.read_csv(f"{OUT}/chunks.csv"); C = C[C.accepted == 1].drop_duplicates("chunk_id", keep="first")
    rng = np.random.default_rng(seed); parts = []
    for role in ("resume", "job"):
        g = C[C.role == role]; parts.append(g.iloc[rng.permutation(len(g))[:n_per_role]])
    S = pd.concat(parts).iloc[lambda d: rng.permutation(len(d))].reset_index(drop=True)
    S.insert(0, "id", range(1, len(S) + 1))
    sheet = S[["id", "role", "original", "paraphrase"]].assign(meaning_preserved="", info_added="", info_lost="", fluent="", notes="")
    sheet.to_csv(f"{OUT}/human_check_sheet.csv", index=False); print(f"wrote {OUT}/human_check_sheet.csv with {len(sheet)} pairs")


def summarise(path=f"{OUT}/human_check_sheet_filled.csv"):
    S = pd.read_csv(path); rows = []
    for col in ("meaning_preserved", "info_added", "info_lost"):
        x = pd.to_numeric(S[col], errors="coerce").dropna(); k, n = int(x.sum()), len(x); lo, hi = wilson(k, n)
        rows.append(dict(item=col, n=n, yes=k, rate=k / n if n else np.nan, ci_lo=lo, ci_hi=hi))
        for role in ("resume", "job"):
            y = pd.to_numeric(S.loc[S.role == role, col], errors="coerce").dropna(); k2, n2 = int(y.sum()), len(y); l2, h2 = wilson(k2, n2)
            rows.append(dict(item=f"{col} [{role}]", n=n2, yes=k2, rate=k2 / n2 if n2 else np.nan, ci_lo=l2, ci_hi=h2))
    f = pd.to_numeric(S.fluent, errors="coerce").dropna(); rows.append(dict(item="fluent (mean 1-5)", n=len(f), yes=np.nan, rate=f.mean(), ci_lo=np.nan, ci_hi=np.nan))
    R = pd.DataFrame(rows).round(3); R.to_csv(f"{OUT}/human_check_summary.csv", index=False); print(R.to_string(index=False))


if __name__ == "__main__":
    {"make": make, "summarise": summarise}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
