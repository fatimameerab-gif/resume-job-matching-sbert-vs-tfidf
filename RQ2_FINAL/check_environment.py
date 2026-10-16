"""Compare installed packages with the exact pins in requirements.txt and record the result.
    python check_environment.py            # prints + writes results/v7/environment_check.txt
    python check_environment.py --strict   # exit 1 on any mismatch (use before paper-final runs)
Pinned-vs-unpinned matters here: results moved with library versions (local word2vec: 0.640 on
the pinned set vs 0.667 on scikit-learn 1.8 / numpy 2.4 / newer gensim)."""
import re, sys, os
from importlib import metadata

def main(strict=False):
    lines, bad = [], 0
    for raw in open("requirements.txt", encoding="utf-8"):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", raw.strip())
        if not m:
            continue
        name, want = m.groups()
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "NOT INSTALLED"
        ok = have == want
        bad += (not ok)
        lines.append(f"{'OK      ' if ok else 'MISMATCH'} {name:22s} pinned={want:10s} installed={have}")
    lines.append(f"\n{bad} mismatch(es). Python {sys.version.split()[0]}")
    text = "\n".join(lines)
    print(text)
    os.makedirs("results/v7", exist_ok=True)
    open("results/v7/environment_check.txt", "w", encoding="utf-8").write(text + "\n")
    if strict and bad:
        sys.exit(1)

if __name__ == "__main__":
    main("--strict" in sys.argv)
