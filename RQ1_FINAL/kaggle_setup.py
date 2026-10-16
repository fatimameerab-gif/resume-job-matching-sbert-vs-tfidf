import os, glob, shutil, zipfile

MARKER = "data/PII_SCRUBBED"


def find_code_root(input_dir="/kaggle/input"):
    """Folder that contains data/PII_SCRUBBED (the extracted RQ1_FINAL dataset)."""
    hits = glob.glob(os.path.join(input_dir, "**", MARKER), recursive=True)
    roots = sorted({os.path.dirname(os.path.dirname(h)) for h in hits})
    if not roots:
        raise SystemExit("RQ1_FINAL dataset not found under " + input_dir + ": add the dataset to this notebook (Add Input) and check that it contains data/PII_SCRUBBED")
    return roots[0]


def setup(work="/kaggle/working", input_dir="/kaggle/input"):
    """Copy the code to work/RQ1_FINAL, restore results of earlier versions (attached notebook outputs), chdir. Returns the working folder."""
    src = find_code_root(input_dir); dst = os.path.join(work, "RQ1_FINAL")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("results", "__pycache__"))
    os.chdir(dst); os.makedirs("results", exist_ok=True)
    prev = [d for d in glob.glob(os.path.join(input_dir, "**", "results"), recursive=True) if os.path.isdir(d) and not d.startswith(src)]
    for d in prev:
        shutil.copytree(d, "results", dirs_exist_ok=True); print("restored earlier results from", d, flush=True)
    if not prev: print("no earlier results attached (fine for the first run)", flush=True)
    print("code root:", src, "| working folder:", dst, "| result folders so far:", sorted(os.listdir("results")), flush=True)
    return dst


def finalize(work="/kaggle/working"):
    """Keep only results in /kaggle/working (so the notebook output is small) and write one zip."""
    res = os.path.join(work, "RQ1_FINAL", "results")
    if not os.path.isdir(res): raise SystemExit("no results folder")
    out = os.path.join(work, "results"); shutil.rmtree(out, ignore_errors=True); shutil.copytree(res, out)
    shutil.rmtree(os.path.join(work, "RQ1_FINAL"), ignore_errors=True)
    zp = os.path.join(work, "results_for_paper.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs in os.walk(out):
            for f in fs: z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), work))
    print("results folders:", sorted(os.listdir(out))); print("zip:", zp, round(os.path.getsize(zp) / 1e6, 1), "MB")
