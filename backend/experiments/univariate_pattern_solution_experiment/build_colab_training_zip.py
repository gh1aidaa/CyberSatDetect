"""
Build a portable zip for Google Colab training (code + split JSON only).

Usage (from repo root):
  python backend/experiments/univariate_pattern_solution_experiment/build_colab_training_zip.py --repo-root .
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="backend/experiments/univariate_pattern_solution_experiment/results/univariate_pattern_colab_training_bundle.zip",
    )
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    out = (repo / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    paths = [
        Path("backend/experiments/univariate_pattern_solution_experiment"),
        Path("backend/config/data_split_qc_filtered.json"),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            p = (repo / rel).resolve()
            if not p.exists():
                continue
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix in {".py", ".md", ".csv", ".json"}:
                        arc = f.relative_to(repo)
                        zf.write(f, arcname=str(arc).replace("\\", "/"))
            else:
                zf.write(p, arcname=str(rel).replace("\\", "/"))
        zf.writestr(
            "COLAB_README.txt",
            "Upload data/reduced to match split JSON, then run train_univariate_pattern_model.py as in Part 12 of the experiment spec.\n",
        )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
