#!/usr/bin/env python3
"""
Google Colab / local runner: trains with backend/config/data_split_qc_filtered.json
and writes ONLY under --output-root (does not modify backend/app or default paths).

Usage (from repo root):
  python backend/experiments/colab_qc_package/colab_run_qc_training.py \\
    --repo-root . \\
    --output-root ./qc_colab_outputs

Colab example:
  python backend/experiments/colab_qc_package/colab_run_qc_training.py \\
    --repo-root /content/CyberSatDetectprojct \\
    --output-root /content/drive/MyDrive/qc_colab_run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="Project root containing backend/, data/reduced/",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Where to save model, checkpoints, thresholds, logs (created if missing)",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.output_root).resolve()

    models_dir = repo / "backend" / "models"
    split_path = repo / "backend" / "config" / "data_split_qc_filtered.json"
    data_dir = repo / "data" / "reduced"

    if not models_dir.is_dir():
        raise FileNotFoundError(models_dir)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    model_d = out / "model"
    res_d = out / "results"
    logs_d = out / "logs"
    for d in (model_d, res_d, logs_d):
        d.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(models_dir))

    import train_hybrid_model as thm

    thm.ROOT = repo
    thm.BACKEND = repo / "backend"
    thm.APP_DIR = model_d
    thm.DATA_DIR = data_dir
    thm.SPLIT_FILE = split_path
    thm.MODEL_OUT = model_d / "final_model.keras"
    thm.THRESH_OUT = res_d / "thresholds_qc_filtered.json"
    thm.CHECKPOINT_MODEL = model_d / "checkpoint_model.keras"
    thm.CHECKPOINT_INFO = model_d / "checkpoint_info.json"

    os.chdir(repo)

    log_path = logs_d / "training.log"

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    tee = Tee(sys.stdout, open(log_path, "w", encoding="utf-8"))
    old_out = sys.stdout
    sys.stdout = tee
    try:
        thm.main()
    finally:
        sys.stdout = old_out
        tee.files[1].close()

    best_src = model_d / "best_model.keras"
    best_dst = model_d / "best_model_qc_filtered.keras"
    if best_src.exists():
        import shutil

        shutil.copy2(best_src, best_dst)

    print(f"\nDone. Outputs under:\n  {out}\n")


if __name__ == "__main__":
    main()
