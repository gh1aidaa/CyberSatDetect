"""
Build ONE zip for Google Colab: all training/eval code + split JSON + data manifest + notebook.

Does NOT embed full .npy / .npz datasets (too large). The manifest lists every filename you must
place under data/reduced and data/attacked_v2 on Colab (or Drive) to match the split.

Usage (from repo root):
  python backend/experiments/univariate_pattern_solution_experiment/build_colab_full_package.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set


EXP_REL = Path("backend/experiments/univariate_pattern_solution_experiment")
SPLIT_REL = Path("backend/config/data_split_qc_filtered.json")


def _collect_py_files(repo: Path) -> List[Path]:
    root = (repo / EXP_REL).resolve()
    out: List[Path] = []
    for f in root.rglob("*.py"):
        rel = f.relative_to(root)
        parts = set(rel.parts)
        if "results" in parts or "models" in parts:
            continue
        out.append(f)
    return sorted(out)


def _load_split(repo: Path) -> Dict[str, Any]:
    with (repo / SPLIT_REL).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _manifest_from_split(split: Dict[str, Any]) -> Dict[str, Any]:
    train = list(split.get("train", []))
    val = list(split.get("validation", []))
    test = list(split.get("test", []))
    all_chunks: Set[str] = set(train) | set(val) | set(test)
    return {
        "normal_dir_expected": "data/reduced",
        "attacked_dir_expected": "data/attacked_v2",
        "train_chunk_count": len(train),
        "validation_chunk_count": len(val),
        "test_chunk_count": len(test),
        "unique_reduced_npy_count": len(all_chunks),
        "train_files": train,
        "validation_files": val,
        "test_files": test,
        "note_ar": (
            "انسخ كل ملفات chunk_*.npy المذكورة في train/validation/test إلى مجلد واحد على Colab "
            "مثلاً /content/data/reduced/ بنفس الأسماء. ملفات attacked_v2 (*.npz) ضعها في /content/data/attacked_v2/"
        ),
    }


def _notebook_json() -> Dict[str, Any]:
    """Minimal Colab notebook: install TF, set cwd, train, evaluate."""
    cells: List[Dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Univariate pattern experiment — Colab\n",
                "\n",
                "1. ارفع الملف `colab_univariate_pattern_full.zip` إلى Colab ثم فك الضغط تحت `/content/repo`.\n",
                "2. اربط Google Drive (اختياري) وانسخ `data/reduced` و `data/attacked_v2` إلى `/content/repo/data/`.\n",
                "3. شغّل الخلايا بالترتيب.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip -q install \"tensorflow>=2.15,<2.17\"\n",
                "import os, sys, subprocess\n",
                "REPO = \"/content/repo\"  # غيّرها إذا فككت الـ zip في مسار آخر\n",
                "os.chdir(REPO)\n",
                "if REPO not in sys.path:\n",
                "    sys.path.insert(0, REPO)\n",
                "print(\"cwd:\", os.getcwd())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# from google.colab import drive\n",
                "# drive.mount(\"/content/drive\")\n",
                "# ثم انسخ البيانات، مثلاً:\n",
                "# !mkdir -p /content/repo/data/reduced /content/repo/data/attacked_v2\n",
                "# !cp -r \"/content/drive/MyDrive/path/to/reduced/\"*.npy /content/repo/data/reduced/\n",
                "# !cp -r \"/content/drive/MyDrive/path/to/attacked_v2/\"*.npz /content/repo/data/attacked_v2/\n",
                "pass\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cmd = (\n",
                "  \"python backend/experiments/univariate_pattern_solution_experiment/train_univariate_pattern_model.py \"\n",
                "  f\"--repo-root {REPO} \"\n",
                "  \"--split-file backend/config/data_split_qc_filtered.json \"\n",
                "  \"--normal-dir data/reduced \"\n",
                "  \"--output-dir backend/experiments/univariate_pattern_solution_experiment/results \"\n",
                "  \"--models-dir backend/experiments/univariate_pattern_solution_experiment/models \"\n",
                "  \"--epochs 3 --batch-size 256\"\n",
                ")\n",
                "print(cmd)\n",
                "subprocess.check_call(cmd, shell=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cmd = (\n",
                "  \"python backend/experiments/univariate_pattern_solution_experiment/evaluate_univariate_pattern_model.py \"\n",
                "  f\"--repo-root {REPO} \"\n",
                "  \"--baseline-results backend/experiments/univariate_pattern_solution_experiment/results \"\n",
                "  \"--split-file backend/config/data_split_qc_filtered.json \"\n",
                "  \"--normal-dir data/reduced \"\n",
                "  \"--attacked-dir data/attacked_v2 \"\n",
                "  \"--models-dir backend/experiments/univariate_pattern_solution_experiment/models \"\n",
                "  \"--output-dir backend/experiments/univariate_pattern_solution_experiment/results \"\n",
                "  \"--conservative-fusion on\"\n",
                ")\n",
                "print(cmd)\n",
                "subprocess.check_call(cmd, shell=True)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## النتائج\n",
                "- `backend/experiments/univariate_pattern_solution_experiment/results/model_selection_comparison.csv`\n",
                "- `.../overall_univariate_pattern_metrics.csv`\n",
                "- `.../overall_conservative_fusion_metrics.csv` (إذا فعلت fusion)\n",
                "- `.../univariate_pattern_solution_report.md`\n",
            ],
        },
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="backend/experiments/univariate_pattern_solution_experiment/results/colab_univariate_pattern_full.zip",
    )
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    out = (repo / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    split = _load_split(repo)
    manifest = _manifest_from_split(split)
    attacked_dir = repo / "data" / "attacked_v2"
    npz_list: List[str] = []
    if attacked_dir.is_dir():
        npz_list = sorted([p.name for p in attacked_dir.glob("*.npz")])
    manifest["attacked_v2_npz_files_found_in_repo"] = npz_list
    manifest["attacked_v2_npz_count"] = len(npz_list)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Config
        zf.write(repo / SPLIT_REL, SPLIT_REL.as_posix())

        # Experiment Python sources only
        for f in _collect_py_files(repo):
            arc = f.relative_to(repo)
            zf.write(f, arcname=arc.as_posix())

        # Small baseline CSV for evaluate (packaged copy)
        baseline = repo / EXP_REL / "results" / "baseline_comparison_summary.csv"
        if baseline.is_file():
            zf.write(
                baseline,
                "backend/experiments/univariate_pattern_solution_experiment/results/baseline_comparison_summary.csv",
            )

        zf.writestr(
            "backend/experiments/univariate_pattern_solution_experiment/colab_package/DATA_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "backend/experiments/univariate_pattern_solution_experiment/colab_package/AR_README.md",
            _readme_ar(),
        )
        zf.writestr(
            "Colab_Univariate_Pattern.ipynb",
            json.dumps(_notebook_json(), indent=1),
        )
        zf.writestr(
            "PACKAGE_INFO.txt",
            "colab_univariate_pattern_full.zip\n"
            "- Unzips to a repo-shaped tree under backend/...\n"
            "- You must add data/reduced/*.npy and data/attacked_v2/*.npz yourself (see DATA_MANIFEST.json).\n"
            "- Open Colab_Univariate_Pattern.ipynb and set REPO path.\n",
        )

    print(out)
    return 0


def _readme_ar() -> str:
    return """# حزمة Colab — تجربة Univariate Pattern

## ما داخل الـ ZIP
- كل سكربتات التدريب والتقييم تحت `backend/experiments/univariate_pattern_solution_experiment/`
- `backend/config/data_split_qc_filtered.json`
- `colab_package/DATA_MANIFEST.json` — قائمة **كل** أسماء ملفات `chunk_*.npy` المطلوبة للتدريب/المعايرة/التقييم
- `baseline_comparison_summary.csv` — للمقارنة مع Chapter 7 في التقييم
- `Colab_Univariate_Pattern.ipynb` — تشغيل سريع

## البيانات (لا تُرفع داخل الـ ZIP لحجمها)
1. أنشئ على Colab (أو Drive):
   - `data/reduced/` — ضع **كل** الملفات المذكورة في الـ manifest (train + validation + test)
   - `data/attacked_v2/` — ضع ملفات `.npz` الخاصة بالتقييم فقط
2. جذر المشروع يجب أن يحتوي مجلد `backend/` بعد فك الضغط، بحيث يعمل `--repo-root` على المجلد الذي يحوي `backend` و `data`.

## أوامر يدوية (بدون Notebook)
من جذر المشروع بعد فك الضغط:

```
python backend/experiments/univariate_pattern_solution_experiment/train_univariate_pattern_model.py ^
  --repo-root . --split-file backend/config/data_split_qc_filtered.json --normal-dir data/reduced ^
  --output-dir backend/experiments/univariate_pattern_solution_experiment/results ^
  --models-dir backend/experiments/univariate_pattern_solution_experiment/models --epochs 3 --batch-size 256
```

```
python backend/experiments/univariate_pattern_solution_experiment/evaluate_univariate_pattern_model.py ^
  --repo-root . --baseline-results backend/experiments/univariate_pattern_solution_experiment/results ^
  --split-file backend/config/data_split_qc_filtered.json --normal-dir data/reduced --attacked-dir data/attacked_v2 ^
  --models-dir backend/experiments/univariate_pattern_solution_experiment/models ^
  --output-dir backend/experiments/univariate_pattern_solution_experiment/results --conservative-fusion on
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
