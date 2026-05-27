"""
One-off: build a single .zip of Colab/Drive training files. Writes only new artifacts under backend/experiments/.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES_REL = [
    "backend/models/train_hybrid_model.py",
    "backend/config/data_split_qc_filtered.json",
    "backend/models/evaluate_model_strict_v2.py",
    "backend/models/regenerate_attacked_dataset.py",
    "backend/experiments/colab_qc_package/README_AR.txt",
    "backend/experiments/colab_qc_package/MANIFEST_FILES.txt",
    "backend/experiments/colab_qc_package/colab_run_qc_training.py",
    "backend/experiments/run_qc_filtered_experiment.py",
]

OPTIONAL_REL = [
    "data/reduced.zip",
]


def main() -> None:
    out_zip = ROOT / "backend" / "experiments" / "CyberSatDetect_Colab_Drive_Upload.zip"
    readme_extra = ROOT / "backend" / "experiments" / "_drive_bundle_README.txt"

    readme_extra.write_text(
        """CyberSatDetect — حزمة رفع Google Drive / Google Colab
================================================

ما بداخل الـ ZIP:
  - كود التدريب والتقييم وتوليد الهجمات حسب المسارات داخل المجلد.
  - تقسيمة QC: backend/config/data_split_qc_filtered.json

البيانات (chunk_*.npy):
  إذا وُجد ملف data/reduced.zip عند بناء الأرشيف، يُدمَج تلقائياً.
  إذا لم يُدمَج (حجم أو غير موجود): انسخ مجلد data/reduced كامل أو reduced.zip يدوياً إلى Drive بجانب هذا الأرشيف.

التشغيل على كولاب بعد فك الضغط:
  python backend/experiments/colab_qc_package/colab_run_qc_training.py ^
    --repo-root . ^
    --output-root ./qc_outputs

ثبّت: pip install tensorflow numpy
""",
        encoding="utf-8",
    )

    missing = [rel for rel in FILES_REL if not (ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    to_bundle = [(ROOT / r, r) for r in FILES_REL]

    for rel in OPTIONAL_REL:
        p = ROOT / rel
        if p.is_file():
            to_bundle.append((p, rel))

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_extra, arcname="DRIVE_UPLOAD_README.txt")
        for abs_path, arcname in to_bundle:
            zf.write(abs_path, arcname=arcname.replace("\\", "/"))

    readme_extra.unlink(missing_ok=True)
    print("Wrote:", out_zip)
    print("Size MB:", round(out_zip.stat().st_size / (1024 * 1024), 2))


if __name__ == "__main__":
    main()
