# CyberSatDetect — Unused / Duplicate / Old Files Report

Scope (scanned): `backend/`, `frontend/`, `archive_unused/`, `scripts/`, `docker/`, `db/`, repo root, plus notable generated artifacts under `backend/app/`.

Rules followed:
- **No files were deleted.**
- A file is **not** marked unused just because it’s not imported; runtime may reference it by **path** (e.g., models, thresholds, static frontend files mounted by FastAPI).
- The backend runtime entrypoint is **`C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\backend\app\api.py`**, which serves `frontend/` via a `/frontend` static mount and loads model/threshold/config artifacts from `backend/app/` / DB registry.

---

## A) ملفات مؤكدة غير مستخدمة

### 1) Virtual environments checked into repo

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\venv\`
  - **Reason**: Python virtualenv folder (local environment). Not imported, not referenced by runtime routes, not used by docker/run scripts.
  - **Confidence**: **High**
  - **Can delete?**: **Review** (delete if you don’t intend to keep venv in repo)
  - **Safe delete command**:
    - `# rm -rf venv`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\.venv\`
  - **Reason**: Duplicate virtualenv folder; same rationale.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm -rf .venv`

### 2) Archived/duplicate code snapshots not referenced by runtime

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_app\detect_hybrid.py`
  - **Reason**: Lives under `archive_unused/` and not referenced by `backend/app/api.py` routes/imports. Appears to be a legacy inference script.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_app/detect_hybrid.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_app\db_connections.py`
  - **Reason**: Archived DB helper not imported by the active backend.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_app/db_connections.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_models\evaluate_model.py`
  - **Reason**: Archived evaluation script; not used by runtime or CLI pipeline.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_models/evaluate_model.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_models\evaluate_attacks.py`
  - **Reason**: Archived evaluation script; not referenced.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_models/evaluate_attacks.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_models\optimize_thresholds.py`
  - **Reason**: Archived threshold optimizer; not referenced by runtime.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_models/optimize_thresholds.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_models\tune_score_weights.py`
  - **Reason**: Archived tuning script; not referenced.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_models/tune_score_weights.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\archive_unused\backend_models\train_hybrid_model_v2.py`
  - **Reason**: Archived training script (v2) not referenced by current runtime/training entrypoints.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm archive_unused/backend_models/train_hybrid_model_v2.py`

> Note: the **entire `archive_unused/` folder** is a strong candidate for removal as a unit (after review).

### 3) Colab/Drive upload duplicate subtree (experimental snapshot)

These files appear to be a packaged snapshot duplicating canonical files under `backend/models/`, `backend/experiments/`, `backend/config/` and are **not referenced** by the runtime API.

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\backend\experiments\CyberSatDetect_Colab_Drive_Upload\` (entire folder)
  - **Reason**: Duplicated code copies under `.../backend/models/*.py` and `.../backend/experiments/*.py`. The runtime uses `backend/app/api.py` and canonical `backend/models/*`, not this subtree.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm -rf backend/experiments/CyberSatDetect_Colab_Drive_Upload`

### 4) Orphan root artifacts (logs/images)

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\eval_run.log`
  - **Reason**: Not referenced by backend/frontend/scripts/config. Looks like a past log artifact.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm eval_run.log`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\hybrid_lstm_gru_autoencoder_architecture.png`
  - **Reason**: Not referenced by backend/frontend/docs. Likely an exported diagram.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm hybrid_lstm_gru_autoencoder_architecture.png`

### 5) Frontend JS files not referenced by any HTML

Checked: no `<script src="spacebg.js">`, `<script src="signup-simple.js">`, and no HTML referencing `server.js`.

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\frontend\spacebg.js`
  - **Reason**: Not referenced by any `frontend/*.html`.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm frontend/spacebg.js`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\frontend\signup-simple.js`
  - **Reason**: Not referenced by any `frontend/*.html` (signup uses `signup.js`).
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm frontend/signup-simple.js`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\frontend\server.js`
  - **Reason**: Node/Express server file appears unrelated to actual deployment (FastAPI serves frontend + API). Not referenced by HTML or backend runtime.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm frontend/server.js`

---

## B) ملفات محتمل أنها غير مستخدمة (تحتاج مراجعة)

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\__init__.py`
  - **Reason**: Empty; not imported by the FastAPI runtime. Might have been added to make repo root a Python package for some tooling.
  - **Confidence**: **Medium**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm __init__.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\scripts\approve_qc_filtered_model.py`
  - **Reason**: Manual helper script (DB registry update). Not called by API routes automatically.
  - **Confidence**: **Medium**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm scripts/approve_qc_filtered_model.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\scripts\print_model_registry.py`
  - **Reason**: Manual diagnostics script, not referenced by runtime.
  - **Confidence**: **Medium**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm scripts/print_model_registry.py`

- **Path**: `C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\scripts\generate_web_smoke_samples.py`
  - **Reason**: Likely dev utility; not referenced by runtime.
  - **Confidence**: **Medium**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm scripts/generate_web_smoke_samples.py`

---

## C) ملفات مكررة أو قديمة (نسخ/تجارب/لقطات)

### 1) Duplicated training/eval scripts under Colab upload snapshot

These are duplicates of canonical files under `backend/models/`:

- **Path**: `...\backend\experiments\CyberSatDetect_Colab_Drive_Upload\backend\models\train_hybrid_model.py`
  - **Reason**: Duplicate of `backend/models/train_hybrid_model.py`.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm backend/experiments/CyberSatDetect_Colab_Drive_Upload/backend/models/train_hybrid_model.py`

- **Path**: `...\backend\experiments\CyberSatDetect_Colab_Drive_Upload\backend\models\evaluate_model_strict_v2.py`
  - **Reason**: Duplicate of `backend/models/evaluate_model_strict_v2.py`.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm backend/experiments/CyberSatDetect_Colab_Drive_Upload/backend/models/evaluate_model_strict_v2.py`

- **Path**: `...\backend\experiments\CyberSatDetect_Colab_Drive_Upload\backend\models\regenerate_attacked_dataset.py`
  - **Reason**: Duplicate of `backend/models/regenerate_attacked_dataset.py`.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete command**:
    - `# rm backend/experiments/CyberSatDetect_Colab_Drive_Upload/backend/models/regenerate_attacked_dataset.py`

### 2) Duplicate virtual environments

- **Paths**: `venv/` and `.venv/`
  - **Reason**: Two copies of local environments.
  - **Confidence**: **High**
  - **Can delete?**: **Review**
  - **Safe delete commands**:
    - `# rm -rf venv`
    - `# rm -rf .venv`

---

## D) ملفات لا يجب حذفها (حتى لو لم تظهر في imports)

### Backend runtime & routes

- **Path**: `...\backend\app\api.py`
  - **Reason**: FastAPI entrypoint, routes, model loading, threshold selection, frontend mount.
  - **Can delete?**: **No**

### Frontend static app served by backend

- **Path**: `...\frontend\*` (HTML/CSS/JS)
  - **Reason**: Served via `/frontend` mount in `backend/app/api.py`.
  - **Can delete?**: **No**, except files explicitly listed in category A after review.

### Model & threshold artifacts (runtime path-based usage)

- **Paths (examples)**:
  - `...\backend\app\best_model_qc_filtered.keras`
  - `...\backend\app\thresholds_qc_filtered.json`
  - `...\backend\app\operating_config.json`
  - `...\backend\app\qc_filtered_best_f1_metrics.json`
  - `...\backend\app\evaluation_strict_v2\*`
  - `...\backend\app\data\app.db`
  - **Reason**: Loaded by path (DB registry / env vars / operating threshold selection / dashboards).
  - **Can delete?**: **No**

### Docker & DB init

- **Paths**:
  - `...\docker\docker-compose.yml`
  - `...\db\init\01-create-dbs-and-users.sql`
  - `...\db\init\02-schemas.sql`
  - **Reason**: Required for containerized DB setup.
  - **Can delete?**: **No**

### Data / docs / samples

- **Paths**: `data/`, `docs/`, `samples/`
  - **Reason**: Even if not imported, they’re inputs/outputs/reproducibility assets.
  - **Can delete?**: **No** (review selectively if storage is an issue)

---

## Additional findings (not “unused”, but important)

- **Case-sensitivity issue risk**:
  - `frontend/index.html` references `styles.css`, but the file present is `frontend/Styles.css`.
  - Works on Windows, may break on Linux/macOS deployments with case-sensitive FS.
  - **Action**: rename `Styles.css` → `styles.css` (or update HTML).

- **Possible missing asset (not unused)**:
  - `backend/app/assets/` contains only `README.txt`.
  - If reporting expects a logo image (common pattern), that logo might be missing rather than unused.

---

## Summary

- **Confirmed unused (A)**: **13** items (including whole folders like `venv/`, `.venv/`, `archive_unused/`, colab upload subtree, and 3 frontend JS + 2 root artifacts).
- **Potential unused (B)**: **4** items (mostly manual scripts and empty `__init__.py`).
- **Duplicates/old (C)**: **3** main groups (venvs, archive snapshot, colab upload duplicates).
- **Safest to delete first** (after review):
  - `frontend/spacebg.js`, `frontend/signup-simple.js`, `frontend/server.js`
  - `eval_run.log`, `hybrid_lstm_gru_autoencoder_architecture.png`
- **Needs your review before deletion**:
  - `archive_unused/` (if you want to keep historical code)
  - `backend/experiments/CyberSatDetect_Colab_Drive_Upload/` (if you still rely on it for colab packaging)
  - `venv/` and `.venv/` (if you intentionally committed them for reproducible offline installs)

