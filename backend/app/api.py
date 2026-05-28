# ============================================================
# Part 1/3: Imports + Config + Continual + DB + Auth
# ============================================================

from pathlib import Path as _Path
from dotenv import load_dotenv
load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

import os
import json
import csv
import uuid
import sqlite3
import hashlib
import shutil
import threading
import time
import traceback
import random
import secrets
import smtplib
import resend
import requests
import html as html_mod
import subprocess

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import bcrypt
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Literal

# Continual learning modules are optional. They may not be present in minimal installs.
try:
    from backend.continual.dataset_builder import build_dataset
    from backend.continual.buffer_manager import process_and_store
except Exception as e:
    print("[CONTINUAL IMPORT ERROR]", e)
    build_dataset = None
    process_and_store = None

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
    Depends,
    Request,
    BackgroundTasks,
    Security,
    Query,
)

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel, EmailStr

import jwt

from backend.app.reporting import build_report_json, read_anomaly_table_rows, write_excel, write_pdf

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ---- optional libs ----
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # Use tf.keras only — standalone `keras` can mismatch TF's loader/deserializer.
    keras = tf.keras
except Exception:
    tf = None
    keras = None
    load_model = None

# =========================
# Continual Learning (admin-only)
# =========================
# =========================
# Continual Learning (admin-only)
# =========================
CONTINUAL_AVAILABLE = True

try:
    from backend.continual.dataset_builder import build_dataset
    from backend.continual.train_continual import fine_tune
    from backend.continual.config import DATASETS_DIR

except Exception as e:
    print("[CONTINUAL ADMIN IMPORT ERROR]", e)
    CONTINUAL_AVAILABLE = False
# =========================
# App / Env
# =========================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path("/var/data")
UPLOADS_DIR = DATA_DIR / "uploads"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Continual Learning Paths
# =========================
CONTINUAL_DIR = DATA_DIR / "continual"
CONTINUAL_NORMAL_POOL_DIR = CONTINUAL_DIR / "normal_pool"
CONTINUAL_DATASET_DIR = CONTINUAL_DIR / "datasets"
CONTINUAL_ANOMALY_POOL_DIR = CONTINUAL_DIR / "anomaly_pool"
MODEL_REGISTRY_DIR = CONTINUAL_DIR / "models"

# create folders if not exist
CONTINUAL_DIR.mkdir(exist_ok=True)
CONTINUAL_NORMAL_POOL_DIR.mkdir(exist_ok=True)
CONTINUAL_DATASET_DIR.mkdir(exist_ok=True)
CONTINUAL_ANOMALY_POOL_DIR.mkdir(exist_ok=True)
MODEL_REGISTRY_DIR.mkdir(exist_ok=True)


DB_PATH = DATA_DIR / "app.db"

# Canonical inference artifacts next to this module (backend/app/...)
# Default production stack: QC-filtered model + matching thresholds (see thresholds_qc_filtered.json).
BUNDLED_MODEL_PATH = (BASE_DIR / "best_model_render.keras").resolve()
BUNDLED_THRESH_PATH = (BASE_DIR / "thresholds_qc_filtered.json").resolve()

# Override with CSD_MODEL_PATH / CSD_THRESH_PATH when CSD_INFERENCE_USE_ENV_WITHOUT_REGISTRY=1
# and registry does not yield a usable model row (see get_active_model_paths).
MODEL_PATH = Path(os.getenv("CSD_MODEL_PATH", str(BUNDLED_MODEL_PATH)))
THRESH_PATH = Path(os.getenv("CSD_THRESH_PATH", str(BUNDLED_THRESH_PATH)))
_INFERENCE_USE_ENV_WITHOUT_REGISTRY = os.getenv(
    "CSD_INFERENCE_USE_ENV_WITHOUT_REGISTRY", ""
).strip().lower() in ("1", "true", "yes", "on")
OPERATING_CONFIG_PATH = Path(os.getenv("CSD_OPERATING_CONFIG", str(BASE_DIR / "operating_config.json")))
STRICT_EVAL_SUMMARY_PATH = Path(
    os.getenv("CSD_STRICT_EVAL_SUMMARY", str(BASE_DIR / "evaluation_strict_v2" / "evaluation_summary.json"))
)

# Optional precomputed QC-filtered evaluation (used for qc_filtered_best model dashboards)
QC_EVAL_REGEN_ATTACKS_PATH = (BASE_DIR.parents[1] / "backend" / "experiments" / "qc_filtered_evaluation_regen_attacks" / "results" / "evaluation_qc_filtered.json").resolve()
QC_EVAL_BASE_ATTACKS_PATH = (BASE_DIR.parents[1] / "backend" / "experiments" / "qc_filtered_evaluation" / "results" / "evaluation_qc_filtered.json").resolve()
QC_BEST_F1_METRICS_PATH = (BASE_DIR / "qc_filtered_best_f1_metrics.json").resolve()

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
if JWT_SECRET == "change-me":
    import warnings
    warnings.warn(
        "\n⚠️  WARNING: JWT_SECRET is using the insecure default value 'change-me'.\n"
        "   Set a strong secret via: export JWT_SECRET=<your-random-secret>\n"
        "   Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"\n",
        stacklevel=2,
    )
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Model input confirmed: (None, 100, 1)
WINDOW_LEN = int(os.getenv("CSD_WINDOW_LEN", "100"))
STRIDE = int(os.getenv("CSD_STRIDE", "50"))

# Threshold selection:
# - Prefer explicit threshold value via env or operating_config.json
# - Else fall back to the active thresholds dict (p99 / p995 / p997 / 3sigma / best_f1) from the thresholds file
DEFAULT_THRESHOLD_KEY = os.getenv("CSD_THRESHOLD_KEY", "p99")
DEFAULT_THRESHOLD_FALLBACK = float(os.getenv("CSD_THRESHOLD_FALLBACK", "0.1"))
DEFAULT_THRESHOLD_VALUE = os.getenv("CSD_THRESHOLD_VALUE")  # optional explicit numeric string

EPS = 1e-8

# =========================
# Web pages / static files
# =========================
WEB_DIR = BASE_DIR
# Repo layout in Docker: /app/frontend and /app/backend/app/api.py
FRONTEND_DIR = BASE_DIR.parent.parent / "frontend"
ADMIN_CONTINUAL_HTML = FRONTEND_DIR / "admin-cl.html"
ADMIN_CONTINUAL_CSS = FRONTEND_DIR / "admin-cl.css"
ADMIN_CONTINUAL_JS = FRONTEND_DIR / "admin-cl.js"

# =========================
# Reports storage
# =========================
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_PDF_DIR = REPORTS_DIR / "pdf"
REPORTS_XLSX_DIR = REPORTS_DIR / "excel"
REPORTS_PDF_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_XLSX_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# App
# =========================
app = FastAPI(title="CyberSatDetect API (Inference-Only, Per-Channel)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Security Headers Middleware
# =========================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https://i.ibb.co; "
            "connect-src 'self' http://127.0.0.1:8000 http://127.0.0.1:8001 http://localhost:8000 http://localhost:8001 https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)


class NoStoreFrontendMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevent stale frontend assets during development/local runs.
        if request.url.path.startswith("/frontend"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(NoStoreFrontendMiddleware)

# Frontend is served from this API under /frontend, so no special CORS is required.

# =========================
# Server-side Text Sanitization
# =========================
def sanitize_text(text: str) -> str:
    """Escape HTML entities to prevent XSS in any user-supplied text."""
    if not isinstance(text, str):
        return text
    return html_mod.escape(text, quote=True)

# 🔒 /static mount removed — was exposing backend/app/ (uploads, DB, model) without auth.
# Frontend is served via /frontend mount below. Uploaded files served via authenticated API only.

# Serve frontend pages/assets so routes like /frontend/otp-verify.html are available.
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def root():
    # Main entrypoint for hosted deployments.
    return RedirectResponse(url="/frontend/index.html")

# =========================
# Rate Limiting
# =========================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# =========================
# Auto Backup Loop
# =========================
def auto_backup_loop():
    while True:
        try:
            # Avoid unicode emoji to prevent Windows console encoding crashes.
            print("[backup] Running auto backup...")
            create_backup()
        except Exception as e:
            print("[backup] Backup failed:", e)

        time.sleep(3600)  # كل ساعة

# =========================
# Start Auto Backup
# =========================
@app.on_event("startup")
def start_auto_backup():
    thread = threading.Thread(
        target=auto_backup_loop,
        daemon=True
    )
    thread.start()
# =========================
# Continual training status
# =========================
TRAINING_STATUS = {
    "running": False,
    "stage": "idle",
    "message": "",
    "dataset_path": None,
    "version": None,
    "last_model_path": None,
    "last_threshold_path": None,
    "last_error": None,
    "last_started_at": None,
    "last_finished_at": None,
}


def select_largest_chunk_dataset() -> Path:
    """
    Pick the largest training source under backend/data:
    - chunks/*.npy
    - continual/datasets/*.npy
    - continual/anomaly_datasets/*.npy
    - combined.npy (legacy)
    - combined_dataset.npz (new: normals+anomalies)
    - combined_normal.npy (new: normals only)
    Aligns paths with continual.config (backend/data/...).
    """
    data_root = BASE_DIR.parent / "data"
    candidates: List[Path] = []
    chunks_dir = data_root / "chunks"
    if chunks_dir.is_dir():
        candidates.extend(chunks_dir.glob("*.npy"))
    datasets_dir = data_root / "continual" / "datasets"
    if datasets_dir.is_dir():
        candidates.extend(datasets_dir.glob("*.npy"))
    anomaly_datasets_dir = data_root / "continual" / "anomaly_datasets"
    if anomaly_datasets_dir.is_dir():
        candidates.extend(anomaly_datasets_dir.glob("*.npy"))
    combined = data_root / "combined.npy"
    if combined.is_file():
        candidates.append(combined)
    combined_npz = data_root / "combined_dataset.npz"
    if combined_npz.is_file():
        candidates.append(combined_npz)
    combined_norm = data_root / "combined_normal.npy"
    if combined_norm.is_file():
        candidates.append(combined_norm)
    if not candidates:
        return combined_npz
    return max(candidates, key=lambda p: p.stat().st_size)


def resolve_continual_training_dataset_path() -> Path:
    """
    Dataset for continual fine-tuning must match the admin workflow: approved pools → Build Dataset
    → ``backend/data/combined_dataset.npz``. Do *not* default to the largest raw chunk, which can be
    stale or unrelated to the newly approved normal/anomaly shards.

    Resolution order:
      1) Path stored in TRAINING_STATUS after the last successful ``/admin/continual/build-dataset`` call.
      2) ``backend/data/combined_dataset.npz`` if present (persisted build artefact).
      3) Legacy fallback: ``select_largest_chunk_dataset()`` (largest candidate under backend/data).
    """
    cached = (TRAINING_STATUS.get("dataset_path") or "").strip()
    if cached:
        p = Path(cached)
        if p.is_file():
            return p.resolve()

    combined_npz = (BASE_DIR.parent / "data" / "combined_dataset.npz").resolve()
    if combined_npz.is_file():
        return combined_npz

    return select_largest_chunk_dataset()


# =========================
# SQLite bootstrap
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def create_incident(user_id, ip, type_, details, action_taken="logged"):
    try:
        # Console alert (avoid emoji for Windows encodings)
        print(f"[ALERT] {type_} | user={user_id} | ip={ip}")

        # Sanitize user-supplied text to prevent stored XSS
        safe_details = sanitize_text(details) if details else details

        with db() as conn:
            conn.execute(
                """
                INSERT INTO incidents 
                (id, created_at, user_id, ip, type, action_taken, details, status)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    uuid.uuid4().hex,
                    datetime.utcnow().isoformat(),
                    str(user_id),
                    ip,
                    type_,
                    action_taken,
                    safe_details,
                    "OPEN"
                ),
            )
    except Exception as e:
        print("Incident logging failed:", e)
def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'USER',
            is_blocked INTEGER DEFAULT 0,
            failed_attempts INTEGER DEFAULT 0,
            locked_until TEXT,
            created_at TEXT NOT NULL
        )
        """)

        # OTP Codes table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS otp_codes (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)

        # Incidents table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            user_id TEXT,
            ip TEXT,
            type TEXT NOT NULL,
            action_taken TEXT,
            details TEXT,
            status TEXT DEFAULT 'OPEN'
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT,
            file_type TEXT,
            file_sha256 TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_results (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            channel_name TEXT NOT NULL,
            num_windows INTEGER NOT NULL,
            num_anomalies INTEGER NOT NULL,
            anomaly_rate REAL NOT NULL,
            threshold REAL NOT NULL,
            results_path TEXT NOT NULL
        )
        """)

        # Model Registry for continual versions
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            model_path TEXT NOT NULL,
            threshold_path TEXT NOT NULL,
            dataset_path TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            approved_at TEXT,
            approved_by TEXT
        )
        """)

        # Reports (mission + cybersecurity)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            report_id TEXT UNIQUE NOT NULL,
            run_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            severity_summary TEXT,
            total_windows INTEGER,
            normal_count INTEGER,
            anomaly_count INTEGER,
            low_count INTEGER,
            medium_count INTEGER,
            high_count INTEGER,
            threshold_used REAL,
            model_version TEXT,
            pdf_path TEXT,
            excel_path TEXT,
            report_json TEXT
        )
        """)

        # Backward-compatible migration: add accuracy if DB was created before this field existed.
        model_cols = conn.execute("PRAGMA table_info(model_registry)").fetchall()
        if "accuracy" not in {c[1] for c in model_cols}:
            conn.execute("ALTER TABLE model_registry ADD COLUMN accuracy REAL")

        # Backward-compatible migration: add lockout columns to users table.
        user_cols = conn.execute("PRAGMA table_info(users)").fetchall()
        user_col_names = {c[1] for c in user_cols}
        if "failed_attempts" not in user_col_names:
            conn.execute("ALTER TABLE users ADD COLUMN failed_attempts INTEGER DEFAULT 0")
        if "locked_until" not in user_col_names:
            conn.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")


init_db()


def generate_otp():
    return str(secrets.randbelow(900000) + 100000)


# =========================
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _is_legacy_sha256(hashed: str) -> bool:
    """Detect old SHA-256 hex hashes (64 hex chars, no '$' prefix)."""
    return len(hashed) == 64 and not hashed.startswith("$") and all(c in "0123456789abcdef" for c in hashed)


def verify_password(password: str, hashed: str) -> bool:
    # Backward compatibility: support old SHA-256 hashes
    if _is_legacy_sha256(hashed):
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == hashed
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def _migrate_password_if_needed(user_id: str, password: str, hashed: str):
    """Re-hash legacy SHA-256 passwords to bcrypt on successful login."""
    if _is_legacy_sha256(hashed):
        new_hash = hash_password(password)
        with db() as conn:
            conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id))
        print(f"[auth] Migrated password hash to bcrypt for user: {user_id}")


def create_jwt(user_id: str, role: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_jwt(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    return decode_jwt(token)


def require_admin(user=Depends(get_current_user)):
    if user.get("role") != "ADMIN":
        raise HTTPException(403, "Admin privileges required")
    return user


def _resolve_under(base_dir: Path, p: Path) -> Path:
    base = base_dir.resolve()
    rp = p.resolve()
    try:
        if rp.is_relative_to(base):
            return rp
    except Exception:
        pass
    # Fallback for older behavior
    if str(rp).lower().startswith(str(base).lower()):
        return rp
    raise HTTPException(400, "Invalid file path")
def send_otp_email(to_email, otp):

    brevo_api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM")

    response = requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "accept": "application/json",
            "api-key": brevo_api_key,
            "content-type": "application/json",
        },
        json={
            "sender": {
                "name": "CyberSatDetect",
                "email": smtp_from
            },
            "to": [
                {"email": to_email}
            ],
            "subject": "CyberSatDetect OTP",
            "htmlContent": f"""
            <h2>Your Verification Code</h2>
            <p>Your OTP is:</p>
            <h1>{otp}</h1>
            """
        },
        timeout=20,
    )

    print("[email] Brevo status:", response.status_code)
    print("[email] Brevo response:", response.text)

    if response.status_code not in (200, 201, 202):
        raise Exception("Failed to send OTP email")


def _admin_notify_recipient_list() -> List[str]:
    raw = (os.getenv("CSD_ADMIN_NOTIFY_EMAILS") or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]

def send_admin_notification(subject: str, body: str) -> None:
    recipients = _admin_notify_recipient_list()
    if not recipients:
        return

    brevo_api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "cyberproject2026@gmail.com")

    if not brevo_api_key:
        print("[admin-notify] Skipped (no BREVO_API_KEY)")
        return

    safe_subject = (subject or "notice").replace("\n", " ").replace("\r", " ")[:200]

    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "accept": "application/json",
                "api-key": brevo_api_key,
                "content-type": "application/json",
            },
            json={
                "sender": {
                    "name": "CyberSatDetect Admin",
                    "email": smtp_from
                },
                "to": [{"email": email} for email in recipients],
                "subject": f"[CyberSatDetect Admin] {safe_subject}",
                "htmlContent": f"<pre>{body or ''}</pre>",
            },
            timeout=20,
        )

        print("[admin-notify] Brevo status:", response.status_code)
        print("[admin-notify] Brevo response:", response.text)

    except Exception as e:
        print(f"[admin-notify] Failed ({safe_subject}): {e}")


# =========================
# Schemas
# =========================
class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ============================================================
# Part 2/3: Model Registry + Dynamic Loading + Inference Utilities
# ============================================================

_MODEL = None
_THRESH = None
_STRICT_METRICS_CACHE: Dict[str, Any] = {}

# strict-eval runtime status (best-effort)
STRICT_EVAL_STATUS: Dict[str, Any] = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_exit_code": None,
    "last_error": None,
    "last_log_path": None,
}


def resolve_artifact_path(p: Path) -> Path:
    """
    Registry rows often store absolute paths from another machine. If the file is missing,
    try the same filename under backend/app, backend/, or continual model dirs.
    """
    p = Path(p).expanduser()
    try:
        if p.is_file():
            return p.resolve()
    except OSError:
        pass
    name = p.name
    for base in (BASE_DIR, BASE_DIR.parent, MODEL_REGISTRY_DIR):
        try:
            cand = (Path(base) / name).resolve()
            if cand.is_file():
                return cand
        except OSError:
            continue
    return p


def get_active_model_paths() -> Tuple[Path, Path]:
    """
    If there is an APPROVED model in model_registry -> use it.
    Else if the newest registry row is PENDING and its artefact files exist -> use that
    (optional pre-approval smoke path).
    Else -> fall back to bundled ``best_model_qc_filtered.keras`` + ``thresholds_qc_filtered.json``
    (rollback / default inference when nothing approved and no pending candidate),
    unless env ``CSD_INFERENCE_USE_ENV_WITHOUT_REGISTRY=1`` then ``CSD_MODEL_PATH`` / ``CSD_THRESH_PATH``.

    Important: REJECTED rows are never auto-selected. After rollback (all REJECTED), inference
    must return to the bundled baseline, not the latest continual checkpoint.
    """
    with db() as conn:
        row = conn.execute("""
        SELECT model_path, threshold_path
        FROM model_registry
        WHERE status='APPROVED'
        ORDER BY approved_at DESC, created_at DESC
        LIMIT 1
        """).fetchone()
        latest = conn.execute(
            """
            SELECT model_path, threshold_path, status
            FROM model_registry
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if row:
        mp = resolve_artifact_path(Path(row["model_path"]))
        tp = resolve_artifact_path(Path(row["threshold_path"]))
    elif latest and str(latest["status"]).upper() == "PENDING":
        mp_cand = resolve_artifact_path(Path(latest["model_path"]))
        tp_cand = resolve_artifact_path(Path(latest["threshold_path"]))
        if mp_cand.exists() and tp_cand.exists():
            mp, tp = mp_cand, tp_cand
        elif _INFERENCE_USE_ENV_WITHOUT_REGISTRY:
            mp = resolve_artifact_path(MODEL_PATH)
            tp = resolve_artifact_path(THRESH_PATH)
        else:
            mp = resolve_artifact_path(BUNDLED_MODEL_PATH)
            tp = resolve_artifact_path(BUNDLED_THRESH_PATH)
    elif _INFERENCE_USE_ENV_WITHOUT_REGISTRY:
        mp = resolve_artifact_path(MODEL_PATH)
        tp = resolve_artifact_path(THRESH_PATH)
    else:
        mp = resolve_artifact_path(BUNDLED_MODEL_PATH)
        tp = resolve_artifact_path(BUNDLED_THRESH_PATH)

    return mp, tp


def resolve_model_version_for_path(model_path: Path) -> str:
    """
    Registry version label for the model weights used at inference time.
    Matches APPROVED rows by resolved filesystem path.
    """
    try:
        target = model_path.resolve()
    except OSError:
        target = model_path
    target_s = str(target)
    with db() as conn:
        rows = conn.execute(
            """
            SELECT version, model_path FROM model_registry
            WHERE status='APPROVED'
            ORDER BY approved_at DESC, created_at DESC
            """
        ).fetchall()
    for row in rows:
        try:
            rp = resolve_artifact_path(Path(row["model_path"]))
            if str(rp.resolve()) == target_s:
                return str(row["version"])
        except Exception:
            continue
    return model_path.stem


def get_latest_model_row(prefer_approved: bool = True) -> Optional[Dict[str, Any]]:
    """
    Returns the latest model_registry row.
    - If prefer_approved=True: return latest APPROVED if exists, else latest row by created_at.
    """
    with db() as conn:
        if prefer_approved:
            row = conn.execute(
                """
                SELECT version, model_path, threshold_path, status, created_at, approved_at, accuracy
                FROM model_registry
                WHERE status='APPROVED'
                ORDER BY approved_at DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                return dict(row)

        row = conn.execute(
            """
            SELECT version, model_path, threshold_path, status, created_at, approved_at, accuracy
            FROM model_registry
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None


def load_thresholds_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(500, f"Thresholds file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_operating_config() -> Dict[str, Any]:
    """
    Optional runtime config for selecting an operating threshold and showing metrics.
    Does not affect training and does not modify the model.
    """
    try:
        if OPERATING_CONFIG_PATH.exists():
            return json.loads(OPERATING_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print("Failed to load operating_config.json:", e)
    return {}


def _strict_metrics_for_threshold(threshold: float) -> Optional[Dict[str, Any]]:
    """
    Compute metrics for a given threshold using strict_v2 window scores if available.
    Results are cached in-memory per threshold string.
    """
    try:
        key = f"{float(threshold):.10f}"
        if key in _STRICT_METRICS_CACHE:
            return _STRICT_METRICS_CACHE[key]

        scores_csv = STRICT_EVAL_SUMMARY_PATH.parent / "window_scores.csv"
        if not scores_csv.exists():
            return None

        import csv as _csv

        TP = TN = FP = FN = 0
        thr = float(threshold)
        with open(scores_csv, "r", encoding="utf-8", newline="") as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    y = int(float(row.get("y_true", 0)))
                    sc = float(row.get("score", 0.0))
                except Exception:
                    continue
                pred = 1 if sc > thr else 0
                if pred == 1 and y == 1:
                    TP += 1
                elif pred == 0 and y == 0:
                    TN += 1
                elif pred == 1 and y == 0:
                    FP += 1
                else:
                    FN += 1

        total = TP + TN + FP + FN
        if total <= 0:
            return None

        tpr = TP / max(1, TP + FN)
        tnr = TN / max(1, TN + FP)
        acc = (TP + TN) / total
        bal = 0.5 * (tpr + tnr)
        precision = TP / max(1, TP + FP)
        recall = tpr
        f1 = (2 * precision * recall) / max(1e-12, (precision + recall))
        far = 1.0 - tnr
        fnr = 1.0 - tpr

        out = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "precision": precision,
            "recall": recall,
            "tnr": tnr,
            "tpr": tpr,
            "f1": f1,
            "far": far,
            "fnr": fnr,
        }
        _STRICT_METRICS_CACHE[key] = out
        return out
    except Exception:
        return None


def _run_strict_eval_v2_job(model_path: Path, window_len: int, stride: int) -> None:
    """
    Run backend/models/evaluate_model_strict_v2.py as a subprocess and write artifacts into
    backend/app/evaluation_strict_v2/ (same location used by dashboard endpoints).
    """
    global _STRICT_METRICS_CACHE

    STRICT_EVAL_STATUS["running"] = True
    STRICT_EVAL_STATUS["last_started_at"] = datetime.utcnow().isoformat()
    STRICT_EVAL_STATUS["last_finished_at"] = None
    STRICT_EVAL_STATUS["last_exit_code"] = None
    STRICT_EVAL_STATUS["last_error"] = None

    out_dir = (BASE_DIR / "evaluation_strict_v2").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (out_dir / f"strict_eval_v2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log").resolve()
    STRICT_EVAL_STATUS["last_log_path"] = str(log_path)

    root = BASE_DIR.parents[1]  # repo root (.. / .. from backend/app)
    script = (root / "backend" / "models" / "evaluate_model_strict_v2.py").resolve()

    # Default dataset locations (repo-root relative)
    normal_dir = (root / "data" / "reduced").resolve()
    attacked_dir = (root / "data" / "attacked_v2").resolve()
    split_json = (root / "backend" / "config" / "data_split.json").resolve()

    cmd = [
        os.environ.get("PYTHON", "python"),
        str(script),
        "--model",
        str(model_path),
        "--normal-dir",
        str(normal_dir),
        "--attacked-dir",
        str(attacked_dir),
        "--split-json",
        str(split_json),
        "--output-dir",
        str(out_dir),
        "--window-size",
        str(int(window_len)),
        "--stride",
        str(int(stride)),
    ]

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=str(root), stdout=f, stderr=subprocess.STDOUT, check=False)
        STRICT_EVAL_STATUS["last_exit_code"] = int(getattr(p, "returncode", -1))
        if STRICT_EVAL_STATUS["last_exit_code"] != 0:
            STRICT_EVAL_STATUS["last_error"] = f"strict eval exited with code {STRICT_EVAL_STATUS['last_exit_code']}"
    except Exception as e:
        STRICT_EVAL_STATUS["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        STRICT_EVAL_STATUS["running"] = False
        STRICT_EVAL_STATUS["last_finished_at"] = datetime.utcnow().isoformat()
        # Invalidate metrics cache (window_scores.csv likely changed)
        _STRICT_METRICS_CACHE = {}


def get_operating_threshold_value(thresholds_cfg: Dict[str, Any]) -> Tuple[float, str]:
    """
    Returns (threshold_value, source_label).
    Priority:
      1) CSD_THRESHOLD_VALUE env (explicit numeric)
      2) backend/app/operating_config.json operating_threshold.value
      3) thresholds.json thresholds[DEFAULT_THRESHOLD_KEY]
      4) thresholds.json thresholds['p99'] if present (compat)
      5) thresholds.json thresholds['best_f1'] if present (compat)
      6) DEFAULT_THRESHOLD_FALLBACK
    """
    # 1) explicit env value
    if DEFAULT_THRESHOLD_VALUE:
        try:
            return float(DEFAULT_THRESHOLD_VALUE), "env:CSD_THRESHOLD_VALUE"
        except Exception:
            pass

    # 2) operating config file
    ocfg = load_operating_config()
    try:
        ot = ocfg.get("operating_threshold", {})
        v = ot.get("value")
        if v is not None:
            return float(v), "operating_config.json"
    except Exception:
        pass

    # 3/4) thresholds.json map
    tmap = thresholds_cfg.get("thresholds", {})
    if isinstance(tmap, dict):
        if DEFAULT_THRESHOLD_KEY in tmap:
            try:
                return float(tmap[DEFAULT_THRESHOLD_KEY]), f"thresholds.json:{DEFAULT_THRESHOLD_KEY}"
            except Exception:
                pass
        if "p99" in tmap:
            try:
                return float(tmap["p99"]), "thresholds.json:p99"
            except Exception:
                pass
        if "best_f1" in tmap:
            try:
                return float(tmap["best_f1"]), "thresholds.json:best_f1"
            except Exception:
                pass

    # 6) fallback
    return float(DEFAULT_THRESHOLD_FALLBACK), "fallback"


def pick_threshold(cfg: Dict[str, Any]) -> float:
    v, _src = get_operating_threshold_value(cfg)
    return float(v)


_KERAS_LOAD_PATCH_DONE = False


def _ensure_keras_dense_quant_compat() -> None:
    """
    Newer Keras saves Dense layers with quantization_config=None; some runtime versions
    reject that kwarg. Strip it before delegating to the original from_config.
    Uses tensorflow.keras only (no standalone keras package).
    """
    global _KERAS_LOAD_PATCH_DONE
    if _KERAS_LOAD_PATCH_DONE:
        return
    try:
        from tensorflow.keras.layers import Dense

        orig = Dense.from_config.__func__

        @classmethod
        def from_config(cls, config):  # noqa: N805
            if isinstance(config, dict) and "quantization_config" in config:
                config = dict(config)
                config.pop("quantization_config", None)
            return orig(cls, config)

        Dense.from_config = from_config
        _KERAS_LOAD_PATCH_DONE = True
    except Exception:
        pass


def load_inference_model():
    """
    Loads model + thresholds dynamically from APPROVED registry if exists.
    Cache in globals. Re-load occurs after approve endpoint sets globals to None.
    """
    global _MODEL, _THRESH

    if _MODEL is None or _THRESH is None:
        if load_model is None or keras is None:
            raise HTTPException(500, "TensorFlow/Keras not installed")

        model_path, thresh_path = get_active_model_paths()

        if not model_path.exists():
            raise HTTPException(500, f"Model file not found: {model_path}")

        _ensure_keras_dense_quant_compat()

        # Keras 3: optional; standalone Keras 2 / some envs have no keras.config
        kconf = getattr(keras, "config", None)
        if kconf is not None and hasattr(kconf, "enable_unsafe_deserialization"):
            try:
                kconf.enable_unsafe_deserialization()
            except Exception:
                pass

        try:
            try:
                _MODEL = load_model(str(model_path), compile=False, safe_mode=False)
            except TypeError:
                _MODEL = load_model(str(model_path), compile=False)
        except Exception as e:
            raise HTTPException(
                500,
                f"Failed to load Keras model from {model_path}. "
                f"Use Python 3.10+ with compatible TensorFlow/Keras 3, or set CSD_MODEL_PATH. Error: {e}",
            ) from e

        shp = getattr(_MODEL, "input_shape", None)
        if not (isinstance(shp, tuple) and len(shp) == 3 and shp[1] == WINDOW_LEN and shp[2] == 1):
            raise HTTPException(
                500,
                f"Model input_shape mismatch. Expected (None,{WINDOW_LEN},1) but got {shp}",
            )

        _THRESH = load_thresholds_from_path(thresh_path)

    return _MODEL


def load_thresholds() -> Dict[str, Any]:
    """
    Keeps original API contract: returns cached thresholds of ACTIVE model.
    """
    global _THRESH
    if _THRESH is None:
        _, thresh_path = get_active_model_paths()
        _THRESH = load_thresholds_from_path(thresh_path)
    return _THRESH


def load_active_thresholds_cfg() -> Dict[str, Any]:
    """
    Load thresholds.json for the ACTIVE inference model (approved registry or bundled fallback).
    Falls back to THRESH_PATH env default if needed.
    """
    try:
        _, tp = get_active_model_paths()
        if tp.exists():
            return load_thresholds_from_path(tp)
    except Exception:
        pass
    try:
        if THRESH_PATH.exists():
            return load_thresholds_from_path(THRESH_PATH)
    except Exception:
        pass
    return {}


def load_dashboard_evaluation_summary() -> Optional[Dict[str, Any]]:
    """
    Prefer QC-filtered precomputed evaluation when ACTIVE thresholds are thresholds_qc_filtered.json.
    Otherwise fall back to strict_v2 evaluation_summary.json.
    """
    try:
        _mp, tp = get_active_model_paths()
        if tp.name.lower() == "thresholds_qc_filtered.json":
            if QC_BEST_F1_METRICS_PATH.exists():
                return json.loads(QC_BEST_F1_METRICS_PATH.read_text(encoding="utf-8"))
            for p in (QC_EVAL_REGEN_ATTACKS_PATH, QC_EVAL_BASE_ATTACKS_PATH):
                if p.exists():
                    return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass

    try:
        if STRICT_EVAL_SUMMARY_PATH.exists():
            return json.loads(STRICT_EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)


def create_backup():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # backup database
    db_src = Path("app/data/app.db")
    db_dst = BACKUP_DIR / f"db_backup_{timestamp}.db"
    shutil.copy(db_src, db_dst)

    # backup runs folder
    runs_src = Path("runs")
    runs_dst = BACKUP_DIR / f"runs_backup_{timestamp}"
    if runs_src.exists():
        shutil.copytree(runs_src, runs_dst)

    return str(db_dst)

# =========================
# Runs storage helpers
# =========================
def run_dir(run_id: str) -> Path:
    d = UPLOADS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def ensure_run_owner(run_id: str, user_sub: str) -> sqlite3.Row:
    with db() as conn:
        r = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Run not found")
    if str(r["user_id"]) != str(user_sub):
        raise HTTPException(403, "Forbidden")
    return r


def set_run_status(run_id: str, status: str):
    with db() as conn:
        conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))


def optional_process_and_store(
    series_1d: "np.ndarray",
    scores: "np.ndarray",
    threshold: float,
    spans: List[Tuple[int, int]],
) -> None:
    """Continual-learning buffer; never fail the main analyze pipeline."""
    if process_and_store is None:
        return
    try:
        process_and_store(
            series_1d=series_1d,
            scores=scores,
            threshold=threshold,
            spans=spans,
        )
    except Exception as e:
        print(f"[optional_process_and_store] skipped: {type(e).__name__}: {e}")


def recover_run_if_analyze_finished(run_id: str) -> None:
    """
    If analyze crashed after writing channel_results but before status=DONE,
    polling would stay on ANALYZING forever — promote to DONE when DB shows results.
    """
    try:
        with db() as conn:
            row = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not row or str(row["status"]) != "ANALYZING":
                return
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM channel_results WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if not cnt or int(cnt["c"]) < 1:
                return
        set_run_status(run_id, "DONE")
    except Exception as e:
        print(f"[recover_run_if_analyze_finished] {run_id}: {e}")


# =========================
# Cleaning (Inference-only)
# =========================
def clean_series(x: "np.ndarray") -> "np.ndarray":
    """
    Inference cleaning:
    - float32
    - replace inf -> nan
    - interpolate (linear) then fill remaining with 0
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x[~np.isfinite(x)] = np.nan

    n = x.size
    idx = np.arange(n, dtype=np.float32)
    mask = np.isfinite(x)

    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    x_interp = x.copy()
    x_interp[~mask] = np.interp(idx[~mask], idx[mask], x[mask]).astype(np.float32)
    x_interp[~np.isfinite(x_interp)] = 0.0
    return x_interp.astype(np.float32)


def make_windows_1ch(
    x: "np.ndarray",
    win_len: int,
    stride: int,
) -> Tuple["np.ndarray", List[Tuple[int, int]]]:
    """
    x: (N,)
    return:
      Xw: (num_windows, win_len, 1)
      spans: list of (start,end)
    """
    n = x.size
    if n < win_len:
        return np.zeros((0, win_len, 1), dtype=np.float32), []

    spans: List[Tuple[int, int]] = []
    windows: List["np.ndarray"] = []

    for s in range(0, n - win_len + 1, stride):
        e = s + win_len
        windows.append(x[s:e])
        spans.append((s, e))

    Xw = np.asarray(windows, dtype=np.float32)[..., None]
    return Xw, spans


def _finite_float32_tensor(name: str, arr: "np.ndarray") -> "np.ndarray":
    """Model output validation: ensure float32 array has no NaN/Inf (same fallback as clean_series)."""
    out = np.asarray(arr, dtype=np.float32)
    if out.size == 0 or np.isfinite(out).all():
        return out
    print(f"[inference] Non-finite values in {name}; replacing with 0.0")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================
# Hybrid score (same as training)
# =========================
def compute_scores_hybrid(model, X: "np.ndarray", cfg: Dict[str, Any]) -> "np.ndarray":
    """
    X: (B, T, 1)
    recon: (B, T, 1)
    pred: (B, T-1, 1) sequence head or (B, 1) / (B, 1, C) single-step vs last timestep
        (same convention as evaluate_model_strict_v2.compute_scores_strict).
    """
    recon, pred = model.predict(X, verbose=0)
    recon = _finite_float32_tensor("recon", recon)
    pred = _finite_float32_tensor("pred", pred)

    w = cfg.get("weights", {})
    if not isinstance(w, dict):
        w = {}

    W_RECON = float(w.get("W_RECON", 1.0))
    W_PRED = float(w.get("W_PRED", 2.0))
    W_GRAD = float(w.get("W_GRAD", 2.0))

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))

    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))

    T = X.shape[1]
    if pred.ndim == 3 and pred.shape[1] == T - 1:
        y_true = X[:, 1:, :]
        e_pred = np.mean((y_true - pred) ** 2, axis=(1, 2))
    elif pred.ndim == 2:
        pred_exp = pred[:, None, :]
        e_pred = np.mean((X[:, -1:, :] - pred_exp) ** 2, axis=(1, 2))
    elif pred.ndim == 3:
        if pred.shape[1] != 1:
            pred = pred[:, :1, :]
        e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2))
    else:
        e_pred = np.zeros(X.shape[0], dtype=np.float32)

    scores = (W_RECON * e_recon) + (W_PRED * e_pred) + (W_GRAD * e_grad)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size and not np.isfinite(scores).all():
        print("[inference] Non-finite values in hybrid window scores; replacing with 0.0")
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scores


def severity(score: float, thr: float) -> str:
    if score <= thr:
        return "NORMAL"
    r = score / max(thr, EPS)
    if r < 1.5:
        return "LOW"
    if r < 3.0:
        return "MEDIUM"
    return "HIGH"


# =========================
# CSV/NPY loaders
# =========================
def load_channels_from_csv(path: Path) -> Tuple[List[str], "np.ndarray"]:
    if pd is None:
        raise HTTPException(500, "pandas is required for CSV")

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    df = df.dropna(how="all")
    if df.empty:
        raise HTTPException(400, "CSV is empty")

    num = df.select_dtypes(include=["number"]).copy()

    if num.shape[1] == 0:
        df2 = df.apply(pd.to_numeric, errors="coerce")
        num = df2.select_dtypes(include=["number"]).copy()
        if num.shape[1] == 0:
            raise HTTPException(400, "No numeric columns found")

    return list(num.columns), num.to_numpy(dtype=np.float32)


def load_channels_from_npy(path: Path) -> Tuple[List[str], "np.ndarray", str]:
    if np is None:
        raise HTTPException(500, "numpy is required for NPY")

    arr = np.load(path)
    arr = np.asarray(arr)

    # (N,) -> 1 channel raw series
    if arr.ndim == 1:
        return ["ch_1"], arr.reshape(-1, 1).astype(np.float32), "raw_series"

    # (N, C) raw OR (B, T) windows
    if arr.ndim == 2:
        if arr.shape[1] == WINDOW_LEN:
            return ["ch_1"], arr.astype(np.float32), "windows_2d"
        else:
            C = arr.shape[1]
            names = [f"ch_{i+1}" for i in range(C)]
            return names, arr.astype(np.float32), "raw_series"

    # (B, T, 1) windows
    if arr.ndim == 3:
        return ["ch_1"], arr.astype(np.float32), "windows_3d"

    raise HTTPException(400, f"Unsupported npy shape: {arr.shape}")


# =========================
# Per-channel analyzers
# =========================
def analyze_one_channel_series(
    model,
    cfg: Dict[str, Any],
    channel_name: str,
    series: "np.ndarray",
    out_dir: Path,
) -> Dict[str, Any]:
    x = clean_series(series)
    Xw, spans = make_windows_1ch(x, WINDOW_LEN, STRIDE)

    out_csv = out_dir / f"{channel_name}_results.csv"
    thr = pick_threshold(cfg)

    if Xw.shape[0] == 0:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["window_index", "start", "end", "score", "threshold", "is_anomaly", "severity"])

        return {
            "channel_name": channel_name,
            "num_windows": 0,
            "num_anomalies": 0,
            "anomaly_rate": 0.0,
            "threshold": float(thr),
            "results_path": str(out_csv),
        }

    scores = compute_scores_hybrid(model, Xw, cfg)
    flags = scores > thr

    num_windows = int(scores.shape[0])
    num_anom = int(flags.sum())
    rate = float(num_anom / max(num_windows, 1))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "start", "end", "score", "threshold", "is_anomaly", "severity"])
        for i, ((s, e), sc, fl) in enumerate(zip(spans, scores, flags)):
            w.writerow([i, s, e, float(sc), float(thr), int(bool(fl)), severity(float(sc), float(thr))])

    return {
        "channel_name": channel_name,
        "num_windows": num_windows,
        "num_anomalies": num_anom,
        "anomaly_rate": rate,
        "threshold": float(thr),
        "results_path": str(out_csv),
    }


def analyze_one_channel_windows(
    model,
    cfg: Dict[str, Any],
    channel_name: str,
    windows: "np.ndarray",
    out_dir: Path,
) -> Dict[str, Any]:
    """
    windows can be (B,T) or (B,T,1)
    """
    thr = pick_threshold(cfg)
    X = np.asarray(windows, dtype=np.float32)

    if X.ndim == 2:
        if X.shape[1] != WINDOW_LEN:
            raise HTTPException(400, f"Windows shape mismatch, expected (B,{WINDOW_LEN}) got {X.shape}")
        X = X[..., None]
    elif X.ndim == 3:
        if X.shape[1] != WINDOW_LEN or X.shape[2] != 1:
            raise HTTPException(400, f"Windows shape mismatch, expected (B,{WINDOW_LEN},1) got {X.shape}")
    else:
        raise HTTPException(400, f"Unsupported windows ndim: {X.ndim}")

    X[~np.isfinite(X)] = 0.0

    scores = compute_scores_hybrid(model, X, cfg)
    flags = scores > thr

    num_windows = int(scores.shape[0])
    num_anom = int(flags.sum())
    rate = float(num_anom / max(num_windows, 1))

    out_csv = out_dir / f"{channel_name}_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "start", "end", "score", "threshold", "is_anomaly", "severity"])
        for i, (sc, fl) in enumerate(zip(scores, flags)):
            w.writerow([i, None, None, float(sc), float(thr), int(bool(fl)), severity(float(sc), float(thr))])

    return {
        "channel_name": channel_name,
        "num_windows": num_windows,
        "num_anomalies": num_anom,
        "anomaly_rate": rate,
        "threshold": float(thr),
        "results_path": str(out_csv),
    }

# ============================================================
# Part 3/3: Endpoints (Auth + Inference + Continual Admin + Web)
# ============================================================

# =========================
# Continual background job
# =========================
def run_continual_training_in_background(dataset_path: Path, admin_user_id: str):
    global _MODEL, _THRESH

    TRAINING_STATUS["running"] = True
    TRAINING_STATUS["stage"] = "training"
    TRAINING_STATUS["message"] = "Continual training started"
    TRAINING_STATUS["dataset_path"] = str(dataset_path)
    TRAINING_STATUS["version"] = None
    TRAINING_STATUS["last_model_path"] = None
    TRAINING_STATUS["last_threshold_path"] = None
    TRAINING_STATUS["last_error"] = None
    TRAINING_STATUS["last_started_at"] = datetime.utcnow().isoformat()
    TRAINING_STATUS["last_finished_at"] = None

    try:
        model_path, thresh_path, accuracy = fine_tune(dataset_path)
        version = Path(model_path).stem

        with db() as conn:
            conn.execute(
                """
                INSERT INTO model_registry
                (id, version, model_path, threshold_path, dataset_path, status, created_at, accuracy)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    uuid.uuid4().hex,
                    version,
                    str(model_path),
                    str(thresh_path),
                    str(dataset_path),
                    "PENDING",
                    datetime.utcnow().isoformat(),
                    accuracy,
                ),
            )

        TRAINING_STATUS["stage"] = "waiting_approval"
        TRAINING_STATUS["message"] = "Training finished and waiting for approval"
        TRAINING_STATUS["version"] = version
        TRAINING_STATUS["last_model_path"] = str(model_path)
        TRAINING_STATUS["last_threshold_path"] = str(thresh_path)

        schedule_admin_notification(
            "Continual training finished (pending approval)",
            f"Initiated by: {admin_user_id}\n"
            f"Version: {version}\n"
            f"Model: {model_path}\n"
            f"Thresholds: {thresh_path}\n"
            f"Accuracy: {accuracy}\n"
            f"Dataset: {dataset_path}\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}Z\n",
        )

    except Exception as e:
        TRAINING_STATUS["stage"] = "failed"
        TRAINING_STATUS["message"] = "Training failed"
        TRAINING_STATUS["last_error"] = str(e)
        err = str(e)[:1800]
        schedule_admin_notification(
            "Continual training failed",
            f"Initiated by: {admin_user_id}\n"
            f"Dataset: {dataset_path}\n"
            f"Error: {err}\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}Z\n",
        )

    finally:
        TRAINING_STATUS["running"] = False
        TRAINING_STATUS["last_finished_at"] = datetime.utcnow().isoformat()


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/admin/evaluation/strict-v2/status")
def strict_eval_status(admin=Depends(require_admin)):
    return STRICT_EVAL_STATUS


@app.post("/admin/evaluation/strict-v2/run")
def run_strict_eval_v2(
    background: BackgroundTasks,
    admin=Depends(require_admin),
):
    """
    Run strict evaluation v2 for the currently ACTIVE inference model (approved or latest registry fallback).
    Writes outputs under backend/app/evaluation_strict_v2/ and updates dashboard endpoints.
    """
    if STRICT_EVAL_STATUS.get("running"):
        return {"status": "already_running", "detail": "Strict evaluation v2 is currently running."}

    mp, _tp = get_active_model_paths()
    if not mp.exists():
        raise HTTPException(500, f"Active model file missing: {mp}")

    background.add_task(_run_strict_eval_v2_job, mp, WINDOW_LEN, STRIDE)
    return {"status": "started", "model_path": str(mp), "output_dir": str((BASE_DIR / "evaluation_strict_v2").resolve())}


# ============================================================
# Reports API
# ============================================================

@app.post("/reports/generate/{run_id}")
def generate_report(
    run_id: str,
    fmt: Literal["both", "pdf", "excel"] = Query(
        "both",
        alias="format",
        description="Report output: pdf, excel, or both",
    ),
    user=Depends(get_current_user),
):
    """
    Generate cybersecurity report (PDF and/or Excel) from existing run results.
    - USER: can generate only for own runs
    - ADMIN: can generate for any run
    """
    with db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")

        if user.get("role") != "ADMIN" and str(run["user_id"]) != str(user.get("sub")):
            raise HTTPException(403, "Forbidden")

        rows = conn.execute(
            """
            SELECT channel_name, num_windows, num_anomalies, anomaly_rate, threshold, results_path
            FROM channel_results
            WHERE run_id=?
            """,
            (run_id,),
        ).fetchall()

        channel_results = [dict(r) for r in rows]

        # Same model/threshold as /analyze for this run (written to results/analysis_meta.json)
        analysis_meta: Dict[str, Any] = {}
        meta_path = run_dir(run_id) / "results" / "analysis_meta.json"
        if meta_path.is_file():
            try:
                analysis_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                analysis_meta = {}
        if not analysis_meta:
            sp = run_dir(run_id) / "results" / "summary.json"
            if sp.is_file():
                try:
                    s = json.loads(sp.read_text(encoding="utf-8"))
                    analysis_meta = dict(s.get("analysis_meta") or {})
                except Exception:
                    pass

        model_version = str(analysis_meta.get("model_version") or "").strip()
        if not model_version:
            approved = conn.execute(
                """
                SELECT version
                FROM model_registry
                WHERE status='APPROVED'
                ORDER BY approved_at DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            model_version = str(approved["version"]) if approved else "BASE"

        model_name = str(analysis_meta.get("model_name") or "").strip() or None
        model_path_snap = str(analysis_meta.get("model_path") or "").strip() or None
        thr_path_snap = str(analysis_meta.get("threshold_path") or "").strip() or None

        report_uuid = uuid.uuid4().hex
        report_id = f"RPT-{datetime.utcnow().strftime('%Y%m%d')}-{report_uuid[:8]}"
        title = "Satellite Telemetry Cybersecurity Anomaly Report"

        report_obj = build_report_json(
            report_id=report_id,
            run=dict(run),
            user=dict(user),
            model_version=model_version,
            model_name=model_name,
            model_path_used_for_run=model_path_snap,
            threshold_path_used_for_run=thr_path_snap,
            analysis_meta=analysis_meta or None,
            channel_results=channel_results,
        )

        pdf_out = REPORTS_PDF_DIR / f"{report_id}.pdf"
        xlsx_out = REPORTS_XLSX_DIR / f"{report_id}.xlsx"

        want_pdf = fmt in ("both", "pdf")
        want_xlsx = fmt in ("both", "excel")

        pdf_path_saved: Optional[Path] = None
        xlsx_path_saved: Optional[Path] = None

        if want_pdf:
            try:
                write_pdf(report_obj, pdf_out)
                pdf_path_saved = pdf_out
            except Exception as e:
                raise HTTPException(500, f"PDF generation failed: {e}") from e

        if want_xlsx:
            try:
                write_excel(report_obj, xlsx_out)
                xlsx_path_saved = xlsx_out
            except Exception as e:
                raise HTTPException(500, f"Excel generation failed: {e}") from e

        dr = report_obj.get("detection_results", {})

        conn.execute(
            """
            INSERT INTO reports
            (id, report_id, run_id, user_id, title, created_at, severity_summary,
             total_windows, normal_count, anomaly_count, low_count, medium_count, high_count,
             threshold_used, model_version, pdf_path, excel_path, report_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                uuid.uuid4().hex,
                report_id,
                run_id,
                str(run["user_id"]),
                title,
                datetime.utcnow().isoformat(),
                dr.get("severity_summary"),
                int(dr.get("total_windows") or 0),
                int(dr.get("normal_count") or 0),
                int(dr.get("anomaly_count") or 0),
                int(dr.get("low_count") or 0),
                int(dr.get("medium_count") or 0),
                int(dr.get("high_count") or 0),
                float(dr.get("threshold_used")) if dr.get("threshold_used") is not None else None,
                model_version,
                str(pdf_path_saved) if pdf_path_saved else None,
                str(xlsx_path_saved) if xlsx_path_saved else None,
                json.dumps(report_obj),
            ),
        )

    return {"status": "generated", "report_id": report_id, "format": fmt}


@app.get("/reports")
def list_reports(user=Depends(get_current_user)):
    with db() as conn:
        if user.get("role") == "ADMIN":
            rows = conn.execute(
                """
                SELECT report_id, run_id, user_id, title, created_at, severity_summary,
                       total_windows, anomaly_count, threshold_used, model_version
                FROM reports
                ORDER BY created_at DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT report_id, run_id, user_id, title, created_at, severity_summary,
                       total_windows, anomaly_count, threshold_used, model_version
                FROM reports
                WHERE user_id=?
                ORDER BY created_at DESC
                """,
                (str(user.get("sub")),),
            ).fetchall()

    return {"reports": [dict(r) for r in rows]}


@app.get("/reports/{report_id}")
def get_report(report_id: str, user=Depends(get_current_user)):
    with db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Report not found")
        if user.get("role") != "ADMIN" and str(row["user_id"]) != str(user.get("sub")):
            raise HTTPException(403, "Forbidden")

    d = dict(row)
    # report_json can be large; return parsed JSON for UI
    try:
        d["report_json"] = json.loads(d["report_json"]) if d.get("report_json") else None
    except Exception:
        d["report_json"] = None
    return d


@app.get("/reports/{report_id}/download/pdf")
def download_report_pdf(report_id: str, user=Depends(get_current_user)):
    with db() as conn:
        row = conn.execute("SELECT user_id, pdf_path FROM reports WHERE report_id=?", (report_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Report not found")
        if user.get("role") != "ADMIN" and str(row["user_id"]) != str(user.get("sub")):
            raise HTTPException(403, "Forbidden")
        p = Path(row["pdf_path"]) if row["pdf_path"] else None
        if not p:
            raise HTTPException(404, "PDF not available")
        p = _resolve_under(REPORTS_DIR, p)
        if not p.exists():
            raise HTTPException(404, "PDF file missing")
    return FileResponse(str(p), media_type="application/pdf", filename=f"{report_id}.pdf")


@app.get("/reports/{report_id}/download/excel")
def download_report_excel(report_id: str, user=Depends(get_current_user)):
    with db() as conn:
        row = conn.execute("SELECT user_id, excel_path FROM reports WHERE report_id=?", (report_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Report not found")
        if user.get("role") != "ADMIN" and str(row["user_id"]) != str(user.get("sub")):
            raise HTTPException(403, "Forbidden")
        p = Path(row["excel_path"]) if row["excel_path"] else None
        if not p:
            raise HTTPException(404, "Excel not available")
        p = _resolve_under(REPORTS_DIR, p)
        if not p.exists():
            raise HTTPException(404, "Excel file missing")
    return FileResponse(
        str(p),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{report_id}.xlsx",
    )


@app.delete("/reports/{report_id}")
def delete_report(report_id: str, user=Depends(get_current_user)):
    with db() as conn:
        row = conn.execute(
            "SELECT user_id, pdf_path, excel_path FROM reports WHERE report_id=?",
            (report_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Report not found")
        if user.get("role") != "ADMIN" and str(row["user_id"]) != str(user.get("sub")):
            raise HTTPException(403, "Forbidden")

        pdf_p = Path(row["pdf_path"]) if row["pdf_path"] else None
        xls_p = Path(row["excel_path"]) if row["excel_path"] else None

        conn.execute("DELETE FROM reports WHERE report_id=?", (report_id,))

    # Delete files (best-effort)
    for p in (pdf_p, xls_p):
        if p:
            try:
                rp = _resolve_under(REPORTS_DIR, p)
                if rp.exists():
                    rp.unlink()
            except Exception:
                pass

    return {"status": "deleted", "report_id": report_id}

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests, slow down"},
    )

@app.post("/admin/backup")
def backup(user=Depends(require_admin)):
    path = create_backup()
    admin_email = user.get("email", "admin")
    schedule_admin_notification(
        "Database backup created",
        f"Administrator: {admin_email}\nBackup path: {path}\nTime (UTC): {datetime.utcnow().isoformat()}Z\n",
    )
    return {"status": "backup_created", "file": path}

# =========================
# Auth endpoints
# =========================
@app.post("/auth/signup")
def signup(data: SignupRequest):
    email = str(data.email).lower().strip()
    pwd_hash = hash_password(data.password)

    print(f"[auth] Signup attempt - Email: {email}")

    with db() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
        if row:
            print(f"[auth] Email already registered: {email}")
            raise HTTPException(400, "Email already registered")

        user_id = uuid.uuid4().hex
        try:
            conn.execute(
                "INSERT INTO users (id,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                (user_id, email, pwd_hash, "USER", datetime.utcnow().isoformat()),
            )
            conn.commit()
            print(f"[auth] User created successfully: {email}, ID: {user_id}")
        except Exception as e:
            print(f"[auth] Error creating user: {e}")
            raise HTTPException(500, f"Database error: {str(e)}")
    
    return {"status": "ok", "message": "User created. You can now login."}

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

@app.post("/auth/login")
@limiter.limit("5/minute")
def login(data: LoginRequest, request: Request):
    email = str(data.email).lower().strip()

    print(f"[auth] Login attempt - Email: {email}")

    with db() as conn:
        row = conn.execute(
            "SELECT id,email,role,is_blocked,password_hash,failed_attempts,locked_until FROM users WHERE email=?",
            (email,),
        ).fetchone()

    # ❌ المستخدم غير موجود
    if not row:
        print(f"[auth] Login failed for: {email}")
        create_incident(
            user_id=email,
            ip=request.client.host,
            type_="FAILED_LOGIN",
            details="Invalid credentials"
        )
        raise HTTPException(401, "Invalid credentials")

    # 🚫 الحساب محظور بشكل دائم
    if row["is_blocked"] == 1:
        print(f"[auth] Account blocked: {email}")
        raise HTTPException(403, "Account is blocked")

    # 🔒 فحص الحظر المؤقت بسبب محاولات فاشلة
    if row["locked_until"]:
        lock_time = datetime.fromisoformat(row["locked_until"])
        if datetime.utcnow() < lock_time:
            remaining = int((lock_time - datetime.utcnow()).total_seconds() / 60) + 1
            print(f"[auth] Account temporarily locked: {email}, {remaining} min remaining")
            create_incident(
                user_id=email,
                ip=request.client.host,
                type_="LOGIN_LOCKED",
                details=f"Account temporarily locked. {remaining} min remaining"
            )
            raise HTTPException(403, f"Account temporarily locked. Try again in {remaining} minutes.")
        else:
            # انتهى وقت الحظر → إعادة تعيين
            with db() as conn:
                conn.execute(
                    "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?",
                    (row["id"],)
                )

    # 🔐 التحقق من كلمة المرور باستخدام bcrypt
    if not verify_password(data.password, row["password_hash"]):
        print(f"[auth] Login failed for: {email}")
        failed = (row["failed_attempts"] or 0) + 1

        with db() as conn:
            if failed >= MAX_FAILED_ATTEMPTS:
                lock_until = (datetime.utcnow() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
                conn.execute(
                    "UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                    (failed, lock_until, row["id"])
                )
                create_incident(
                    user_id=email,
                    ip=request.client.host,
                    type_="ACCOUNT_LOCKED",
                    details=f"Account locked after {failed} failed attempts for {LOCKOUT_MINUTES} min"
                )
                print(f"[auth] Account locked: {email} after {failed} failed attempts")
                raise HTTPException(403, f"Account locked for {LOCKOUT_MINUTES} minutes due to too many failed attempts.")
            else:
                conn.execute(
                    "UPDATE users SET failed_attempts=? WHERE id=?",
                    (failed, row["id"])
                )

        create_incident(
            user_id=email,
            ip=request.client.host,
            type_="FAILED_LOGIN",
            details=f"Invalid credentials (attempt {failed}/{MAX_FAILED_ATTEMPTS})"
        )
        raise HTTPException(401, "Invalid credentials")

    # ✅ تسجيل دخول ناجح → إعادة تعيين المحاولات الفاشلة
    with db() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?",
            (row["id"],)
        )

    # 🔄 ترقية كلمة المرور من SHA-256 إلى bcrypt إذا لازم
    _migrate_password_if_needed(row["id"], data.password, row["password_hash"])

    print(f"[auth] User found: {email}, ID: {row['id']}")

    # 🔐 إنشاء OTP بدل JWT
    otp = generate_otp()
    expires = datetime.utcnow() + timedelta(minutes=5)

    print(f"[auth] Generated OTP: {otp}")

    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO otp_codes (id, user_id, code, expires_at) VALUES (?,?,?,?)",
                (uuid.uuid4().hex, str(row["id"]), otp, expires.isoformat())
            )
            conn.commit()
            print(f"[auth] OTP saved to database for user: {row['id']}")
        except Exception as e:
            print(f"[auth] Error saving OTP: {e}")
            raise HTTPException(500, f"Database error: {str(e)}")

    # 📩 إرسال OTP بالإيميل
    try:
        send_otp_email(email, otp)
        print(f"[email] OTP email sent to: {email}")
    except Exception as e:
        print(f"[email] Error: {e}")
        raise HTTPException(500, "Email sending failed")

    # ✅ الرد الصحيح
    response = {
        "message": "OTP sent",
        "email": email
    }
    print(f"[auth] Returning response: {response}")
    return response
class OTPRequest(BaseModel):
    email: EmailStr
    otp: str

@app.post("/auth/verify-otp")
def verify_otp(data: OTPRequest):

    email = str(data.email).lower().strip()

    with db() as conn:
        user = conn.execute(
            "SELECT id,email,role FROM users WHERE email=?",
            (email,),
        ).fetchone()

        if not user:
            raise HTTPException(404, "User not found")

        otp_row = conn.execute(
            "SELECT * FROM otp_codes WHERE user_id=? ORDER BY expires_at DESC LIMIT 1",
            (str(user["id"]),),
        ).fetchone()

    if not otp_row:
        raise HTTPException(400, "No OTP found")

    if otp_row["code"] != data.otp:
        raise HTTPException(400, "Invalid OTP")

    if datetime.utcnow() > datetime.fromisoformat(otp_row["expires_at"]):
        raise HTTPException(400, "OTP expired")

    # ✅ نجاح → نعطي JWT
    token = create_jwt(
        str(user["id"]),
        str(user["role"]),
        str(user["email"])
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return user


@app.post("/admin/incidents/{incident_id}/close")
def close_incident(incident_id: str, admin=Depends(require_admin)):

    with db() as conn:
        conn.execute(
            "UPDATE incidents SET status='CLOSED' WHERE id=?",
            (incident_id,)
        )

    return {
        "status": "closed",
        "incident_id": incident_id
    }


@app.get("/admin/incidents")
def list_incidents(admin=Depends(require_admin)):

    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC"
        ).fetchall()

    return {
        "incidents": [dict(r) for r in rows]
    }


@app.get("/admin/users")
def list_users(admin=Depends(require_admin)):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, email, role, is_blocked, failed_attempts, locked_until, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
    return {"users": [dict(r) for r in rows]}


@app.post("/admin/users/{user_id}/toggle-block")
def toggle_block_user(user_id: str, admin=Depends(require_admin)):
    with db() as conn:
        row = conn.execute("SELECT is_blocked, email FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, "User not found")
        new_val = 0 if row["is_blocked"] else 1
        conn.execute("UPDATE users SET is_blocked=? WHERE id=?", (new_val, user_id))
        action = "blocked" if new_val else "unblocked"
        create_incident(
            user_id=admin.get("email", "admin"),
            ip="admin-action",
            type_="USER_" + action.upper(),
            details=f"Admin {action} user {row['email']}",
            action_taken=action,
        )
    schedule_admin_notification(
        f"User account {action}",
        f"Administrator: {admin.get('email', 'admin')}\n"
        f"Target user: {row['email']} (id={user_id})\n"
        f"New state: is_blocked={new_val}\n"
        f"Time (UTC): {datetime.utcnow().isoformat()}Z\n",
    )
    return {"status": action, "user_id": user_id}


@app.get("/admin/audit-logs")
def get_audit_logs(admin=Depends(require_admin)):
    with db() as conn:
        incidents = conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        runs = conn.execute(
            "SELECT run_id, user_id, filename, status, created_at FROM runs ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    return {
        "incidents": [dict(r) for r in incidents],
        "runs": [dict(r) for r in runs],
    }


# =========================
# Upload (CSV/NPY)
# =========================
@app.post("/runs/upload")
@limiter.limit("10/minute")
async def upload_run(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):

    # تحقق من اسم الملف
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    name = sanitize_text(file.filename.strip())
    ext = name.lower().split(".")[-1]

    # تحقق من نوع الملف
    if ext not in ("csv", "npy"):
        create_incident(
            user_id=user["sub"],
            ip="unknown",
            type_="INVALID_FILE",
            details=f"Invalid type: {ext}"
        )
        raise HTTPException(400, "Only .csv or .npy allowed")

    # قراءة الملف
    content = await file.read()

    # ✅ حساب الهاش (SHA-256)
    file_hash = hashlib.sha256(content).hexdigest()

    # تحقق من الحجم
    if len(content) > MAX_FILE_SIZE:
        create_incident(
            user_id=user["sub"],
            ip="unknown",
            type_="FILE_TOO_LARGE",
            details=f"Size: {len(content)}"
        )
        raise HTTPException(400, "File too large")

    # إنشاء run_id
    run_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    rdir = run_dir(run_id)

    # حفظ الملف
    raw_path = rdir / f"raw.{ext}"
    save_bytes(raw_path, content)

    # حفظ في الداتا بيز
    with db() as conn:
        conn.execute(
            """
            INSERT INTO runs 
            (run_id, user_id, filename, file_type, file_sha256, status, created_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(user["sub"]),
                name,
                ext,
                file_hash,
                "UPLOADED",
                datetime.utcnow().isoformat()
            ),
        )

    # Clean + validate in the same request (avoids a second /prepare call that may 404 on stale servers)
    try:
        prep = prepare_run_work(run_id)
    except HTTPException:
        raise
    return {
        "run_id": run_id,
        "status": prep["status"],
        "channels": prep.get("channels", []),
    }


def prepare_run_work(run_id: str) -> Dict[str, Any]:
    """
    Read CSV/NPY, clean like inference, write prepare_summary.json, set status CLEANED.
    Used by upload and by POST /runs/{id}/prepare.
    """
    if np is None:
        raise HTTPException(500, "numpy is required")

    rdir = run_dir(run_id)
    raw_csv = rdir / "raw.csv"
    raw_npy = rdir / "raw.npy"

    if not raw_csv.exists() and not raw_npy.exists():
        raise HTTPException(400, "RAW file not found")

    results_dir = rdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    set_run_status(run_id, "PREPARING")

    channels_info: List[Dict[str, Any]] = []

    try:
        if raw_csv.exists():
            names, mat = load_channels_from_csv(raw_csv)
            for j, cname in enumerate(names):
                x = clean_series(mat[:, j])
                n = int(x.size)
                Xw, _ = make_windows_1ch(x, WINDOW_LEN, STRIDE)
                channels_info.append(
                    {
                        "channel_name": cname,
                        "num_samples": n,
                        "num_windows": int(Xw.shape[0]),
                    }
                )
        else:
            names, arr, mode = load_channels_from_npy(raw_npy)

            if mode == "raw_series":
                if arr.ndim != 2:
                    raise HTTPException(400, "Unexpected raw_series shape")
                for j, cname in enumerate(names):
                    x = clean_series(arr[:, j])
                    n = int(x.size)
                    Xw, _ = make_windows_1ch(x, WINDOW_LEN, STRIDE)
                    channels_info.append(
                        {
                            "channel_name": cname,
                            "num_samples": n,
                            "num_windows": int(Xw.shape[0]),
                        }
                    )
            elif mode in ("windows_2d", "windows_3d"):
                nw = int(arr.shape[0])
                channels_info.append(
                    {
                        "channel_name": "ch_1",
                        "num_samples": None,
                        "num_windows": nw,
                    }
                )
            else:
                raise HTTPException(400, f"Unknown NPY mode: {mode}")

        summary = {
            "run_id": run_id,
            "channels": channels_info,
            "prepared_at": datetime.utcnow().isoformat(),
        }
        (results_dir / "prepare_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        set_run_status(run_id, "CLEANED")

        return {
            "run_id": run_id,
            "status": "CLEANED",
            "channels": channels_info,
        }

    except HTTPException:
        set_run_status(run_id, "FAILED")
        raise
    except Exception as e:
        set_run_status(run_id, "FAILED")
        print(f"[prepare_run_work] {run_id}: {type(e).__name__}: {e}")
        raise HTTPException(500, f"Prepare failed: {e}") from e


# =========================
# Prepare (validate + clean only, no model)
# =========================
@app.post("/runs/{run_id}/prepare")
@limiter.limit("30/minute")
def prepare_run(
    request: Request,
    run_id: str,
    user=Depends(get_current_user),
):
    """
    Read CSV/NPY, apply the same cleaning used before inference, persist a small summary.
    Sets run status to CLEANED so the anomaly page can run /analyze separately.
    """
    ensure_run_owner(run_id, user["sub"])
    return prepare_run_work(run_id)


# =========================
# Analyze (Per channel)
# =========================
@app.post("/runs/{run_id}/analyze")
@limiter.limit("5/minute")
def analyze_run(
    request: Request,
    run_id: str,
    user=Depends(get_current_user)
):
    ensure_run_owner(run_id, user["sub"])

    if np is None:
        raise HTTPException(500, "numpy is required")

    set_run_status(run_id, "ANALYZING")

    rdir = run_dir(run_id)
    raw_csv = rdir / "raw.csv"
    raw_npy = rdir / "raw.npy"

    if not raw_csv.exists() and not raw_npy.exists():
        set_run_status(run_id, "FAILED")
        raise HTTPException(400, "RAW file not found")

    results_dir = rdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    channel_summaries: List[Dict[str, Any]] = []

    try:

        model = load_inference_model()
        cfg = load_thresholds()

        mp_active, tp_active = get_active_model_paths()
        _mn = getattr(model, "name", None)
        _name_str = _mn.strip() if isinstance(_mn, str) and _mn.strip() else ""
        analysis_meta_snapshot = {
            "model_name": _name_str or mp_active.stem,
            "model_path": str(mp_active.resolve()),
            "threshold_path": str(tp_active.resolve()),
            "model_version": resolve_model_version_for_path(mp_active),
            "window_length": WINDOW_LEN,
            "stride": STRIDE,
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
        }

        print(f"[analyze_diag] run_id={run_id}")
        print(f"[analyze_diag] model_path={mp_active} exists={mp_active.is_file()}")
        print(f"[analyze_diag] threshold_path={tp_active} exists={tp_active.is_file()}")
        print(f"[analyze_diag] model.input_shape={getattr(model, 'input_shape', None)}")
        _outs = getattr(model, "outputs", None)
        if _outs:
            for _i, _t in enumerate(_outs):
                print(f"[analyze_diag] model.outputs[{_i}] shape={getattr(_t, 'shape', None)}")
        _thr_dbg = pick_threshold(cfg)
        print(f"[analyze_diag] pick_threshold value={_thr_dbg}")

        # =========================
        # CSV DATA
        # =========================
        if raw_csv.exists():

            names, mat = load_channels_from_csv(raw_csv)  # mat (N,C)

            for j, cname in enumerate(names):

                series = mat[:, j]

                summary = analyze_one_channel_series(
                    model, cfg, cname, series, results_dir
                )

                channel_summaries.append(summary)

                # -----------------------------------
                # Continual Learning Buffer
                # -----------------------------------
                x = clean_series(series)

                Xw, spans = make_windows_1ch(
                    x,
                    WINDOW_LEN,
                    STRIDE
                )

                if Xw.shape[0] > 0:
                    if j == 0:
                        print(f"[analyze_diag] first channel X_windows shape={Xw.shape} dtype={Xw.dtype}")

                    scores = compute_scores_hybrid(
                        model,
                        Xw,
                        cfg
                    )

                    threshold = pick_threshold(cfg)

                    optional_process_and_store(x, scores, threshold, spans)

        # =========================
        # NPY DATA
        # =========================
        else:

            names, arr, mode = load_channels_from_npy(raw_npy)

            if mode == "raw_series":

                if arr.ndim != 2:
                    raise HTTPException(400, "Unexpected raw_series shape")

                for j, cname in enumerate(names):

                    series = arr[:, j]

                    summary = analyze_one_channel_series(
                        model,
                        cfg,
                        cname,
                        series,
                        results_dir
                    )

                    channel_summaries.append(summary)

                    # -----------------------------------
                    # Continual Learning Buffer
                    # -----------------------------------
                    x = clean_series(series)

                    Xw, spans = make_windows_1ch(
                        x,
                        WINDOW_LEN,
                        STRIDE
                    )

                    if Xw.shape[0] > 0:
                        if j == 0:
                            print(f"[analyze_diag] first channel X_windows shape={Xw.shape} dtype={Xw.dtype}")

                        scores = compute_scores_hybrid(
                            model,
                            Xw,
                            cfg
                        )

                        threshold = pick_threshold(cfg)

                        optional_process_and_store(x, scores, threshold, spans)

            elif mode in ("windows_2d", "windows_3d"):

                summary = analyze_one_channel_windows(
                    model,
                    cfg,
                    "ch_1",
                    arr,
                    results_dir
                )

                channel_summaries.append(summary)

            else:
                raise HTTPException(400, f"Unknown NPY mode: {mode}")

        # =========================
        # Save results to DB
        # =========================
        with db() as conn:

            conn.execute(
                "DELETE FROM channel_results WHERE run_id=?",
                (run_id,)
            )

            for s in channel_summaries:

                conn.execute(
                    """
                    INSERT INTO channel_results
                    (id, run_id, channel_name, num_windows, num_anomalies, anomaly_rate, threshold, results_path)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        uuid.uuid4().hex,
                        run_id,
                        s["channel_name"],
                        int(s["num_windows"]),
                        int(s["num_anomalies"]),
                        float(s["anomaly_rate"]),
                        float(s["threshold"]),
                        str(s["results_path"]),
                    ),
                )

        summary_path = results_dir / "summary.json"
        meta_path = results_dir / "analysis_meta.json"
        meta_out = {"run_id": run_id, **analysis_meta_snapshot}
        meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

        summary_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "channels": channel_summaries,
                    "analysis_meta": analysis_meta_snapshot,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        set_run_status(run_id, "DONE")

        return {
            "run_id": run_id,
            "status": "DONE",
            "channels": channel_summaries
        }

    except HTTPException:
        set_run_status(run_id, "FAILED")
        raise
    except Exception as e:
        set_run_status(run_id, "FAILED")
        traceback.print_exc()
        print(f"[analyze_run] {run_id}: {type(e).__name__}: {e}")
        raise HTTPException(
            500,
            f"Analyze failed: {e}",
        ) from e

# =========================
# Get run status
# =========================
@app.get("/runs/{run_id}")
def get_run(run_id: str, user=Depends(get_current_user)):
    ensure_run_owner(run_id, user["sub"])
    recover_run_if_analyze_finished(run_id)
    with db() as conn:
        r = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Run not found")
    return dict(r)


@app.post("/runs/{run_id}/bail-analyze")
@limiter.limit("10/minute")
def bail_analyze_run(request: Request, run_id: str, user=Depends(get_current_user)):
    """
    If the UI is stuck on ANALYZING (slow CPU, killed browser tab, or rare server glitch),
    mark the run as FAILED so the user can upload again or start a new run.
    """
    ensure_run_owner(run_id, user["sub"])
    with db() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    if str(row["status"]) == "ANALYZING":
        set_run_status(run_id, "FAILED")
    return {"run_id": run_id, "status": "FAILED"}


# =========================
# Get results (per-channel detailed)
# =========================
@app.get("/runs/{run_id}/results")
def get_results(run_id: str, user=Depends(get_current_user), preview_rows: int = 50):
    ensure_run_owner(run_id, user["sub"])

    with db() as conn:
        r = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Run not found")
        if str(r["status"]) != "DONE":
            raise HTTPException(400, "Results not ready")

        rows = conn.execute(
            """
            SELECT channel_name, num_windows, num_anomalies, anomaly_rate, threshold, results_path
            FROM channel_results
            WHERE run_id=?
            ORDER BY anomaly_rate DESC
            """,
            (run_id,),
        ).fetchall()

    channels_out = []
    for row in rows:
        p = Path(row["results_path"])
        preview = []

        if p.exists():
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, rr in enumerate(reader):
                    preview.append(rr)
                    if i + 1 >= max(1, int(preview_rows)):
                        break

        channels_out.append(
            {
                "channel_name": row["channel_name"],
                "num_windows": int(row["num_windows"]),
                "num_anomalies": int(row["num_anomalies"]),
                "anomaly_rate": float(row["anomaly_rate"]),
                "threshold": float(row["threshold"]),
                "rows_preview": preview,
            }
        )

    return {"run_id": run_id, "channels": channels_out}


# ============================================================
# Expert Reports (per-run, cybersecurity-focused)
# ============================================================

def _run_raw_file_path(run_id: str) -> Optional[Path]:
    rdir = run_dir(run_id)
    for cand in (rdir / "raw.csv", rdir / "raw.npy"):
        if cand.exists():
            return cand
    return None


def _get_channel_results_rows(run_id: str) -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT channel_name, num_windows, num_anomalies, anomaly_rate, threshold, results_path
            FROM channel_results
            WHERE run_id=?
            ORDER BY anomaly_rate DESC
            """,
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _load_analysis_meta_for_run(run_id: str) -> Dict[str, Any]:
    try:
        meta_path = run_dir(run_id) / "results" / "analysis_meta.json"
        if meta_path.is_file():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _risk_level_from_report(report_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Map existing report posture to requested scale:
      - overall_state from build_report_json is Normal / Suspicious / Critical
      - Return Low/Medium/High/Critical + short explanation.
    """
    es = report_obj.get("executive_summary") or {}
    overall = str(es.get("overall_state") or "").strip().lower()
    dr = report_obj.get("detection_results") or {}
    ar = float(dr.get("anomaly_rate") or 0.0) * 100.0
    hi = int(dr.get("high_count") or 0)
    med = int(dr.get("medium_count") or 0)

    if overall == "critical":
        return "Critical", f"Critical posture (High windows={hi} or anomaly rate={ar:.2f}%)."
    if overall == "suspicious":
        return "High", f"Suspicious posture (Medium windows={med} or anomaly rate={ar:.2f}%)."
    # Normal posture: split Low/Medium based on anomaly rate
    if ar >= 1.0:
        return "Medium", f"Elevated anomaly rate {ar:.2f}% but no Medium/High posture triggers."
    return "Low", f"No escalation triggers; anomaly rate {ar:.2f}%."


@app.get("/reports/runs")
def list_expert_reports(
    user=Depends(get_current_user),
    risk: Optional[str] = Query(None, description="Optional risk filter: low|medium|high|critical"),
    q: Optional[str] = Query(None, description="Optional filename substring filter"),
):
    """
    Returns one row per completed run with expert-report summary fields.
    Does NOT create saved report rows; it is built from run + channel CSV outputs.
    """
    risk_norm = str(risk or "").strip().lower()
    q_norm = str(q or "").strip().lower()

    with db() as conn:
        rows = conn.execute(
            """
            SELECT r.run_id, r.user_id, r.filename, r.file_type, r.file_sha256, r.status, r.created_at, u.email AS user_email
            FROM runs r
            LEFT JOIN users u ON u.id = r.user_id
            WHERE r.user_id=?
            ORDER BY r.created_at DESC
            """,
            (str(user["sub"]),),
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        run_id = str(r["run_id"])
        filename = r["filename"]
        if q_norm and q_norm not in str(filename or "").lower():
            continue

        cr = _get_channel_results_rows(run_id) if str(r["status"]) == "DONE" else []
        total_windows = sum(int(x.get("num_windows") or 0) for x in cr)
        total_anom = sum(int(x.get("num_anomalies") or 0) for x in cr)
        anomaly_rate = (total_anom / max(1, total_windows)) if total_windows else None

        # Best-effort: derive a report JSON and risk only when DONE
        report_obj = None
        risk_level = None
        risk_reason = None
        if str(r["status"]) == "DONE" and cr:
            meta = _load_analysis_meta_for_run(run_id)
            report_obj = build_report_json(
                report_id=f"RUN-{run_id}",
                run=dict(r),
                user=dict(user),
                model_version=str(meta.get("model_version") or ""),
                channel_results=cr,
                analysis_meta=meta or None,
                model_name=str(meta.get("model_name") or "") or None,
                model_path_used_for_run=str(meta.get("model_path") or "") or None,
                threshold_path_used_for_run=str(meta.get("threshold_path") or "") or None,
            )
            risk_level, risk_reason = _risk_level_from_report(report_obj)

        if risk_norm and risk_level and risk_norm != str(risk_level).lower():
            continue

        raw_path = _run_raw_file_path(run_id)
        size_bytes = raw_path.stat().st_size if raw_path and raw_path.exists() else None

        out.append(
            {
                "run_id": run_id,
                "filename": filename,
                "uploaded_at": r["created_at"],
                "user_email": r.get("user_email"),
                "file_type": r["file_type"],
                "file_size_bytes": size_bytes,
                "status": r["status"],
                "num_channels": len(cr) if cr else None,
                "num_windows": total_windows if total_windows else None,
                "anomaly_windows": total_anom if total_windows else None,
                "anomaly_rate": float(anomaly_rate) if anomaly_rate is not None else None,
                "risk_level": risk_level,
                "risk_reason": risk_reason,
            }
        )

    return {"reports": out}


@app.get("/reports/run/{run_id}")
def get_expert_report(run_id: str, user=Depends(get_current_user), anomaly_limit: int = 2500):
    """
    Returns a cybersecurity-expert report for a specific run, built from stored run results.
    """
    ensure_run_owner(run_id, user["sub"])
    recover_run_if_analyze_finished(run_id)

    with db() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run_row:
        raise HTTPException(404, "Run not found")
    if str(run_row["status"]) != "DONE":
        raise HTTPException(400, "Results not ready")

    cr = _get_channel_results_rows(run_id)
    meta = _load_analysis_meta_for_run(run_id)
    run_user_email = None
    try:
        with db() as conn:
            urow = conn.execute("SELECT email FROM users WHERE id=?", (str(run_row["user_id"]),)).fetchone()
        if urow and urow["email"]:
            run_user_email = str(urow["email"])
    except Exception:
        run_user_email = None

    report_obj = build_report_json(
        report_id=f"RUN-{run_id}",
        run=dict(run_row),
        user=dict(user),
        model_version=str(meta.get("model_version") or ""),
        channel_results=cr,
        analysis_meta=meta or None,
        model_name=str(meta.get("model_name") or "") or None,
        model_path_used_for_run=str(meta.get("model_path") or "") or None,
        threshold_path_used_for_run=str(meta.get("threshold_path") or "") or None,
    )

    # Enrich with requested expert fields (without fabricating)
    det = report_obj.get("detection_results") or {}
    risk_level, risk_reason = _risk_level_from_report(report_obj)
    raw_path = _run_raw_file_path(run_id)
    size_bytes = raw_path.stat().st_size if raw_path and raw_path.exists() else None

    anomaly_rows = read_anomaly_table_rows(cr, limit=int(anomaly_limit))

    return {
        "title": "Cybersecurity Expert Reports",
        "file_summary": {
            "filename": run_row["filename"],
            "uploaded_at": run_row["created_at"],
            "user_email": run_user_email,
            "file_type": run_row["file_type"],
            "file_size_bytes": size_bytes,
            "status": run_row["status"],
            "num_windows": det.get("total_windows"),
        },
        "detection_summary": {
            "normal_windows": det.get("normal_count"),
            "anomaly_windows": det.get("anomaly_count"),
            "normal_rate": det.get("normal_rate"),
            "anomaly_rate": det.get("anomaly_rate"),
            "all_windows_scores": {
                "min": det.get("score_min"),
                "mean": det.get("score_mean"),
                "max": det.get("score_max"),
            },
            "normal_windows_scores": {
                "min": det.get("normal_windows_score_min"),
                "mean": det.get("normal_windows_score_mean"),
                "max": det.get("normal_windows_score_max"),
            },
            "flagged_windows_scores": {
                "min": det.get("anomaly_windows_score_min"),
                "mean": det.get("anomaly_windows_score_mean"),
                "max": det.get("anomaly_windows_score_max"),
            },
            "threshold_used": det.get("threshold_used"),
            "threshold_type": (load_operating_config().get("operating_threshold", {}) or {}).get("name") or "operating_threshold",
            "labels_ar": {
                "normal": "نوافذ طبيعية — على العتبة أو تحتها",
                "flagged": "نوافذ غير طبيعية — فوق عتبة التشغيل",
            },
        },
        "risk_assessment": {
            "risk_level": risk_level,
            "reason": risk_reason,
            "attack_indicators": report_obj.get("security_interpretation"),
            "executive_summary": report_obj.get("executive_summary"),
        },
        "technical_details": {
            "model_name": (meta.get("model_name") or report_obj.get("technical_appendix", {}).get("model_name")),
            "model_version": (meta.get("model_version") or report_obj.get("technical_appendix", {}).get("model_version_used_for_run")),
            "threshold_file": (meta.get("threshold_path") or report_obj.get("technical_appendix", {}).get("threshold_file_path")),
            "operating_config": str(OPERATING_CONFIG_PATH),
            "scoring_formula": report_obj.get("technical_appendix", {}).get("scoring_formula"),
            "weights": report_obj.get("technical_appendix", {}).get("weights"),
            "num_channels": len(cr),
            "window_size": meta.get("window_length"),
            "stride": meta.get("stride"),
        },
        "anomaly_details": {
            "rows": anomaly_rows,
            "rows_truncated": len(anomaly_rows) >= int(anomaly_limit),
            "limit": int(anomaly_limit),
        },
        "recommendations": report_obj.get("recommendations"),
        "exports": {
            "pdf": f"/reports/run/{run_id}/pdf",
            "excel": f"/reports/run/{run_id}/excel",
        },
    }


@app.get("/reports/run/{run_id}/pdf")
def export_expert_report_pdf(run_id: str, user=Depends(get_current_user)):
    ensure_run_owner(run_id, user["sub"])
    meta = _load_analysis_meta_for_run(run_id)
    cr = _get_channel_results_rows(run_id)
    with db() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run_row or str(run_row["status"]) != "DONE":
        raise HTTPException(400, "Results not ready")

    report_obj = build_report_json(
        report_id=f"RUN-{run_id}",
        run=dict(run_row),
        user=dict(user),
        model_version=str(meta.get("model_version") or ""),
        channel_results=cr,
        analysis_meta=meta or None,
        model_name=str(meta.get("model_name") or "") or None,
        model_path_used_for_run=str(meta.get("model_path") or "") or None,
        threshold_path_used_for_run=str(meta.get("threshold_path") or "") or None,
    )

    out_dir = REPORTS_PDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"RUN-{run_id}.pdf"
    write_pdf(report_obj, out_path)
    return FileResponse(str(out_path), media_type="application/pdf", filename=out_path.name)


@app.get("/reports/run/{run_id}/excel")
def export_expert_report_excel(run_id: str, user=Depends(get_current_user)):
    ensure_run_owner(run_id, user["sub"])
    meta = _load_analysis_meta_for_run(run_id)
    cr = _get_channel_results_rows(run_id)
    with db() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run_row or str(run_row["status"]) != "DONE":
        raise HTTPException(400, "Results not ready")

    report_obj = build_report_json(
        report_id=f"RUN-{run_id}",
        run=dict(run_row),
        user=dict(user),
        model_version=str(meta.get("model_version") or ""),
        channel_results=cr,
        analysis_meta=meta or None,
        model_name=str(meta.get("model_name") or "") or None,
        model_path_used_for_run=str(meta.get("model_path") or "") or None,
        threshold_path_used_for_run=str(meta.get("threshold_path") or "") or None,
    )

    out_dir = REPORTS_XLSX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"RUN-{run_id}.xlsx"
    write_excel(report_obj, out_path)
    return FileResponse(
        str(out_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=out_path.name,
    )


def _segment_window_series(
    run_id: str,
    channel_name: str,
    window_index: int,
) -> Tuple[List[float], Optional[int], Optional[int]]:
    """
    Return cleaned telemetry samples for one segment window (same indexing as results CSV).
    """
    if np is None:
        raise HTTPException(500, "numpy is required")

    rdir = run_dir(run_id)
    raw_csv = rdir / "raw.csv"
    raw_npy = rdir / "raw.npy"

    wi = int(window_index)

    if raw_csv.exists():
        names, mat = load_channels_from_csv(raw_csv)
        try:
            j = names.index(channel_name)
        except ValueError:
            raise HTTPException(404, f"Channel '{channel_name}' not found in raw data")
        series = mat[:, j]
        x = clean_series(series)
        Xw, spans = make_windows_1ch(x, WINDOW_LEN, STRIDE)
        if wi < 0 or wi >= Xw.shape[0]:
            raise HTTPException(404, "window_index out of range")
        vals = np.asarray(Xw[wi, :, 0], dtype=np.float64)
        s, e = spans[wi]
        return vals.tolist(), int(s), int(e)

    if raw_npy.exists():
        names, arr, mode = load_channels_from_npy(raw_npy)

        if mode == "raw_series":
            if arr.ndim != 2:
                raise HTTPException(400, "Unexpected raw_series shape")
            try:
                j = names.index(channel_name)
            except ValueError:
                raise HTTPException(404, f"Channel '{channel_name}' not found in raw data")
            series = arr[:, j]
            x = clean_series(series)
            Xw, spans = make_windows_1ch(x, WINDOW_LEN, STRIDE)
            if wi < 0 or wi >= Xw.shape[0]:
                raise HTTPException(404, "window_index out of range")
            vals = np.asarray(Xw[wi, :, 0], dtype=np.float64)
            s, e = spans[wi]
            return vals.tolist(), int(s), int(e)

        if mode in ("windows_2d", "windows_3d"):
            if channel_name not in names and len(names) != 1:
                raise HTTPException(404, f"Channel '{channel_name}' not found in raw data")
            if arr.ndim == 2:
                b = arr.shape[0]
                if wi < 0 or wi >= b:
                    raise HTTPException(404, "window_index out of range")
                vals = np.asarray(arr[wi, :], dtype=np.float64).reshape(-1)
                return vals.tolist(), None, None
            if arr.ndim == 3:
                b = arr.shape[0]
                if wi < 0 or wi >= b:
                    raise HTTPException(404, "window_index out of range")
                vals = np.asarray(arr[wi, :, 0], dtype=np.float64).reshape(-1)
                return vals.tolist(), None, None
            raise HTTPException(400, f"Unsupported windows array ndim: {arr.ndim}")

        raise HTTPException(400, f"Unsupported NPY mode for segment window: {mode}")

    raise HTTPException(400, "RAW file not found")


@app.get("/runs/{run_id}/segment-window")
def get_segment_window(
    run_id: str,
    channel: str,
    window_index: int,
    user=Depends(get_current_user),
):
    """Telemetry trace for one analysis window (for Analysis page chart)."""
    if np is None:
        raise HTTPException(500, "numpy is required")

    ensure_run_owner(run_id, user["sub"])

    with db() as conn:
        r = conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Run not found")
    if str(r["status"]) != "DONE":
        raise HTTPException(400, "Results not ready")

    values, start, end = _segment_window_series(run_id, channel, window_index)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "run_id": run_id,
            "channel": channel,
            "window_index": window_index,
            "values": [],
            "start": start,
            "end": end,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "window_length": 0,
        }

    return {
        "run_id": run_id,
        "channel": channel,
        "window_index": int(window_index),
        "values": values,
        "start": start,
        "end": end,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "window_length": int(arr.size),
    }


# ============================================================
# Continual Admin - List Normal Pool Files
# ============================================================

@app.get("/admin/continual/normal-files")
def list_normal_files(admin=Depends(require_admin)):

    normal_dir = BASE_DIR.parent / "data" / "continual" / "normal_pool"

    if not normal_dir.exists():
        return {"count": 0, "files": []}

    files = []

    for f in normal_dir.glob("*.npy"):
        files.append(f.name)

    return {
        "count": len(files),
        "files": sorted(files)
    }

# ============================================================
# Continual Admin - Read Normal Pool Data
# ============================================================

@app.get("/admin/continual/normal-data")
def read_normal_data(file: str, view: str = "series", admin=Depends(require_admin)):

    import numpy as np

    normal_dir = BASE_DIR.parent / "data" / "continual" / "normal_pool"
    file_path = normal_dir / file

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    X = np.load(file_path)

    # تحويل البيانات
    if view == "windows":
        data = X.tolist()
        flat = X.reshape(-1)

    else:
        flat = X.reshape(-1)
        data = flat.tolist()

    # حساب الإحصائيات
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    minv = float(np.min(flat))
    maxv = float(np.max(flat))

    return {
        "file": file,
        "shape": list(X.shape),
        "view": view,
        "mean": mean,
        "std": std,
        "min": minv,
        "max": maxv,
        "data": data
    }

# ============================================================
# Continual Admin - Approve Dataset
# ============================================================

@app.post("/admin/continual/approve-normal")
def approve_normal_file(file: str, admin=Depends(require_admin)):

    import shutil

    normal_dir = BASE_DIR.parent / "data" / "continual" / "normal_pool"
    dataset_dir = BASE_DIR.parent / "data" / "continual" / "datasets"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    src = normal_dir / file

    if not src.exists():
        raise HTTPException(404, "File not found")

    dst = dataset_dir / file

    shutil.move(src, dst)

    return {
        "status": "approved",
        "dataset_file": str(dst)
    }

# ============================================================
# Continual Admin - Reject Dataset
# ============================================================

@app.post("/admin/continual/reject-normal")
def reject_normal_file(file: str, admin=Depends(require_admin)):

    normal_dir = BASE_DIR.parent / "data" / "continual" / "normal_pool"
    src = normal_dir / file

    if not src.exists():
        raise HTTPException(404, "File not found")

    src.unlink()

    return {
        "status": "rejected",
        "deleted_file": file
    }

# ============================================================
# Continual Admin - List Datasets
# ============================================================

@app.get("/admin/continual/datasets")
def list_datasets(admin=Depends(require_admin)):

    dataset_dir = BASE_DIR.parent / "data" / "continual" / "datasets"

    if not dataset_dir.exists():
        return {"datasets": []}

    files = []

    for f in dataset_dir.glob("*.npy"):
        files.append(f.name)

    return {
        "count": len(files),
        "datasets": sorted(files)
    }


# ============================================================
# Continual Admin - Anomaly Pool Files
# ============================================================

@app.get("/admin/continual/anomaly-files")
def list_anomaly_files(admin=Depends(require_admin)):
    anomaly_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_pool"
    if not anomaly_dir.exists():
        return {"count": 0, "files": []}
    files = [f.name for f in anomaly_dir.glob("*.npy")]
    return {"count": len(files), "files": sorted(files)}


@app.get("/admin/continual/anomaly-data")
def read_anomaly_data(file: str, view: str = "series", admin=Depends(require_admin)):
    import numpy as np

    anomaly_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_pool"
    file_path = anomaly_dir / file
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    X = np.load(file_path)
    if view == "windows":
        data = X.tolist()
        flat = X.reshape(-1)
    else:
        flat = X.reshape(-1)
        data = flat.tolist()

    return {
        "file": file,
        "shape": list(X.shape),
        "view": view,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "data": data,
    }


@app.post("/admin/continual/approve-anomaly")
def approve_anomaly_file(file: str, admin=Depends(require_admin)):
    import shutil

    anomaly_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_pool"
    dataset_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    src = anomaly_dir / file
    if not src.exists():
        raise HTTPException(404, "File not found")
    dst = dataset_dir / file
    shutil.move(src, dst)
    return {"status": "approved", "dataset_file": str(dst)}


@app.post("/admin/continual/reject-anomaly")
def reject_anomaly_file(file: str, admin=Depends(require_admin)):
    anomaly_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_pool"
    src = anomaly_dir / file
    if not src.exists():
        raise HTTPException(404, "File not found")
    src.unlink()
    return {"status": "rejected", "deleted_file": file}


@app.get("/admin/continual/anomaly-datasets")
def list_anomaly_datasets(admin=Depends(require_admin)):
    dataset_dir = BASE_DIR.parent / "data" / "continual" / "anomaly_datasets"
    if not dataset_dir.exists():
        return {"datasets": []}
    files = [f.name for f in dataset_dir.glob("*.npy")]
    return {"count": len(files), "datasets": sorted(files)}




# =========================
# Dashboard runs
# =========================
@app.get("/dashboard/runs")
def dashboard_runs(user=Depends(get_current_user)):
    with db() as conn:
        runs = conn.execute(
            """
            SELECT
                r.run_id,
                r.filename,
                r.file_type,
                r.status,
                r.created_at,
                COALESCE(AVG(cr.anomaly_rate), 0.0) AS anomaly_rate
            FROM runs r
            LEFT JOIN channel_results cr ON cr.run_id = r.run_id
            WHERE r.user_id=?
            GROUP BY r.run_id, r.filename, r.file_type, r.status, r.created_at
            ORDER BY r.created_at DESC
            """,
            (str(user["sub"]),),
        ).fetchall()

        anomaly_types = conn.execute(
            """
            SELECT
                cr.channel_name AS type,
                SUM(cr.num_anomalies) AS count
            FROM channel_results cr
            INNER JOIN runs r ON r.run_id = cr.run_id
            WHERE r.user_id=?
            GROUP BY cr.channel_name
            HAVING SUM(cr.num_anomalies) > 0
            ORDER BY count DESC
            """,
            (str(user["sub"]),),
        ).fetchall()

    runs_out = [dict(r) for r in runs]
    done_runs = sum(1 for r in runs_out if str(r.get("status")) == "DONE")
    prediction_success = round((done_runs / len(runs_out)) * 100, 1) if runs_out else None

    # Snapshot must reflect *inference* weights (get_active_model_paths), not only the newest registry row.
    # After rollback there may be no APPROVED row; registry still lists older continual runs as REJECTED.
    latest_model = get_latest_model_row(prefer_approved=True)

    active_model_path, active_threshold_path = get_active_model_paths()
    try:
        active_mp_s = str(active_model_path.resolve())
        active_tp_s = str(active_threshold_path.resolve())
    except OSError:
        active_mp_s = str(active_model_path)
        active_tp_s = str(active_threshold_path)

    registry_matches_active = False
    if latest_model:
        try:
            mp_reg = resolve_artifact_path(Path(latest_model["model_path"]))
            tp_reg = resolve_artifact_path(Path(latest_model["threshold_path"]))
            registry_matches_active = (
                str(mp_reg.resolve()) == active_mp_s and str(tp_reg.resolve()) == active_tp_s
            )
        except Exception:
            registry_matches_active = False

    model_status = "Unavailable"
    model_name = active_model_path.stem if active_model_path.exists() else "—"
    try:
        bundled_mp_s = str(resolve_artifact_path(BUNDLED_MODEL_PATH).resolve())
    except OSError:
        bundled_mp_s = str(BUNDLED_MODEL_PATH)
    if active_mp_s == bundled_mp_s:
        model_version = "BASE"
    else:
        model_version = resolve_model_version_for_path(active_model_path)

    last_training = (
        datetime.utcfromtimestamp(active_model_path.stat().st_mtime).isoformat()
        if active_model_path.exists()
        else None
    )
    model_accuracy = None
    model_balanced_accuracy = None
    model_f1 = None
    model_precision = None
    model_recall = None
    model_far = None
    model_note = "Using base model configuration."

    if registry_matches_active and latest_model:
        model_balanced_accuracy = latest_model.get("accuracy")
        last_training = (
            latest_model.get("approved_at") or latest_model.get("created_at") or last_training
        )
        model_note = f"Latest model from registry (status: {latest_model.get('status')})."
    elif latest_model and not registry_matches_active:
        try:
            lm_stem = resolve_artifact_path(Path(latest_model["model_path"])).stem
            model_note = (
                f"Inference uses {model_name} ({model_version}). "
                f"Registry lists {lm_stem} ({latest_model.get('status')}) — not selected for inference."
            )
        except Exception:
            model_note = (
                f"Inference uses {model_name}. A different model exists in the registry but is not active."
            )

    model_files_ready = active_model_path.exists() and active_threshold_path.exists()
    model_status = "Operational" if model_files_ready else "Unavailable"

    # Snapshot metrics: evaluation summary first; fill any missing fields from thresholds.performance_metrics.
    try:
        summary = load_dashboard_evaluation_summary() or {}
        if isinstance(summary.get("metrics"), dict):
            m = summary.get("metrics") or {}
            if model_accuracy is None:
                model_accuracy = m.get("accuracy")
            if model_balanced_accuracy is None:
                model_balanced_accuracy = m.get("balanced_accuracy")
            if model_f1 is None:
                model_f1 = m.get("f1")
            if model_precision is None:
                model_precision = m.get("precision")
            if model_recall is None:
                model_recall = m.get("recall") or m.get("tpr")
            if model_far is None:
                model_far = m.get("far") or m.get("fpr")

        best = summary.get("best_operating_point_by_f1_among_candidates", {}) or {}
        if not best:
            best = summary.get("best_threshold_by_accuracy", {}) or summary.get("best_threshold_by_f1", {}) or {}

        if best:
            if model_accuracy is None:
                model_accuracy = best.get("accuracy")
            if model_balanced_accuracy is None:
                model_balanced_accuracy = best.get("balanced_accuracy")

        thresholds_cfg = load_active_thresholds_cfg()
        pm = thresholds_cfg.get("performance_metrics", {}) if isinstance(thresholds_cfg, dict) else {}
        if model_accuracy is None:
            model_accuracy = pm.get("accuracy") or pm.get("balanced_accuracy")
        if model_balanced_accuracy is None:
            model_balanced_accuracy = pm.get("balanced_accuracy")
        if model_f1 is None:
            model_f1 = pm.get("f1")
        if model_precision is None:
            model_precision = pm.get("precision")
        if model_recall is None:
            model_recall = pm.get("recall") or pm.get("true_positive_rate")
        if model_far is None:
            model_far = pm.get("far") or pm.get("false_positive_rate")
    except Exception:
        pass

    return {
        "runs": runs_out,
        "anomaly_types": [dict(a) for a in anomaly_types],
        "model": {
            "name": model_name,
            "version": model_version,
            "status": model_status,
            "last_training": last_training,
            "accuracy": model_accuracy,
            "balanced_accuracy": model_balanced_accuracy,
            "f1": model_f1,
            "precision": model_precision,
            "recall": model_recall,
            "far": model_far,
            "prediction_success": prediction_success,
            "health_pct": prediction_success if prediction_success is not None else (100.0 if model_files_ready else 0.0),
            "note": model_note,
        },
    }


# =========================
# Model Accuracy/Performance
# =========================
@app.get("/dashboard/model-accuracy")
def get_model_accuracy(user=Depends(get_current_user)):
    """Return model performance metrics (prefer strict evaluation if available)."""
    try:
        summary = load_dashboard_evaluation_summary()
        if summary:
            curves = summary.get("curves", {}) or {}

            # strict_v2 shape
            best = summary.get("best_operating_point_by_f1_among_candidates", {}) or {}
            if best:
                thresholds_cfg = load_active_thresholds_cfg()
                op_thr, op_src = get_operating_threshold_value(thresholds_cfg if isinstance(thresholds_cfg, dict) else {})

                tnr = best.get("tnr")
                tpr = best.get("recall") or best.get("tpr")
                thr_value = op_thr

                # If we still don't have TPR/TNR for the actual operating threshold, compute from strict window scores.
                if tpr is None or tnr is None:
                    m = _strict_metrics_for_threshold(float(thr_value))
                    if m:
                        tpr = m.get("recall")
                        tnr = m.get("tnr")

                return {
                    "source": "strict_v2",
                    "roc_auc": curves.get("roc_auc"),
                    "pr_auc": curves.get("pr_auc"),
                    "accuracy": best.get("accuracy"),
                    "balanced_accuracy": best.get("balanced_accuracy"),
                    "precision": best.get("precision"),
                    "recall": tpr,
                    "tnr": tnr,
                    "far": best.get("far"),
                    "fnr": best.get("fnr"),
                    "f1": best.get("f1"),
                    "threshold_best_f1": best.get("threshold"),
                    "operating_threshold_value": thr_value,
                    "operating_threshold_source": op_src,
                    "confusion_matrix": {
                        "TP": best.get("TP"),
                        "TN": best.get("TN"),
                        "FP": best.get("FP"),
                        "FN": best.get("FN"),
                    },
                }

            # qc_filtered operational snapshot (bundled metrics JSON; policy is operational p99)
            if isinstance(summary.get("metrics"), dict) and isinstance(summary.get("selection"), dict):
                m = summary.get("metrics") or {}
                sel = summary.get("selection") or {}
                thr_eval = sel.get("threshold")
                cm_raw = summary.get("confusion_matrix") or {}
                thresholds_cfg = load_active_thresholds_cfg()
                op_thr, op_src = get_operating_threshold_value(
                    thresholds_cfg if isinstance(thresholds_cfg, dict) else {}
                )
                return {
                    "source": "qc_filtered_p99",
                    "roc_auc": m.get("roc_auc"),
                    "pr_auc": m.get("pr_auc"),
                    "accuracy": m.get("accuracy"),
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall") or m.get("tpr"),
                    "tnr": m.get("tnr"),
                    "far": m.get("far") or m.get("fpr"),
                    "fnr": m.get("fnr"),
                    "f1": m.get("f1"),
                    "operational_policy": sel.get("policy") or "p99",
                    "threshold_evaluation_snapshot": thr_eval,
                    "threshold_best_f1": thr_eval,
                    "operating_threshold_value": float(op_thr),
                    "operating_threshold_source": op_src,
                    "confusion_matrix": {
                        "TP": cm_raw.get("TP"),
                        "TN": cm_raw.get("TN"),
                        "FP": cm_raw.get("FP"),
                        "FN": cm_raw.get("FN"),
                    },
                }

            best = summary.get("best_threshold_by_f1", {}) or summary.get("best_threshold_by_accuracy", {}) or {}
            if best:
                return {
                    "source": "qc_filtered_eval",
                    "roc_auc": curves.get("roc_auc"),
                    "pr_auc": curves.get("pr_auc"),
                    "accuracy": best.get("accuracy"),
                    "balanced_accuracy": best.get("balanced_accuracy"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall") or best.get("tpr"),
                    "tnr": best.get("tnr"),
                    "far": best.get("far"),
                    "fnr": best.get("fnr"),
                    "f1": best.get("f1"),
                    "threshold_best_f1": best.get("threshold"),
                    "operating_threshold_value": best.get("threshold"),
                    "operating_threshold_source": str(summary.get("thresholds_file") or "qc_filtered_eval"),
                    "confusion_matrix": {
                        "TP": best.get("TP"),
                        "TN": best.get("TN"),
                        "FP": best.get("FP"),
                        "FN": best.get("FN"),
                    },
                }

        # Fallback: performance_metrics embedded in active thresholds file (e.g. thresholds_qc_filtered.json)
        thresholds_cfg = load_active_thresholds_cfg()
        metrics = thresholds_cfg.get("performance_metrics", {}) if isinstance(thresholds_cfg, dict) else {}
        cm = metrics.get("confusion_matrix") or {}
        if metrics:
            op_thr, op_src = get_operating_threshold_value(thresholds_cfg if isinstance(thresholds_cfg, dict) else {})
            return {
                "source": "thresholds.performance_metrics",
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall") or metrics.get("true_positive_rate"),
                "tnr": metrics.get("true_negative_rate"),
                "far": metrics.get("far") or metrics.get("false_positive_rate"),
                "fnr": metrics.get("fnr"),
                "f1": metrics.get("f1"),
                "operational_policy": metrics.get("operational_policy"),
                "operating_threshold_value": float(op_thr) if op_thr is not None else metrics.get("threshold_value"),
                "operating_threshold_source": op_src,
                "threshold_evaluation_snapshot": metrics.get("threshold_value"),
                "threshold_best_f1": metrics.get("threshold_value"),
                "model_status": metrics.get("model_status", "Operational"),
                "confusion_matrix": {
                    "TP": cm.get("TP"),
                    "TN": cm.get("TN"),
                    "FP": cm.get("FP"),
                    "FN": cm.get("FN"),
                },
            }
        return {
            "source": "thresholds.json",
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "tnr": metrics.get("true_negative_rate"),
            "recall": metrics.get("true_positive_rate"),
            "operating_threshold_value": metrics.get("threshold_value", 0.1),
            "model_status": metrics.get("model_status", "Operational"),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load accuracy metrics: {str(e)}")


@app.get("/model/active")
def get_active_model_info(user=Depends(get_current_user)):
    """
    Active inference model + threshold configuration + (optional) strict_v2 evaluation summary.
    Useful for frontend "model card" displays.
    """
    mp, tp = get_active_model_paths()
    thresholds_cfg = load_active_thresholds_cfg()
    op_thr, op_src = get_operating_threshold_value(thresholds_cfg if isinstance(thresholds_cfg, dict) else {})

    strict_summary = None
    try:
        if STRICT_EVAL_SUMMARY_PATH.exists():
            strict_summary = json.loads(STRICT_EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        strict_summary = None

    return {
        "model": {
            "path": str(mp),
            "name": mp.stem,
            "version": resolve_model_version_for_path(mp) if mp.exists() else None,
            "exists": bool(mp.exists()),
            "input_shape_expected": [None, WINDOW_LEN, 1],
        },
        "thresholds": {
            "path": str(tp),
            "exists": bool(tp.exists()),
            "operating_threshold_value": float(op_thr),
            "operating_threshold_source": str(op_src),
            "available_threshold_keys": list((thresholds_cfg.get("thresholds", {}) or {}).keys())
            if isinstance(thresholds_cfg, dict)
            else [],
            "weights": thresholds_cfg.get("weights") if isinstance(thresholds_cfg, dict) else None,
        },
        "strict_evaluation_v2": strict_summary,
        "strict_evaluation_summary_path": str(STRICT_EVAL_SUMMARY_PATH),
    }


# ============================================================
# Continual Admin Web Endpoints
# ============================================================

@app.get("/admin/continual")
def continual_admin_page(admin=Depends(require_admin)):
    if not ADMIN_CONTINUAL_HTML.exists():
        raise HTTPException(404, "admin-cl.html not found")
    return FileResponse(str(ADMIN_CONTINUAL_HTML))


@app.get("/admin/continual/status")
def continual_status(admin=Depends(require_admin)):
    return TRAINING_STATUS


@app.get("/admin/continual/models")
def list_continual_models(admin=Depends(require_admin)):
    with db() as conn:
        rows = conn.execute(
            """
            SELECT version, model_path, threshold_path, dataset_path, status,
                   created_at, approved_at, approved_by
            FROM model_registry
            ORDER BY created_at DESC
            """
        ).fetchall()

    return {"models": [dict(r) for r in rows]}


# ============================================================
# Continual Admin API Endpoints
# ============================================================

@app.post("/admin/continual/build-dataset")
def api_build_dataset(admin=Depends(require_admin)):
    if not CONTINUAL_AVAILABLE:
        raise HTTPException(500, "Continual modules not available")

    dataset_path = build_dataset()
    if not dataset_path:
        raise HTTPException(400, "Dataset was not created")

    TRAINING_STATUS["stage"] = "dataset_built"
    TRAINING_STATUS["message"] = "Dataset created successfully"
    TRAINING_STATUS["dataset_path"] = str(dataset_path)

    return {"status": "dataset_created", "dataset_path": str(dataset_path)}

@app.post("/admin/continual/train")
def train_continual(user=Depends(require_admin)):
    if not CONTINUAL_AVAILABLE:
        raise HTTPException(503, "Continual learning module not available. System using stable 90.24% baseline.")
    
    try:
        # Use the combined dataset from Build Dataset (approved pools), not the largest legacy chunk.
        dataset_path = resolve_continual_training_dataset_path()

        print("DATASET PATH:", dataset_path)

        if not dataset_path.exists():
            raise HTTPException(400, f"Dataset not found: {dataset_path}")

        print("Using dataset:", dataset_path)

        # تشغيل continual learning الحقيقي
        model_path, thresh_path, accuracy = fine_tune(dataset_path)

        version = Path(model_path).stem

        # تسجيل المودل في registry
        with db() as conn:
            conn.execute(
                """
                INSERT INTO model_registry
                (id, version, model_path, threshold_path, dataset_path, status, created_at, accuracy)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    uuid.uuid4().hex,
                    version,
                    str(model_path),
                    str(thresh_path),
                    str(dataset_path),
                    "PENDING",
                    datetime.utcnow().isoformat(),
                    accuracy,
                ),
            )

        return {
            "status": "training_complete",
            "version": version,
            "model_path": str(model_path),
            "threshold_path": str(thresh_path)
        }

    except HTTPException:
        raise
    except ValueError as e:
        print("TRAIN ERROR:", e)
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        print("TRAIN ERROR:", e)
        raise HTTPException(500, str(e)) from e


@app.post("/admin/continual/rollback/{version}")
def rollback_model(version: str, user=Depends(require_admin)):

    global _MODEL, _THRESH

    try:
        with db() as conn:

            # تأكد إن المودل موجود
            model = conn.execute(
                "SELECT * FROM model_registry WHERE version=?",
                (version,)
            ).fetchone()

            if not model:
                raise HTTPException(404, "Model version not found")

            # Rollback: لا يبقى APPROVED — الاستدلال يعود لـ المودل/العتبة الافتراضيين (انظر get_active_model_paths).
            conn.execute(
                "UPDATE model_registry SET status='REJECTED', approved_at=NULL, approved_by=NULL"
            )

        _MODEL = None
        _THRESH = None

        mp = resolve_artifact_path(BUNDLED_MODEL_PATH)
        tp = resolve_artifact_path(BUNDLED_THRESH_PATH)
        print(f"Rollback done → best model ({mp.name}) (requested from {version})")

        schedule_admin_notification(
            "Model registry rollback",
            f"Administrator: {user.get('email', 'admin')}\n"
            f"Requested version context: {version}\n"
            f"All registry rows set to REJECTED; inference falls back to bundled model.\n"
            f"Active model: {mp}\n"
            f"Active thresholds: {tp}\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}Z\n",
        )

        return {
            "status": "rollback_complete",
            "active_version": "bundled_best",
            "model_path": str(mp),
            "threshold_path": str(tp),
        }

    except Exception as e:
        print("ROLLBACK ERROR:", e)
        raise HTTPException(500, str(e))


@app.post("/admin/continual/approve/{version}")
def approve_model(version: str, user=Depends(require_admin)):

    global _MODEL, _THRESH

    try:
        with db() as conn:

            model = conn.execute(
                "SELECT * FROM model_registry WHERE version=?",
                (version,)
            ).fetchone()

            if not model:
                raise HTTPException(404, "Model not found")

            # ❌ نلغي تفعيل كل الموديلات
            conn.execute(
                "UPDATE model_registry SET status='REJECTED', approved_at=NULL, approved_by=NULL"
            )

            # ✅ نفعل هذا المودل
            conn.execute(
                "UPDATE model_registry SET status='APPROVED', approved_at=?, approved_by=? WHERE version=?",
                (datetime.utcnow().isoformat(), user.get("email", "admin"), version)
            )

        # Force inference reload to start using the newly approved model immediately.
        _MODEL = None
        _THRESH = None

        schedule_admin_notification(
            f"Model approved: {version}",
            f"Administrator: {user.get('email', 'admin')}\n"
            f"Version: {version}\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}Z\n"
            f"Inference cache cleared; active weights will reload on next request.\n",
        )

        return {
            "status": "approved",
            "version": version
        }

    except Exception as e:
        print("APPROVE ERROR:", e)
        raise HTTPException(500, str(e))