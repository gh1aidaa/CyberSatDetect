#!/usr/bin/env python3
"""
Trigger security incidents via the live API (INVALID_FILE, FILE_TOO_LARGE, FAILED_LOGIN).

Usage (from repo root or backend; server must be running):
  python backend/scripts/seed_security_incidents.py --base-url http://127.0.0.1:8001

Auth (pick one):
  - Set CSD_API_TOKEN to a valid Bearer JWT, or
  - Default: mint a JWT using the first row in app.db and JWT_SECRET (must match the
    running uvicorn process; default in code is "change-me").

Optional:
  --failed-login-email user@x.com   POST wrong password once -> FAILED_LOGIN incident
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

try:
    import jwt
except ImportError:
    print("PyJWT required: pip install PyJWT", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests required: pip install requests", file=sys.stderr)
    sys.exit(1)


def _repo_backend_app_dir() -> Path:
    # backend/scripts/seed_security_incidents.py -> backend/app
    return Path(__file__).resolve().parent.parent / "app"


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_backend_dotenv() -> None:
    """Set os.environ defaults from backend/.env (no python-dotenv dependency)."""
    env_path = _backend_dir() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _db_path() -> Path:
    p = Path(os.environ.get("CSD_DB_PATH", str(_repo_backend_app_dir() / "data" / "app.db")))
    if not p.is_file():
        raise SystemExit(f"Database not found: {p}")
    return p


def _mint_token(user_id: str, role: str, email: str, secret: str, algo: str, expire_min: int) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "role": role,
        "email": email,
        "exp": now + timedelta(minutes=expire_min),
        "iat": now,
    }
    return jwt.encode(payload, secret, algorithm=algo)


def _pick_user(conn: sqlite3.Connection) -> tuple[str, str, str]:
    row = conn.execute(
        "SELECT id, email, role FROM users ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if not row:
        raise SystemExit("No users in database; register one user first.")
    return str(row[0]), str(row[1]), str(row[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("CSD_BASE_URL", "http://127.0.0.1:8001"))
    ap.add_argument(
        "--failed-login-email",
        default=os.environ.get("CSD_FAILED_LOGIN_EMAIL", ""),
        help="If set, one failed login attempt is sent (creates FAILED_LOGIN).",
    )
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    _load_backend_dotenv()

    token = (os.environ.get("CSD_API_TOKEN") or "").strip()
    secret = os.environ.get("JWT_SECRET", "change-me")
    algo = os.environ.get("JWT_ALGO", "HS256")
    expire_min = int(os.environ.get("JWT_EXPIRE_MIN", "60"))

    if not token:
        dbp = _db_path()
        with sqlite3.connect(str(dbp)) as conn:
            uid, email, role = _pick_user(conn)
        token = _mint_token(uid, role, email, secret, algo, expire_min)
        print(f"[auth] Minted JWT for user {email} ({role}) (JWT_SECRET from environment or backend/.env).")

    headers = {"Authorization": f"Bearer {token}"}

    # --- INVALID_FILE ---
    files_bad = {"file": ("malware.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = requests.post(f"{base}/runs/upload", headers=headers, files=files_bad, timeout=60)
    print(f"[upload pdf] HTTP {r.status_code} body={r.text[:200]}")
    if r.status_code not in (400, 401, 403):
        print("  (expected 400 for invalid extension)", file=sys.stderr)

    # --- FILE_TOO_LARGE (valid name, body > 50MB) ---
    over = 50 * 1024 * 1024 + 4096
    huge = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        with open(huge.name, "wb") as f:
            chunk = b"\x00" * (1024 * 1024)
            left = over
            while left > 0:
                n = min(left, len(chunk))
                f.write(chunk[:n])
                left -= n
        with open(huge.name, "rb") as fh:
            files_huge = {"file": ("telemetry_oversize.csv", fh, "text/csv")}
            r2 = requests.post(f"{base}/runs/upload", headers=headers, files=files_huge, timeout=120)
        print(f"[upload huge] HTTP {r2.status_code} body={r2.text[:200]}")
        if r2.status_code not in (400, 401, 403):
            print("  (expected 400 for file too large)", file=sys.stderr)
    finally:
        try:
            os.unlink(huge.name)
        except OSError:
            pass

    # --- FAILED_LOGIN (optional) ---
    email_fail = (args.failed_login_email or "").strip()
    if email_fail:
        r3 = requests.post(
            f"{base}/auth/login",
            json={"email": email_fail, "password": "__wrong_password_seed__"},
            timeout=30,
        )
        print(f"[failed login] HTTP {r3.status_code} body={r3.text[:200]}")

    print("\nDone. Open Security Center as ADMIN and refresh Incidents (INVALID_FILE, FILE_TOO_LARGE, maybe FAILED_LOGIN).")


if __name__ == "__main__":
    main()
