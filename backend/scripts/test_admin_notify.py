#!/usr/bin/env python3
"""
Trigger one admin notification email (POST /admin/backup) using a minted ADMIN JWT.

Prerequisites in backend/.env (same as running uvicorn):
  GMAIL_USER=...
  GMAIL_APP_PASSWORD=...
  CSD_ADMIN_NOTIFY_EMAILS=recipient@gmail.com

Usage:
  python backend/scripts/test_admin_notify.py --base-url http://127.0.0.1:8001

Then check:
  - Inbox of addresses in CSD_ADMIN_NOTIFY_EMAILS (subject starts with [CyberSatDetect Admin])
  - Server console for [admin-notify] Sent or Failed / Skipped
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import jwt
except ImportError:
    print("pip install PyJWT", file=sys.stderr)
    sys.exit(1)
try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _app_dir() -> Path:
    return _backend_dir() / "app"


def _load_backend_dotenv() -> None:
    env_path = _backend_dir() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _db_path() -> Path:
    p = Path(os.environ.get("CSD_DB_PATH", str(_app_dir() / "data" / "app.db")))
    if not p.is_file():
        raise SystemExit(f"Database not found: {p}")
    return p


def _pick_admin(conn: sqlite3.Connection) -> tuple[str, str, str]:
    row = conn.execute(
        "SELECT id, email, role FROM users WHERE role='ADMIN' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if row:
        return str(row[0]), str(row[1]), str(row[2])
    raise SystemExit("No ADMIN user in database. Promote a user to ADMIN or log in as admin first.")


def _mint(uid: str, role: str, email: str) -> str:
    secret = os.environ.get("JWT_SECRET", "change-me")
    algo = os.environ.get("JWT_ALGO", "HS256")
    exp = int(os.environ.get("JWT_EXPIRE_MIN", "60"))
    now = datetime.utcnow()
    payload = {
        "sub": uid,
        "role": role,
        "email": email,
        "exp": now + timedelta(minutes=exp),
        "iat": now,
    }
    return jwt.encode(payload, secret, algorithm=algo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("CSD_BASE_URL", "http://127.0.0.1:8001"))
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    _load_backend_dotenv()

    notify = (os.environ.get("CSD_ADMIN_NOTIFY_EMAILS") or "").strip()
    gu = (os.environ.get("GMAIL_USER") or "").strip()
    gp = (os.environ.get("GMAIL_APP_PASSWORD") or "").strip()

    print("[check] CSD_ADMIN_NOTIFY_EMAILS:", "set" if notify else "EMPTY (no email will be sent)")
    print("[check] GMAIL_USER / GMAIL_APP_PASSWORD:", "set" if gu and gp else "MISSING (notify will skip)")

    with sqlite3.connect(str(_db_path())) as conn:
        uid, email, role = _pick_admin(conn)
    token = _mint(uid, role, email)
    print(f"[auth] Using ADMIN JWT for: {email}")

    r = requests.post(
        f"{base}/admin/backup",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    print(f"[backup] HTTP {r.status_code}")
    try:
        print("[backup] body:", r.json())
    except Exception:
        print("[backup] body:", r.text[:500])

    print("\nNext: watch uvicorn terminal for [admin-notify] Sent / Failed / Skipped")
    print("       and check inbox(es) in CSD_ADMIN_NOTIFY_EMAILS.")


if __name__ == "__main__":
    main()
