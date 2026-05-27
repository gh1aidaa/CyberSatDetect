#!/usr/bin/env python3
"""
Collect terminal-style verification output for thesis screenshots (Ch. 6 / 7).
Run from repo:  python backend/scripts/generate_report_test_outputs.py
Writes: backend/REPORT_TEST_RESULTS_FOR_THESIS.txt
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"
DB_PATH = Path(os.environ.get("CSD_DB_PATH", str(APP_DIR / "data" / "app.db")))
OUT_PATH = BACKEND_DIR / "REPORT_TEST_RESULTS_FOR_THESIS.txt"


def _load_dotenv() -> None:
    p = BACKEND_DIR / ".env"
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def section(title: str) -> str:
    bar = "=" * 72
    return f"\n{bar}\n{title}\n{bar}\n"


def test_model_output_validation() -> str:
    """6.7.2 - fake predict() returns NaN/Inf; capture [inference] logs."""
    lines: list[str] = []
    buf = io.StringIO()
    try:
        import numpy as np

        sys.path.insert(0, str(BACKEND_DIR))
        os.chdir(str(BACKEND_DIR))
        _load_dotenv()

        from app.api import WINDOW_LEN, compute_scores_hybrid

        class FakeModel:
            def predict(self, X, verbose=0):
                B, T, C = X.shape
                recon = np.ones((B, T, C), dtype=np.float32)
                recon[0, 0, 0] = np.nan
                pred = np.ones((B, T - 1, C), dtype=np.float32)
                pred[0, 0, 0] = np.inf
                return recon, pred

        X = np.random.randn(4, WINDOW_LEN, 1).astype(np.float32)
        cfg = {"weights": {}}
        with redirect_stdout(buf):
            scores = compute_scores_hybrid(FakeModel(), X, cfg)
        log = buf.getvalue()
        lines.append("Procedure: FakeModel.predict() returns NaN in recon, +Inf in pred.\n")
        lines.append("Server log (stdout) during compute_scores_hybrid:\n")
        lines.append(log if log.strip() else "(no log - unexpected)\n")
        finite = bool(np.isfinite(scores).all())
        lines.append(f"\nResult: np.isfinite(scores).all() = {finite}\n")
        lines.append(f"scores (first 4 windows): {scores[:4]}\n")
        if not finite:
            lines.append("FAIL: scores still non-finite.\n")
        else:
            lines.append("PASS: hybrid window scores are all finite after validation.\n")
    except Exception as e:
        lines.append(f"ERROR: {type(e).__name__}: {e}\n")
    return "".join(lines)


def sql_snapshots() -> str:
    lines: list[str] = []
    if not DB_PATH.is_file():
        lines.append(f"DB not found: {DB_PATH}\n")
        return "".join(lines)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        lines.append("--- Incident counts (security logging) ---\n")
        for row in conn.execute(
            "SELECT type, COUNT(*) AS n FROM incidents GROUP BY type ORDER BY n DESC"
        ):
            lines.append(f"  {row['type']}: {row['n']}\n")

        lines.append("\n--- Recent runs: file_sha256 (integrity) sample ---\n")
        for row in conn.execute(
            """
            SELECT run_id, substr(file_sha256,1,24) AS sha24, length(file_sha256) AS sha_len
            FROM runs WHERE file_sha256 IS NOT NULL
            ORDER BY created_at DESC LIMIT 5
            """
        ):
            lines.append(
                f"  run_id={row['run_id'][:40]}...  sha256_prefix={row['sha24']}...  len={row['sha_len']}\n"
            )

        lines.append("\n--- Users: email stored (normalization demo) ---\n")
        for row in conn.execute("SELECT email FROM users ORDER BY created_at DESC LIMIT 5"):
            lines.append(f"  stored email: {row['email']!r}\n")
    finally:
        conn.close()
    return "".join(lines)


def security_headers_from_code() -> str:
    """Static excerpt - headers are defined in api.py SecurityHeadersMiddleware."""
    api = APP_DIR / "api.py"
    if not api.is_file():
        return "api.py not found.\n"
    text = api.read_text(encoding="utf-8", errors="ignore")
    lines: list[str] = []
    lines.append("Active HTTP security headers (from SecurityHeadersMiddleware in api.py):\n")
    for key in (
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "Referrer-Policy",
        "Permissions-Policy",
    ):
        if key in text:
            lines.append(f"  [OK] {key}\n")
        else:
            lines.append(f"  [?] {key} (not found by string search)\n")
    return "".join(lines)


def admin_notify_status() -> str:
    _load_dotenv()
    notify = (os.environ.get("CSD_ADMIN_NOTIFY_EMAILS") or "").strip()
    gu = (os.environ.get("GMAIL_USER") or "").strip()
    gp = (os.environ.get("GMAIL_APP_PASSWORD") or "").strip()
    lines = [
        "Admin email notify configuration:\n",
        f"  CSD_ADMIN_NOTIFY_EMAILS: {'set' if notify else 'EMPTY'}\n",
        f"  GMAIL_USER / GMAIL_APP_PASSWORD: {'set' if gu and gp else 'MISSING'}\n",
        "  (Run test_admin_notify.py separately to trigger [admin-notify] Sent in server log.)\n",
    ]
    return "".join(lines)


def main() -> None:
    parts = []
    parts.append("CyberSatDetect - automated verification output\n")
    parts.append(f"Generated (local): {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"Database: {DB_PATH}\n")

    parts.append(section("6.7.2 Model output validation (non-finite -> zero + log)"))
    parts.append(test_model_output_validation())

    parts.append(section("6.3.4 File integrity (DB snapshot) + 6.3.1 incident counts"))
    parts.append(sql_snapshots())

    parts.append(section("7.4.2 Security headers (code presence check)"))
    parts.append(security_headers_from_code())

    parts.append(section("6.5.5 Administrative notifications (env check)"))
    parts.append(admin_notify_status())

    parts.append(
        section("Notes for screenshots")
        + "1. Open this file in editor with monospace font -> screenshot.\n"
        + "2. For live 429 rate limit: POST /runs/upload >10/min with valid token -> Too many requests.\n"
        + "3. For live admin email: run python backend/scripts/test_admin_notify.py -> check server log + inbox.\n"
    )

    text = "".join(parts)
    OUT_PATH.write_text(text, encoding="utf-8")
    # Avoid Windows console encoding errors (cp1256 etc.)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    print(f">>> Saved to: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
