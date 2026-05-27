from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "backend" / "app" / "data" / "app.db"
    model_path = (root / "backend" / "app" / "best_model_qc_filtered.keras").resolve()
    thresh_path = (root / "backend" / "app" / "thresholds_qc_filtered.json").resolve()

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not thresh_path.exists():
        raise SystemExit(f"Thresholds not found: {thresh_path}")

    version = "qc_filtered_best"
    now = datetime.utcnow().isoformat()
    approved_by = "system"

    acc_val = None
    try:
        import json as _json

        data = _json.loads(thresh_path.read_text(encoding="utf-8"))
        pm = data.get("performance_metrics") if isinstance(data, dict) else None
        if isinstance(pm, dict) and pm.get("accuracy") is not None:
            acc_val = float(pm["accuracy"])
    except Exception:
        acc_val = None

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Reject all, then approve this one (keeps inference selection deterministic).
    con.execute("UPDATE model_registry SET status='REJECTED', approved_at=NULL, approved_by=NULL")

    row = con.execute("SELECT id FROM model_registry WHERE version=?", (version,)).fetchone()
    if row:
        con.execute(
            """
            UPDATE model_registry
            SET model_path=?,
                threshold_path=?,
                status='APPROVED',
                created_at=?,
                approved_at=?,
                approved_by=?,
                accuracy=?
            WHERE version=?
            """,
            (str(model_path), str(thresh_path), now, now, approved_by, acc_val, version),
        )
    else:
        con.execute(
            """
            INSERT INTO model_registry
            (id, version, model_path, threshold_path, dataset_path, status, created_at, approved_at, approved_by, accuracy)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (uuid.uuid4().hex, version, str(model_path), str(thresh_path), None, "APPROVED", now, now, approved_by, acc_val),
        )

    con.commit()
    con.close()

    print("approved", version)
    print("model_path", str(model_path))
    print("threshold_path", str(thresh_path))
    print("registry_accuracy", acc_val)


if __name__ == "__main__":
    main()

