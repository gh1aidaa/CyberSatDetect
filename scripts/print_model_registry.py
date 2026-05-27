from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "backend" / "app" / "data" / "app.db"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT version,status,model_path,threshold_path,created_at,approved_at,approved_by,accuracy "
        "FROM model_registry ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    for r in rows:
        print(json.dumps(dict(r), ensure_ascii=False))
    con.close()


if __name__ == "__main__":
    main()

