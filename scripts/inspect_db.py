"""Inspect the SQLite database used by the backend.

Usage:
    python scripts/inspect_db.py              # show all tables + row counts
    python scripts/inspect_db.py users        # show first 20 rows of `users`
    python scripts/inspect_db.py users 100    # show first 100 rows of `users`
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from textwrap import shorten


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "backend" / "app" / "data" / "app.db"


def list_tables(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        print("(no user tables in the database)")
        return

    print(f"Database file : {DB_PATH}")
    print(f"Size on disk  : {DB_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Tables        : {len(tables)}\n")

    print(f"{'TABLE':<35} {'ROWS':>8}    COLUMNS")
    print("-" * 90)
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
        print(f"{t:<35} {n:>8}    {', '.join(cols)[:60]}")


def show_table(conn: sqlite3.Connection, table: str, limit: int = 20) -> None:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    cols = [r[1] for r in cur.fetchall()]
    if not cols:
        print(f"Table '{table}' not found.")
        return

    rows = conn.execute(f"SELECT * FROM '{table}' LIMIT ?", (limit,)).fetchall()
    total = conn.execute(f"SELECT COUNT(*) FROM '{table}'").fetchone()[0]

    print(f"Table : {table}    (showing {len(rows)} of {total} rows)\n")
    header = " | ".join(c[:14] for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(shorten(str(v), width=14, placeholder="…") for v in row))


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        if len(sys.argv) >= 2:
            table = sys.argv[1]
            limit = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
            show_table(conn, table, limit)
        else:
            list_tables(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
