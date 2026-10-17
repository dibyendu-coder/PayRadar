"""SQLite database management for PayRadar.

Data Model:
- invoices(id, client, issued_on, due_on, amount, currency, terms_days, project_type, paid_on NULL, source_file)
- settings(key, value)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DEFAULT_DB_PATH = os.environ.get("PAYRADAR_DB_PATH", "payradar.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    client TEXT NOT NULL,
    issued_on TEXT NOT NULL,
    due_on TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    terms_days INTEGER NOT NULL DEFAULT 30,
    project_type TEXT NOT NULL DEFAULT 'consulting',
    paid_on TEXT,
    source_file TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    invoice_id TEXT PRIMARY KEY,
    risk REAL NOT NULL,
    predicted_days_late REAL NOT NULL,
    scored_at TEXT NOT NULL,
    model TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invoices_client ON invoices(client);
CREATE INDEX IF NOT EXISTS idx_invoices_issued_on ON invoices(issued_on);
CREATE INDEX IF NOT EXISTS idx_invoices_paid_on ON invoices(paid_on);
CREATE INDEX IF NOT EXISTS idx_scores_scored_at ON scores(scored_at);
"""


@contextmanager
def get_db(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional scope around a series of SQLite operations."""
    path = str(db_path or DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Initialize database tables and default settings if they do not exist."""
    with get_db(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        # Set default monthly_costs setting if not already present
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = 'monthly_costs'")
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                ("monthly_costs", "5000.0"),
            )


def insert_invoice(
    invoice_data: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    """Insert or replace an invoice record."""
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO invoices (
                id, client, issued_on, due_on, amount, currency,
                terms_days, project_type, paid_on, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice_data["id"],
                invoice_data["client"],
                invoice_data["issued_on"],
                invoice_data["due_on"],
                float(invoice_data["amount"]),
                invoice_data.get("currency", "USD"),
                int(invoice_data.get("terms_days", 30)),
                invoice_data.get("project_type", "general"),
                invoice_data.get("paid_on"),
                invoice_data.get("source_file"),
            ),
        )


def get_all_invoices(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return all invoices ordered by issued_on."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM invoices ORDER BY issued_on ASC")
        return [dict(row) for row in cur.fetchall()]


def get_open_invoices(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return open (unpaid) invoices where paid_on is NULL."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM invoices WHERE paid_on IS NULL ORDER BY due_on ASC")
        return [dict(row) for row in cur.fetchall()]


def get_setting(
    key: str, default: str | None = None, db_path: str | Path | None = None
) -> str | None:
    """Get a configuration setting value."""
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str, db_path: str | Path | None = None) -> None:
    """Set or update a configuration setting value."""
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, str(value)),
        )


def save_scores(scores: list[dict[str, Any]], db_path: str | Path | None = None) -> None:
    """Save or update predictions in the scores table."""
    init_db(db_path)
    with get_db(db_path) as conn:
        for s in scores:
            conn.execute(
                """
                INSERT OR REPLACE INTO scores (
                    invoice_id, risk, predicted_days_late, scored_at, model
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    s["invoice_id"],
                    float(s["risk"]),
                    float(s["predicted_days_late"]),
                    s["scored_at"],
                    s["model"],
                ),
            )


def get_scores(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return all scored open invoices from the scores table."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM scores ORDER BY risk DESC")
        return [dict(row) for row in cur.fetchall()]
