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

CREATE TABLE IF NOT EXISTS drafts (
    invoice_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    tone TEXT NOT NULL,
    risk REAL NOT NULL,
    reason TEXT,
    is_user_edited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'Gemma',
    source TEXT NOT NULL DEFAULT 'gemma'
);

CREATE INDEX IF NOT EXISTS idx_invoices_client ON invoices(client);
CREATE INDEX IF NOT EXISTS idx_invoices_issued_on ON invoices(issued_on);
CREATE INDEX IF NOT EXISTS idx_invoices_paid_on ON invoices(paid_on);
CREATE INDEX IF NOT EXISTS idx_scores_scored_at ON scores(scored_at);
CREATE INDEX IF NOT EXISTS idx_drafts_updated_at ON drafts(updated_at);
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
        # Ensure source column exists on drafts table
        cur.execute("PRAGMA table_info(drafts)")
        cols = [r[1] for r in cur.fetchall()]
        if cols and "source" not in cols:
            cur.execute("ALTER TABLE drafts ADD COLUMN source TEXT NOT NULL DEFAULT 'gemma'")


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


def get_scored_open_invoices(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return open invoices joined with their latest predictions from scores table."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT i.*, s.risk, s.predicted_days_late, s.scored_at, s.model
            FROM invoices i
            LEFT JOIN scores s ON i.id = s.invoice_id
            WHERE i.paid_on IS NULL
            ORDER BY COALESCE(s.risk, -1.0) DESC, i.due_on ASC
            """
        )
        return [dict(row) for row in cur.fetchall()]


def save_draft(
    invoice_id: str,
    subject: str,
    body: str,
    tone: str,
    risk: float,
    reason: str | None = None,
    is_user_edited: bool = False,
    force: bool = False,
    model: str = "Gemma",
    source: str = "gemma",
    db_path: str | Path | None = None,
) -> bool:
    """Save or update a reminder draft.

    If the draft has previously been edited by the user (is_user_edited == 1)
    and force is False and the new save is an automated draft (is_user_edited == False),
    the save is rejected to ensure we never overwrite user edits.

    Returns True if saved, False if overwrite was blocked.
    """
    init_db(db_path)
    import datetime

    now_iso = datetime.datetime.now().isoformat()
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT is_user_edited, created_at FROM drafts WHERE invoice_id = ?", (invoice_id,)
        )
        existing = cur.fetchone()

        if existing:
            prev_user_edited = bool(existing["is_user_edited"])
            if prev_user_edited and not is_user_edited and not force:
                return False  # Block overwriting user edits

            existing["created_at"]
            conn.execute(
                """
                UPDATE drafts
                SET subject = ?, body = ?, tone = ?, risk = ?, reason = ?,
                    is_user_edited = ?, updated_at = ?, model = ?, source = ?
                WHERE invoice_id = ?
                """,
                (
                    subject,
                    body,
                    tone,
                    float(risk),
                    reason or "",
                    1 if is_user_edited else (1 if prev_user_edited else 0),
                    now_iso,
                    model,
                    source,
                    invoice_id,
                ),
            )
            return True
        else:
            conn.execute(
                """
                INSERT INTO drafts (
                    invoice_id, subject, body, tone, risk, reason,
                    is_user_edited, created_at, updated_at, model, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invoice_id,
                    subject,
                    body,
                    tone,
                    float(risk),
                    reason or "",
                    1 if is_user_edited else 0,
                    now_iso,
                    now_iso,
                    model,
                    source,
                ),
            )
            return True


def get_draft(invoice_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """Return the saved reminder draft for a specific invoice, or None."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM drafts WHERE invoice_id = ?", (invoice_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_drafts(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return all saved drafts ordered by updated_at descending."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM drafts ORDER BY updated_at DESC")
        return [dict(row) for row in cur.fetchall()]
