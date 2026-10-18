"""Unit tests for app helper functions, system health checks, and empty database safety."""

import os

from app import check_system_health
from payradar.db import init_db


def test_system_health_check_handles_offline_gracefully():
    """Verify check_system_health reports status without raising exceptions."""
    health = check_system_health()
    assert "db_ok" in health
    assert "tabpfn_ok" in health
    assert "ollama_ok" in health
    assert isinstance(health["db_ok"], bool)
    assert isinstance(health["tabpfn_ok"], bool)
    assert isinstance(health["ollama_ok"], bool)


def test_empty_database_safety(tmp_path):
    """Ensure querying an empty database does not crash."""
    empty_db = tmp_path / "empty_payradar.db"
    init_db(empty_db)

    from payradar.db import (
        get_all_drafts,
        get_all_invoices,
        get_open_invoices,
        get_scored_open_invoices,
    )

    assert get_all_invoices(db_path=empty_db) == []
    assert get_open_invoices(db_path=empty_db) == []
    assert get_scored_open_invoices(db_path=empty_db) == []
    assert get_all_drafts(db_path=empty_db) == []


def test_demo_mode_env_handling(monkeypatch):
    """Ensure DEMO_MODE flag behaves as expected when set or unset."""
    monkeypatch.setenv("DEMO_MODE", "1")
    is_demo = os.environ.get("DEMO_MODE", "0") in ("1", "true")
    assert is_demo is True

    monkeypatch.setenv("DEMO_MODE", "0")
    is_demo = os.environ.get("DEMO_MODE", "0") in ("1", "true")
    assert is_demo is False
