"""Unit tests for Phase 0 scaffold and database schema."""

import tempfile
from pathlib import Path

from payradar import __version__
from payradar.db import (
    get_all_invoices,
    get_open_invoices,
    get_setting,
    init_db,
    insert_invoice,
    set_setting,
)


def test_version():
    assert __version__ == "0.1.0"


def test_db_init_and_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_payradar.db"
        init_db(test_db)

        # Check default settings
        monthly_costs = get_setting("monthly_costs", db_path=test_db)
        assert monthly_costs == "5000.0"

        # Update setting
        set_setting("monthly_costs", "6500.0", db_path=test_db)
        assert get_setting("monthly_costs", db_path=test_db) == "6500.0"

        # Insert historical paid invoice
        inv1 = {
            "id": "INV-2024-001",
            "client": "Acme Corp",
            "issued_on": "2024-01-15",
            "due_on": "2024-02-14",
            "amount": 2500.00,
            "currency": "USD",
            "terms_days": 30,
            "project_type": "web_dev",
            "paid_on": "2024-02-18",
            "source_file": "inv_001.pdf",
        }
        insert_invoice(inv1, db_path=test_db)

        # Insert open unpaid invoice
        inv2 = {
            "id": "INV-2024-002",
            "client": "Beta LLC",
            "issued_on": "2024-02-01",
            "due_on": "2024-03-02",
            "amount": 4200.00,
            "currency": "USD",
            "terms_days": 30,
            "project_type": "design",
            "paid_on": None,
            "source_file": "inv_002.pdf",
        }
        insert_invoice(inv2, db_path=test_db)

        all_invs = get_all_invoices(db_path=test_db)
        assert len(all_invs) == 2

        open_invs = get_open_invoices(db_path=test_db)
        assert len(open_invs) == 1
        assert open_invs[0]["id"] == "INV-2024-002"
        assert open_invs[0]["paid_on"] is None
