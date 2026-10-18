"""Unit tests for DEMO_MODE=1 precomputed artifact loading and behavior."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_demo_directory_artifacts_exist():
    """Verify all required precomputed JSON artifacts exist in demo/ for DEMO_MODE=1."""
    demo_dir = PROJECT_ROOT / "demo"
    assert (demo_dir / "invoices_history.json").exists()
    assert (demo_dir / "scores.json").exists()
    assert (demo_dir / "drafts.json").exists()
    assert (demo_dir / "forecast.json").exists()
    assert (demo_dir / "settings.json").exists()


def test_demo_artifacts_content_validity():
    """Verify demo files contain non-empty valid synthetic data."""
    demo_dir = PROJECT_ROOT / "demo"

    with open(demo_dir / "invoices_history.json", "r", encoding="utf-8") as f:
        invoices = json.load(f)
    assert len(invoices) >= 300
    open_invoices = [inv for inv in invoices if inv.get("paid_on") is None]
    assert len(open_invoices) > 0

    with open(demo_dir / "scores.json", "r", encoding="utf-8") as f:
        scores = json.load(f)
    assert len(scores) == len(open_invoices)
    assert "risk" in scores[0]
    assert "predicted_days_late" in scores[0]

    with open(demo_dir / "drafts.json", "r", encoding="utf-8") as f:
        drafts = json.load(f)
    assert len(drafts) == len(open_invoices)
    for d in drafts:
        assert "subject" in d
        assert "body" in d
        assert "tone" in d

    with open(demo_dir / "forecast.json", "r", encoding="utf-8") as f:
        forecast = json.load(f)
    assert len(forecast) == 8
    assert "expected_cash_in" in forecast[0]
    assert "weekly_cost" in forecast[0]
