"""Opt-in live integration test for Gemma inference via Ollama.

Skipped by default unless OLLAMA_LIVE=1 is explicitly set in environment.
Run via:
  OLLAMA_LIVE=1 uv run pytest -m live
"""

from __future__ import annotations

import os

import pytest

from payradar.chase import get_or_create_chase_draft
from payradar.extract import get_available_models

OLLAMA_LIVE_ENABLED = os.environ.get("OLLAMA_LIVE", "0").strip() in ("1", "true", "True")


@pytest.mark.live
@pytest.mark.skipif(
    not OLLAMA_LIVE_ENABLED,
    reason="Live Ollama tests are opt-in. Set OLLAMA_LIVE=1 to execute.",
)
def test_live_gemma_chase_draft_generation(tmp_path):
    """Verify live Gemma 4 response returns valid subject, body, and source='gemma'."""
    models = get_available_models()
    assert len(models) > 0, "No Ollama models available on localhost:11434"

    db_path = tmp_path / "test_live_draft.db"

    sample_invoice = {
        "id": "INV-LIVE-001",
        "client": "Vanguard Media",
        "amount": 8511.39,
        "currency": "USD",
        "due_on": "2026-11-09",
        "issued_on": "2026-10-10",
        "risk": 0.85,
        "predicted_days_late": 20.0,
    }

    draft = get_or_create_chase_draft(
        invoice=sample_invoice,
        risk=0.85,
        predicted_days_late=20.0,
        force_regenerate=True,
        model_name=os.environ.get("GEMMA_MODEL", "gemma4:e2b"),
        db_path=db_path,
    )

    # Assert real model response returned
    assert draft["is_fallback"] is False, "Draft unexpectedly fell back to plain template"
    assert draft["source"] == "gemma"
    assert draft["model"] == "Gemma"
    assert isinstance(draft["subject"], str) and len(draft["subject"]) > 5
    assert isinstance(draft["body"], str) and len(draft["body"]) > 30

    # Assert client or invoice reference is reflected in generated text
    combined_text = f"{draft['subject']} {draft['body']}".lower()
    assert (
        "inv-live-001" in combined_text
        or "vanguard" in combined_text
        or "8,511" in combined_text
        or "8511" in combined_text
        or "november 9" in combined_text
        or "payment" in combined_text
    )
