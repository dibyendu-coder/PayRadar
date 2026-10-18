"""Unit tests for chase reminder drafting, tone categorization, and fallback templates."""

from payradar.chase import (
    generate_plain_words_reason,
    get_fallback_template,
    get_or_create_chase_draft,
    get_tone_category,
)
from payradar.db import get_draft, init_db, save_draft


def test_tone_thresholds():
    """Verify tone thresholds: <0.3 friendly, 0.3-0.6 clear and polite, >0.6 firm."""
    tone_low, label_low, _ = get_tone_category(0.15)
    assert tone_low == "friendly_nudge"
    assert "Friendly" in label_low

    tone_mid, label_mid, _ = get_tone_category(0.45)
    assert tone_mid == "clear_and_polite"
    assert "Polite" in label_mid

    tone_mid_border, _, _ = get_tone_category(0.60)
    assert tone_mid_border == "clear_and_polite"

    tone_high, label_high, _ = get_tone_category(0.85)
    assert tone_high == "firm_and_specific"
    assert "Firm" in label_high


def test_plain_words_reason_generation():
    """Verify plain words reason accurately reflects client history without future leakage."""
    open_inv = {
        "id": "INV-TEST-01",
        "client": "Acme Corp",
        "issued_on": "2026-10-01",
        "due_on": "2026-10-31",
        "amount": 5000.0,
        "risk": 0.85,
        "predicted_days_late": 15.0,
    }
    history = [
        # Older invoice - late
        {
            "id": "INV-OLD-1",
            "client": "Acme Corp",
            "issued_on": "2026-06-01",
            "due_on": "2026-07-01",
            "paid_on": "2026-07-20",  # 19 days late (>7)
            "amount": 4000.0,
        },
        # Older invoice - on time
        {
            "id": "INV-OLD-2",
            "client": "Acme Corp",
            "issued_on": "2026-08-01",
            "due_on": "2026-08-31",
            "paid_on": "2026-08-31",  # 0 days late
            "amount": 4500.0,
        },
        # Future invoice relative to issued_on - MUST BE IGNORED
        {
            "id": "INV-FUTURE",
            "client": "Acme Corp",
            "issued_on": "2026-11-01",
            "due_on": "2026-12-01",
            "paid_on": "2026-12-25",
            "amount": 9000.0,
        },
    ]

    reason = generate_plain_words_reason(
        open_inv, history_invoices=history, risk=0.85, predicted_days_late=15.0
    )
    assert "Acme Corp" in reason or "client" in reason.lower()
    assert "1 of their last 2 invoices" in reason
    assert "85.0%" in reason


def test_fallback_template_formatting():
    """Verify fallback templates format properly with client, amount, and due date."""
    inv = {
        "id": "INV-2026-999",
        "client": "Apex Labs",
        "amount": 7500.0,
        "currency": "USD",
        "due_on": "2026-10-15",
    }
    sub_low, body_low = get_fallback_template(inv, "friendly_nudge")
    assert "INV-2026-999" in sub_low
    assert "$7,500.00" in sub_low
    assert "Apex Labs" in body_low
    assert "friendly note" in body_low.lower()

    sub_high, body_high = get_fallback_template(inv, "firm_and_specific")
    assert "Action Required" in sub_high or "Outstanding" in sub_high
    assert "$7,500.00" in body_high
    assert "2026-10-15" in body_high
    assert "outstanding" in body_high.lower()


def test_user_edited_draft_protection(tmp_path):
    """Verify that automated drafting never overwrites a user-edited draft."""
    db_path = tmp_path / "test_chase.db"
    init_db(db_path)

    inv = {
        "id": "INV-PROTECTED",
        "client": "Nova Tech",
        "amount": 3000.0,
        "currency": "USD",
        "due_on": "2026-10-20",
        "risk": 0.5,
        "predicted_days_late": 5.0,
    }

    # 1. Initial automated save
    save_draft(
        invoice_id="INV-PROTECTED",
        subject="Initial Subject",
        body="Initial Body",
        tone="clear_and_polite",
        risk=0.5,
        is_user_edited=False,
        db_path=db_path,
    )

    # 2. User edits the draft in the UI
    user_edited_subject = "My Custom Subject - Please Pay ASAP!"
    user_edited_body = "Hey Alex, following up on our call yesterday."
    saved = save_draft(
        invoice_id="INV-PROTECTED",
        subject=user_edited_subject,
        body=user_edited_body,
        tone="clear_and_polite",
        risk=0.5,
        is_user_edited=True,
        db_path=db_path,
    )
    assert saved is True

    # 3. Automated regeneration attempt without force
    draft = get_or_create_chase_draft(
        inv,
        force_regenerate=False,
        db_path=db_path,
    )

    # Must preserve the user's custom edits!
    assert draft["subject"] == user_edited_subject
    assert draft["body"] == user_edited_body
    assert draft["is_user_edited"] is True

    # 4. Attempting automated save directly without force should be blocked
    blocked = save_draft(
        invoice_id="INV-PROTECTED",
        subject="New Auto Subject",
        body="New Auto Body",
        tone="clear_and_polite",
        risk=0.5,
        is_user_edited=False,
        force=False,
        db_path=db_path,
    )
    assert blocked is False

    # Check DB still has user's text
    row = get_draft("INV-PROTECTED", db_path=db_path)
    assert row["subject"] == user_edited_subject


def test_draft_source_tracking(tmp_path):
    """Verify drafts record source as 'template' or 'gemma'."""
    db_path = tmp_path / "test_source.db"
    init_db(db_path)
    inv = {
        "id": "INV-SOURCE-1",
        "client": "Test Client",
        "amount": 1000.0,
        "currency": "USD",
        "due_on": "2026-10-25",
        "risk": 0.2,
    }
    # Draft with fallback template
    draft_template = get_or_create_chase_draft(inv, use_fallback_only=True, db_path=db_path)
    assert draft_template["source"] == "template"

    row = get_draft("INV-SOURCE-1", db_path=db_path)
    assert row["source"] == "template"
