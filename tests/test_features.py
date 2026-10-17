"""Unit tests for leakage prevention in feature engineering."""

from payradar.features import compute_invoice_features


def test_handcrafted_leakage_invariance():
    """Verify invoice N features are identical whether or not N, N+1, and later exist."""
    # Handcrafted sequence of invoices for client "Acme"
    history = [
        # Invoice 1: Issued Jan 1, Due Jan 31, Paid Jan 31 (0 days late)
        {
            "id": "INV-1",
            "client": "Acme",
            "issued_on": "2024-01-01",
            "due_on": "2024-01-31",
            "amount": 1000.0,
            "terms_days": 30,
            "project_type": "web",
            "paid_on": "2024-01-31",
        },
        # Invoice 2: Issued Feb 1, Due March 2, Paid March 12 (10 days late)
        {
            "id": "INV-2",
            "client": "Acme",
            "issued_on": "2024-02-01",
            "due_on": "2024-03-02",
            "amount": 2000.0,
            "terms_days": 30,
            "project_type": "web",
            "paid_on": "2024-03-12",
        },
        # Invoice 3: Issued March 15, Due April 14, Paid April 24 (10 days late)
        {
            "id": "INV-3",
            "client": "Acme",
            "issued_on": "2024-03-15",
            "due_on": "2024-04-14",
            "amount": 3000.0,
            "terms_days": 30,
            "project_type": "design",
            "paid_on": "2024-04-24",
        },
        # Invoice 4: Issued May 1 (Future)
        {
            "id": "INV-4",
            "client": "Acme",
            "issued_on": "2024-05-01",
            "due_on": "2024-05-31",
            "amount": 4000.0,
            "terms_days": 30,
            "project_type": "consulting",
            "paid_on": "2024-06-15",
        },
    ]

    target_inv = history[2]  # INV-3 (issued 2024-03-15)

    # 1. Compute features when only past invoices exist (INV-1, INV-2)
    past_only = [history[0], history[1]]
    feats_past_only = compute_invoice_features(target_inv, past_only)

    # 2. Compute features when target invoice itself is also in universe
    with_target = [history[0], history[1], history[2]]
    feats_with_target = compute_invoice_features(target_inv, with_target)

    # 3. Compute features when future invoices (INV-4) and open invoices are in universe
    with_future = history + [
        {
            "id": "INV-OPEN",
            "client": "Acme",
            "issued_on": "2024-06-01",
            "due_on": "2024-06-30",
            "amount": 5000.0,
            "terms_days": 30,
            "project_type": "dev",
            "paid_on": None,
        }
    ]
    feats_with_future = compute_invoice_features(target_inv, with_future)

    # All three must produce EXACTLY identical client history features
    assert feats_past_only == feats_with_target == feats_with_future

    # Verify actual computed numbers for INV-3:
    # Prior invoices for Acme issued before 2024-03-15 AND paid by 2024-03-15:
    # INV-1 (paid 2024-01-31, 0 days late, amt 1000)
    # INV-2 (paid 2024-03-12, 10 days late, amt 2000)
    # Count: 2
    # Mean days late: (0 + 10) / 2 = 5.0
    # Median amount: (1000 + 2000) / 2 = 1500.0
    # Amount vs median: 3000 - 1500 = 1500.0
    assert feats_past_only["client_prev_count"] == 2
    assert feats_past_only["client_prev_mean_days_late"] == 5.0
    assert feats_past_only["amount_vs_client_median"] == 1500.0


def test_open_or_unpaid_by_issue_date_does_not_leak():
    """If invoice A was issued earlier but NOT yet paid when invoice B was issued, it cannot leak."""
    inv_a = {
        "id": "INV-A",
        "client": "Beta",
        "issued_on": "2024-01-01",
        "due_on": "2024-01-31",
        "amount": 1000.0,
        "terms_days": 30,
        "project_type": "web",
        # Paid on March 1st (after INV-B was issued!)
        "paid_on": "2024-03-01",
    }
    inv_b = {
        "id": "INV-B",
        "client": "Beta",
        "issued_on": "2024-02-01",
        "due_on": "2024-03-02",
        "amount": 2000.0,
        "terms_days": 30,
        "project_type": "web",
        "paid_on": None,
    }

    # When INV-B was issued on Feb 1, INV-A was still unpaid!
    feats_b = compute_invoice_features(inv_b, [inv_a, inv_b])
    assert feats_b["client_prev_count"] == 0
    assert feats_b["client_prev_mean_days_late"] == 0.0
    assert feats_b["amount_vs_client_median"] == 0.0
