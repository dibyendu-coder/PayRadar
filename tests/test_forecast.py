"""Unit tests for 8-week cash flow radar and gap week forecasting."""

import datetime

from payradar.forecast import build_weekly_forecast, calculate_expected_pay_dates


def test_forecast_deterministic_weekly_totals_and_gap_flags():
    """Verify weekly cash totals and gap flags with fixed today and fixed predictions."""
    fixed_today = datetime.date(2026, 10, 1)  # Thursday
    monthly_costs = 5200.0  # Weekly cost = 5200 * 12 / 52 = 1200.00

    # Handcrafted open invoices:
    open_invoices = [
        # Invoice 1: Due 2026-10-02, Delay = 0 days -> Expected 2026-10-02 (Week 1: Oct 1 - Oct 7)
        {"id": "INV-W1", "amount": 1500.0, "due_on": "2026-10-02", "predicted_days_late": 0.0},
        # Invoice 2: Due 2026-10-05, Delay = 5 days -> Expected 2026-10-10 (Week 2: Oct 8 - Oct 14)
        {"id": "INV-W2", "amount": 800.0, "due_on": "2026-10-05", "predicted_days_late": 5.0},
        # Invoice 3: Due 2026-10-10, Delay = 10 days -> Expected 2026-10-20 (Week 3: Oct 15 - Oct 21)
        {"id": "INV-W3", "amount": 2500.0, "due_on": "2026-10-10", "predicted_days_late": 10.0},
    ]

    forecast_df = build_weekly_forecast(
        open_invoices,
        monthly_costs=monthly_costs,
        today=fixed_today,
        weeks_ahead=8,
    )

    assert len(forecast_df) == 8

    # Week 1 (Oct 1 - Oct 7): Cash in = 1500.0, Weekly cost = 1200.0 -> Gap = False
    row1 = forecast_df.iloc[0]
    assert row1["week_num"] == 1
    assert row1["start_date"] == "2026-10-01"
    assert row1["expected_cash_in"] == 1500.0
    assert row1["weekly_cost"] == 1200.0
    assert not row1["is_gap_week"]

    # Week 2 (Oct 8 - Oct 14): Cash in = 800.0, Weekly cost = 1200.0 -> Gap = True (Shortfall!)
    row2 = forecast_df.iloc[1]
    assert row2["week_num"] == 2
    assert row2["expected_cash_in"] == 800.0
    assert row2["net_cash_flow"] == -400.0
    assert bool(row2["is_gap_week"]) is True

    # Week 3 (Oct 15 - Oct 21): Cash in = 2500.0 -> Gap = False
    row3 = forecast_df.iloc[2]
    assert row3["expected_cash_in"] == 2500.0
    assert not row3["is_gap_week"]

    # Week 4 - Week 8: No cash in (0.0) -> Gap = True
    for w in range(3, 8):
        row = forecast_df.iloc[w]
        assert row["expected_cash_in"] == 0.0
        assert bool(row["is_gap_week"]) is True


def test_calculate_expected_pay_dates():
    invoices = [
        {"id": "INV-1", "due_on": "2026-09-10"},
        {"id": "INV-2", "due_on": "2026-09-20"},
    ]
    delays = [5.2, -2.0]
    res = calculate_expected_pay_dates(
        invoices, predicted_delays=delays, today=datetime.date(2026, 9, 1)
    )

    assert res[0]["raw_expected_pay_date"] == "2026-09-15"
    assert res[0]["expected_pay_date"] == "2026-09-15"
    assert res[1]["expected_pay_date"] == "2026-09-18"


def test_overdue_invoice_clamped_to_today():
    """Verify that an overdue open invoice has its expected pay date clamped to today."""
    fixed_today = datetime.date(2026, 10, 15)

    # An invoice due on Sept 1st with predicted delay 5 days -> raw pay date Sept 6th (in the past relative to Oct 15)
    invoices = [
        {"id": "INV-PAST-DUE", "due_on": "2026-09-01", "predicted_days_late": 5.0},
        {"id": "INV-FUTURE", "due_on": "2026-10-25", "predicted_days_late": 2.0},
    ]

    res = calculate_expected_pay_dates(invoices, today=fixed_today)

    # Clamped row
    assert res[0]["raw_expected_pay_date"] == "2026-09-06"
    assert res[0]["expected_pay_date"] == "2026-10-15"
    assert res[0]["pay_status"] == "overdue, expected any day"
    assert res[0]["is_overdue"] is True

    # Future row
    assert res[1]["raw_expected_pay_date"] == "2026-10-27"
    assert res[1]["expected_pay_date"] == "2026-10-27"
    assert res[1]["pay_status"] == "on schedule"
    assert res[1]["is_overdue"] is False


def test_forecast_statement_based_on_invoices_issued_so_far():
    """Verify forecast table header and returned text state 'based on invoices issued so far'."""
    from payradar.forecast import format_forecast_text

    invoices = [
        {"id": "INV-1", "amount": 2000.0, "due_on": "2026-10-20", "predicted_days_late": 0.0}
    ]
    df = build_weekly_forecast(invoices, monthly_costs=4000.0, today=datetime.date(2026, 10, 15))

    required_phrase = "based on invoices issued so far"
    assert required_phrase in df.attrs["table_header"]
    assert required_phrase in df.attrs["text"]
    assert required_phrase in format_forecast_text(df)
