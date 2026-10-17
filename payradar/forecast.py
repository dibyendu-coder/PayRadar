"""8-week rolling cash gap radar and expected cash-in forecasting.

Non-negotiable requirements:
- Expected pay date = due_on + predicted days_late.
- Aggregate expected cash-in by week for the next 8 weeks.
- Compare against monthly_costs setting (prorated weekly).
- Return weekly table with gap week flags (expected_cash_in < weekly_cost).
- Deterministic: "today" is parameterized with default to date.today().
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from payradar.features import parse_date


def calculate_expected_pay_dates(
    open_invoices: list[dict[str, Any]],
    predicted_delays: list[float] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute expected pay date for each open invoice as due_on + predicted days_late."""
    enriched = []
    for idx, inv in enumerate(open_invoices):
        rec = dict(inv)
        due_dt = parse_date(rec.get("due_on"))
        if due_dt is None:
            continue

        if predicted_delays is not None and idx < len(predicted_delays):
            delay = float(predicted_delays[idx])
        else:
            delay = float(rec.get("predicted_days_late", 0.0))

        rec["predicted_days_late"] = delay
        # Expected pay date = due_on + delay
        expected_dt = due_dt + datetime.timedelta(days=int(round(delay)))
        rec["expected_pay_date"] = expected_dt.strftime("%Y-%m-%d")
        enriched.append(rec)

    return enriched


def build_weekly_forecast(
    open_invoices: list[dict[str, Any]],
    monthly_costs: float = 5000.0,
    today: datetime.date | None = None,
    weeks_ahead: int = 8,
) -> pd.DataFrame:
    """Build an 8-week cash flow forecast comparing expected cash-in against weekly costs.

    Weekly cost threshold is prorated as (monthly_costs * 12 / 52).
    A week is flagged as a gap week if expected_cash_in < weekly_cost.
    """
    if today is None:
        today = datetime.date.today()

    weekly_cost = round(float(monthly_costs) * 12.0 / 52.0, 2)

    # Ensure expected_pay_date is populated
    processed_invoices = []
    for inv in open_invoices:
        if "expected_pay_date" not in inv:
            due_dt = parse_date(inv.get("due_on"))
            delay = float(inv.get("predicted_days_late", 0.0))
            if due_dt is not None:
                exp_dt = due_dt + datetime.timedelta(days=int(round(delay)))
                inv["expected_pay_date"] = exp_dt.strftime("%Y-%m-%d")
        processed_invoices.append(inv)

    weeks = []
    for w in range(weeks_ahead):
        w_start = today + datetime.timedelta(days=w * 7)
        w_end = w_start + datetime.timedelta(days=6)

        # Invoices falling into this week
        matching_invoices = []
        cash_in = 0.0

        for inv in processed_invoices:
            exp_d = parse_date(inv.get("expected_pay_date"))
            if exp_d is None:
                continue

            # Invoices expected prior to today roll into Week 1 as immediately due/overdue
            if w == 0 and exp_d < today:
                matching_invoices.append(inv)
                cash_in += float(inv.get("amount", 0.0))
            elif w_start <= exp_d <= w_end:
                matching_invoices.append(inv)
                cash_in += float(inv.get("amount", 0.0))

        cash_in = round(cash_in, 2)
        net_flow = round(cash_in - weekly_cost, 2)
        is_gap = bool(cash_in < weekly_cost)

        weeks.append({
            "week_num": w + 1,
            "start_date": w_start.strftime("%Y-%m-%d"),
            "end_date": w_end.strftime("%Y-%m-%d"),
            "expected_cash_in": cash_in,
            "weekly_cost": weekly_cost,
            "net_cash_flow": net_flow,
            "is_gap_week": is_gap,
            "invoice_count": len(matching_invoices),
            "invoice_ids": [inv.get("id", "") for inv in matching_invoices],
        })

    return pd.DataFrame(weeks)
