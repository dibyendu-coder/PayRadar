"""Feature engineering for invoice late-payment risk and delay prediction.

Non-negotiable rule:
Client history features (client_prev_count, client_prev_mean_days_late, amount_vs_client_median)
may use ONLY invoices issued strictly before the current invoice's issue date AND already paid
by its issue date. Open invoices and future invoices must NEVER leak into history.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_amount",
    "terms_days",
    "issue_month",
    "issue_dow",
    "client",
    "project_type",
    "client_prev_count",
    "client_prev_mean_days_late",
    "amount_vs_client_median",
]

CATEGORICAL_COLUMNS = ["client", "project_type"]


def parse_date(d: Any) -> datetime.date | None:
    """Safely parse date string or datetime into date object."""
    if d is None or pd.isna(d):
        return None
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.date.fromisoformat(d.strip())
        except ValueError:
            return pd.to_datetime(d).date()
    return None


def compute_invoice_features(
    invoice: dict[str, Any] | pd.Series,
    all_invoices: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Compute features for a single invoice given the universe of available invoices.

    Strict leakage prevention:
    - Finds all other invoices with the same client.
    - Filters to only those with issued_on < current_issued_on AND paid_on <= current_issued_on.
    """
    curr_issued = parse_date(invoice.get("issued_on"))
    if curr_issued is None:
        raise ValueError("Invoice must have a valid 'issued_on' date")

    curr_amount = float(invoice.get("amount", 0.0))
    curr_terms = int(invoice.get("terms_days", 30))
    curr_client = str(invoice.get("client", "Unknown"))
    curr_project = str(invoice.get("project_type", "general"))

    # Convert all_invoices to list of dicts if DataFrame
    if isinstance(all_invoices, pd.DataFrame):
        records = all_invoices.to_dict(orient="records")
    else:
        records = all_invoices

    # Identify valid prior paid invoices for this client
    prior_days_late = []
    prior_amounts = []

    curr_id = invoice.get("id")

    for rec in records:
        # Skip same invoice if present
        if curr_id is not None and rec.get("id") == curr_id:
            continue

        if str(rec.get("client")) != curr_client:
            continue

        rec_issued = parse_date(rec.get("issued_on"))
        rec_paid = parse_date(rec.get("paid_on"))
        rec_due = parse_date(rec.get("due_on"))

        # Must be issued strictly before current invoice
        if rec_issued is None or rec_issued >= curr_issued:
            continue

        # Must be paid by the current invoice's issue date (no look-ahead)
        if rec_paid is None or rec_paid > curr_issued:
            continue

        if rec_due is not None:
            delay = (rec_paid - rec_due).days
            prior_days_late.append(delay)
            prior_amounts.append(float(rec.get("amount", 0.0)))

    # Compute history stats
    prev_count = len(prior_days_late)
    prev_mean_late = float(np.mean(prior_days_late)) if prev_count > 0 else 0.0
    prev_median_amt = float(np.median(prior_amounts)) if prev_count > 0 else 0.0
    amt_vs_median = (curr_amount - prev_median_amt) if prev_count > 0 else 0.0

    features = {
        "log_amount": float(np.log1p(max(0.0, curr_amount))),
        "terms_days": curr_terms,
        "issue_month": curr_issued.month,
        "issue_dow": curr_issued.weekday(),  # 0 = Monday, 6 = Sunday
        "client": curr_client,
        "project_type": curr_project,
        "client_prev_count": prev_count,
        "client_prev_mean_days_late": prev_mean_late,
        "amount_vs_client_median": amt_vs_median,
    }

    return features


def build_feature_dataset(
    invoices: list[dict[str, Any]] | pd.DataFrame,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build a complete feature DataFrame for a list or table of invoices.

    For each invoice in the collection, client history features are strictly
    computed using only earlier paid invoices in the same collection.
    """
    if isinstance(invoices, pd.DataFrame):
        df_raw = invoices.copy()
        raw_records = invoices.to_dict(orient="records")
    else:
        df_raw = pd.DataFrame(invoices)
        raw_records = invoices

    rows = []
    for rec in raw_records:
        feat = compute_invoice_features(rec, raw_records)
        feat["id"] = rec.get("id")
        feat["issued_on"] = rec.get("issued_on")
        feat["due_on"] = rec.get("due_on")
        feat["amount"] = float(rec.get("amount", 0.0))
        feat["paid_on"] = rec.get("paid_on")

        if include_targets:
            paid_d = parse_date(rec.get("paid_on"))
            due_d = parse_date(rec.get("due_on"))
            if paid_d is not None and due_d is not None:
                days_late = (paid_d - due_d).days
                feat["days_late"] = days_late
                feat["is_late_7d"] = 1 if days_late > 7 else 0
            else:
                feat["days_late"] = np.nan
                feat["is_late_7d"] = np.nan

        rows.append(feat)

    df_feats = pd.DataFrame(rows)
    return df_feats
