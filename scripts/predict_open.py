"""Score open invoices and compute rolling 8-week cash gap forecast.

Usage:
  uv run python scripts/predict_open.py [--today YYYY-MM-DD] [--db path/to/db]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from payradar.db import get_all_invoices, get_open_invoices, get_setting
from payradar.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_dataset,
    compute_invoice_features,
    parse_date,
)
from payradar.forecast import build_weekly_forecast, calculate_expected_pay_dates
from payradar.model import (
    LatePaymentClassifier,
    PaymentDelayRegressor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_risk_tier(risk_prob: float) -> str:
    """Categorize late payment risk score."""
    if risk_prob >= 0.60:
        return "HIGH RISK (>60%)"
    if risk_prob >= 0.30:
        return "MEDIUM (30-60%)"
    return "LOW (<30%)"


def score_open_invoices(
    today: datetime.date | None = None,
    db_path: str | Path | None = None,
) -> tuple[list[dict], Any]:
    """Score all open invoices and compute 8-week cash forecast."""
    if today is None:
        today = datetime.date.today()

    all_invoices = get_all_invoices(db_path=db_path)
    open_invoices = get_open_invoices(db_path=db_path)

    if not all_invoices or not open_invoices:
        # Fallback to demo dataset
        demo_json = PROJECT_ROOT / "demo" / "invoices_history.json"
        if demo_json.exists():
            with open(demo_json, "r", encoding="utf-8") as f:
                all_invoices = json.load(f)
            open_invoices = [inv for inv in all_invoices if inv.get("paid_on") is None]

    if not open_invoices:
        print("[INFO] No open invoices found in database.")
        sys.exit(0)

    # Historical paid invoices for model fitting
    paid_invoices = [inv for inv in all_invoices if inv.get("paid_on") is not None]

    # Build features for training
    df_train = build_feature_dataset(paid_invoices, include_targets=True)
    X_train = df_train[FEATURE_COLUMNS].copy()
    y_train_clf = df_train["is_late_7d"].values.astype(int)
    y_train_reg = df_train["days_late"].values.astype(float)

    # Compute features for open invoices (strict leakage prevention against all invoices)
    open_feature_rows = []
    for inv in open_invoices:
        feat = compute_invoice_features(inv, all_invoices)
        open_feature_rows.append(feat)

    import pandas as pd

    X_open = pd.DataFrame(open_feature_rows)[FEATURE_COLUMNS]

    # Predict risk and delay
    tabpfn_token = os.environ.get("TABPFN_TOKEN")
    model_name = "TabPFN-3.5"

    if tabpfn_token:
        try:
            device = os.environ.get("DEVICE", "cpu")
            clf = LatePaymentClassifier(device=device)
            clf.fit(X_train, y_train_clf)
            reg = PaymentDelayRegressor(device=device)
            reg.fit(X_train, y_train_reg)

            risks = clf.predict_risk(X_open)
            delays = reg.predict(X_open)
        except Exception as e:
            print(f"[WARN] TabPFN inference failed: {e}. Falling back to HistGradientBoosting.")
            tabpfn_token = None

    if not tabpfn_token:
        model_name = "HistGradientBoosting (Baseline)"
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_hgb = X_train.copy()
        X_open_hgb = X_open.copy()
        X_train_hgb[CATEGORICAL_COLUMNS] = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
        X_open_hgb[CATEGORICAL_COLUMNS] = encoder.transform(X_open[CATEGORICAL_COLUMNS])

        cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]
        hgb_clf = HistGradientBoostingClassifier(categorical_features=cat_indices, random_state=42)
        hgb_clf.fit(X_train_hgb, y_train_clf)
        hgb_reg = HistGradientBoostingRegressor(categorical_features=cat_indices, random_state=42)
        hgb_reg.fit(X_train_hgb, y_train_reg)

        risks = hgb_clf.predict_proba(X_open_hgb)[:, 1]
        delays = hgb_reg.predict(X_open_hgb)

    # Attach predictions
    scored = []
    for idx, inv in enumerate(open_invoices):
        rec = dict(inv)
        rec["predicted_risk"] = float(risks[idx])
        rec["predicted_days_late"] = float(delays[idx])
        scored.append(rec)

    # Sort open invoices by risk descending
    scored.sort(key=lambda x: x["predicted_risk"], reverse=True)

    # Compute expected pay dates
    scored_with_dates = calculate_expected_pay_dates(scored)

    # Build weekly forecast
    cost_setting = get_setting("monthly_costs", default="5000.0", db_path=db_path)
    monthly_costs = float(cost_setting or 5000.0)

    forecast_df = build_weekly_forecast(
        scored_with_dates,
        monthly_costs=monthly_costs,
        today=today,
        weeks_ahead=8,
    )

    # Print results
    print("=" * 105)
    print("PAYRADAR OPEN INVOICE RISK RADAR [SYNTHETIC DEMO DATA]")
    print(f"Scoring Engine: {model_name} | Reference Date (Today): {today.strftime('%Y-%m-%d')}")
    print(f"Monthly Operating Cost: ${monthly_costs:,.2f} (${monthly_costs * 12 / 52:,.2f}/week)")
    print("=" * 105)

    print(f"\n1. OPEN INVOICES RANKED BY LATE-PAYMENT RISK (Total: {len(scored)} open invoices):")
    print("-" * 105)
    print(
        f"{'Invoice ID':<14} | {'Client':<26} | {'Amount':<10} | {'Due Date':<11} | {'Delay':<8} | {'Pay Date':<11} | {'Risk':<7} | {'Tier'}"
    )
    print("-" * 105)

    for inv in scored_with_dates:
        amt_str = f"${float(inv['amount']):,.2f}"
        delay_str = f"{float(inv['predicted_days_late']):+.1f}d"
        risk_pct = f"{float(inv['predicted_risk']) * 100:.1f}%"
        tier_str = get_risk_tier(float(inv["predicted_risk"]))
        print(
            f"{inv['id']:<14} | {inv['client']:<26} | {amt_str:<10} | {inv['due_on']:<11} | {delay_str:<8} | {inv['expected_pay_date']:<11} | {risk_pct:<7} | {tier_str}"
        )

    print("-" * 105)

    print("\n2. 8-WEEK CASH FLOW & GAP RADAR FORECAST:")
    print("-" * 105)
    print(
        f"{'Week':<6} | {'Period':<23} | {'Cash-In':<11} | {'Weekly Cost':<12} | {'Net Cash Flow':<14} | {'Status'}"
    )
    print("-" * 105)

    total_in = 0.0
    for _, row in forecast_df.iterrows():
        period = f"{row['start_date']} -> {row['end_date']}"
        cash_str = f"${float(row['expected_cash_in']):,.2f}"
        cost_str = f"${float(row['weekly_cost']):,.2f}"
        net_str = f"${float(row['net_cash_flow']):+,.2f}"
        status = "[!] CASH GAP WEEK" if row["is_gap_week"] else "[OK] Covered"
        total_in += float(row["expected_cash_in"])
        print(
            f"Wk {int(row['week_num']):<3} | {period:<23} | {cash_str:<11} | {cost_str:<12} | {net_str:<14} | {status}"
        )

    total_costs = float(forecast_df["weekly_cost"].sum())
    total_net = total_in - total_costs
    gap_count = int(forecast_df["is_gap_week"].sum())

    print("-" * 105)
    print(
        f"8-Week Cumulative: Expected In: ${total_in:,.2f} | Costs: ${total_costs:,.2f} | Net: ${total_net:+,.2f} | Gap Weeks: {gap_count}/8"
    )
    print("=" * 105 + "\n")

    return scored_with_dates, forecast_df


def main():
    parser = argparse.ArgumentParser(description="Predict late payment risk on open invoices")
    parser.add_argument(
        "--today", type=str, default=None, help="Evaluation reference date (YYYY-MM-DD)"
    )
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    ref_date = parse_date(args.today) if args.today else datetime.date(2026, 10, 15)
    score_open_invoices(today=ref_date, db_path=args.db)


if __name__ == "__main__":
    main()
