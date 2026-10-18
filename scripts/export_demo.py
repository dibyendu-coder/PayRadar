"""Export database records and precomputed chase drafts into demo/ for DEMO_MODE=1.

Usage:
  uv run python scripts/export_demo.py [--db payradar.db] [--outdir demo]
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from payradar.chase import get_or_create_chase_draft
from payradar.db import (
    get_all_drafts,
    get_all_invoices,
    get_open_invoices,
    get_scores,
    get_setting,
)
from payradar.forecast import build_weekly_forecast, calculate_expected_pay_dates

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_demo_artifacts(
    db_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    ref_date: datetime.date | None = None,
    use_templates: bool = False,
) -> dict[str, Path]:
    """Export all invoices, scores, drafts, settings, and forecast to JSON files in demo/."""
    if out_dir is None:
        target_dir = PROJECT_ROOT / "demo"
    else:
        target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if ref_date is None:
        ref_date = datetime.date(2026, 10, 15)

    print(f"[EXPORT] Reading database: {db_path or 'default DB'}...")
    invoices = get_all_invoices(db_path=db_path)
    open_invoices = get_open_invoices(db_path=db_path)
    scores = get_scores(db_path=db_path)
    scores_by_id = {s["invoice_id"]: s for s in scores}

    # 1. Export invoices_history.json
    inv_path = target_dir / "invoices_history.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(invoices, f, indent=2)
    print(f"[EXPORT] Saved {len(invoices)} invoices to {inv_path.name}")

    # 2. Export scores.json
    scores_path = target_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"[EXPORT] Saved {len(scores)} scores to {scores_path.name}")

    # 3. Export settings.json
    monthly_costs = get_setting("monthly_costs", default="5000.0", db_path=db_path)
    settings_dict = {"monthly_costs": monthly_costs or "5000.0"}
    settings_path = target_dir / "settings.json"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=2)
    print(f"[EXPORT] Saved settings to {settings_path.name}")

    # 4. Generate & export drafts.json for all open invoices
    # If not explicitly requested to use templates, test if Ollama responds within 2s
    if not use_templates:
        import httpx
        from payradar.extract import DEFAULT_OLLAMA_URL
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{DEFAULT_OLLAMA_URL}/api/tags")
                if res.status_code != 200:
                    use_templates = True
        except Exception:
            use_templates = True

    print(
        f"[EXPORT] Generating precomputed chase drafts for {len(open_invoices)} open invoices "
        f"(mode: {'Fast Templates' if use_templates else 'Live Gemma'})..."
    )
    drafts_list = []

    for inv in open_invoices:
        inv_id = inv["id"]
        score_info = scores_by_id.get(inv_id, {})
        risk = score_info.get("risk", 0.1)
        delay = score_info.get("predicted_days_late", 0.0)

        # Get or create draft
        draft = get_or_create_chase_draft(
            invoice=inv,
            history_invoices=invoices,
            risk=risk,
            predicted_days_late=delay,
            force_regenerate=False,
            use_fallback_only=use_templates,
            db_path=db_path,
        )
        drafts_list.append(draft)

    drafts_path = target_dir / "drafts.json"
    with open(drafts_path, "w", encoding="utf-8") as f:
        json.dump(drafts_list, f, indent=2)
    print(f"[EXPORT] Saved {len(drafts_list)} precomputed drafts to {drafts_path.name}")

    # 5. Export 8-week forecast.json
    scored_invoices = []
    for inv in open_invoices:
        rec = dict(inv)
        score_info = scores_by_id.get(rec["id"], {})
        rec["predicted_risk"] = score_info.get("risk", 0.1)
        rec["predicted_days_late"] = score_info.get("predicted_days_late", 0.0)
        scored_invoices.append(rec)

    enriched = calculate_expected_pay_dates(scored_invoices, today=ref_date)
    forecast_df = build_weekly_forecast(
        enriched,
        monthly_costs=float(monthly_costs or 5000.0),
        today=ref_date,
        weeks_ahead=8,
    )

    forecast_records = forecast_df.to_dict(orient="records")
    forecast_path = target_dir / "forecast.json"
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(forecast_records, f, indent=2)
    print(f"[EXPORT] Saved 8-week forecast to {forecast_path.name}")

    print("\n[SUCCESS] All demo artifacts successfully exported for DEMO_MODE=1.")
    return {
        "invoices": inv_path,
        "scores": scores_path,
        "settings": settings_path,
        "drafts": drafts_path,
        "forecast": forecast_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Export demo JSON artifacts for DEMO_MODE=1")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    parser.add_argument("--outdir", type=str, default=None, help="Target demo directory")
    parser.add_argument("--ref-date", type=str, default="2026-10-15", help="Reference date YYYY-MM-DD")
    parser.add_argument("--templates", action="store_true", help="Force plain templates for precomputed drafts")
    args = parser.parse_args()

    ref_d = datetime.datetime.strptime(args.ref_date, "%Y-%m-%d").date()
    export_demo_artifacts(db_path=args.db, out_dir=args.outdir, ref_date=ref_d, use_templates=args.templates)


if __name__ == "__main__":
    main()

