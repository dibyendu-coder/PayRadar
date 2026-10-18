"""Live test for chase drafting with Gemma 4 on real open invoices.

Runs live generation with GEMMA_MODEL=gemma4:e2b for:
1. High-risk invoice (>60% risk, firm tone)
2. Low-risk invoice (<30% risk, friendly tone)

Prints the drafts, elapsed seconds, and whether each came from Gemma or template.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["GEMMA_MODEL"] = "gemma4:e2b"

from payradar.chase import get_or_create_chase_draft
from payradar.db import get_all_invoices, get_scored_open_invoices

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_live_test() -> None:
    all_invoices = get_all_invoices()
    open_invoices = get_scored_open_invoices()

    if not open_invoices:
        print("[ERROR] No open invoices found in database.")
        sys.exit(1)

    # Find highest risk invoice and lowest risk invoice
    high_risk_inv = max(open_invoices, key=lambda x: float(x.get("risk", 0.0) or 0.0))
    low_risk_inv = min(open_invoices, key=lambda x: float(x.get("risk", 0.0) or 0.0))

    test_targets = [
        ("HIGH-RISK INVOICE", high_risk_inv),
        ("LOW-RISK INVOICE", low_risk_inv),
    ]

    print("=" * 80)
    print("LIVE CHASE DRAFT GENERATION TEST (Model: gemma4:e2b via Ollama)")
    print("=" * 80)

    for label, inv in test_targets:
        inv_id = inv["id"]
        client = inv["client"]
        amount = float(inv["amount"])
        risk = float(inv.get("risk", 0.0) or 0.0)
        delay = float(inv.get("predicted_days_late", 0.0) or 0.0)

        print(f"\n>>> Generating Draft for {label}:")
        print(f"    Invoice ID: {inv_id} | Client: {client} | Amount: ${amount:,.2f}")
        print(f"    Risk Score: {risk * 100:.1f}% | Expected Delay: {delay:+.1f} days")

        t0 = time.time()
        draft = get_or_create_chase_draft(
            invoice=inv,
            history_invoices=all_invoices,
            risk=risk,
            predicted_days_late=delay,
            force_regenerate=True,
            model_name="gemma4:e2b",
        )
        elapsed_sec = time.time() - t0

        source = "Template" if draft.get("is_fallback") else "Gemma"
        model_name = draft.get("model", "Unknown")

        print(f"    Elapsed Time: {elapsed_sec:.2f} seconds")
        print(f"    Draft Source: {source} (Model: {model_name})")
        print(f"    Tone Assigned: {draft.get('tone_label')}")
        print(f"    Subject: {draft.get('subject')}")
        print("    Body:")
        for line in draft.get("body", "").splitlines():
            print(f"      {line}")
        print("-" * 80)

    print("\n[SUCCESS] Live chase generation test complete.\n")


if __name__ == "__main__":
    run_live_test()
