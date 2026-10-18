"""Chase reminder email drafting and risk rationale generation.

Non-negotiable requirements:
- Gemma drafts a short reminder email from invoice fields and its risk score.
- Tone by risk:
  - < 0.3: friendly nudge
  - 0.3 to 0.6: clear and polite
  - > 0.6: firm and specific about due date and amount
- Target hardware is 4 GB VRAM: short context, fast local inference.
- Nothing is sent anywhere automatically: drafts are shown for the user to copy.
- Save drafts to SQLite DB, never overwrite a user-edited draft.
- Fall back to a plain template if Gemma times out or errors.
- Plain-words rationale for late payment risk (e.g. 'this client has paid late on 6 of their last 8 invoices').
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from payradar.db import get_draft, save_draft

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHASE_TIMEOUT_SECONDS = float(os.environ.get("CHASE_TIMEOUT", "90.0"))



def get_tone_category(risk: float) -> tuple[str, str, str]:
    """Return tone key, display label, and drafting instructions based on late-payment risk.

    Risk thresholds:
    - below 0.3: friendly nudge
    - 0.3 to 0.6: clear and polite
    - above 0.6: firm and specific
    """
    if risk < 0.30:
        return (
            "friendly_nudge",
            "Friendly Nudge (<30% risk)",
            "Friendly, warm, and casual check-in. Thank the client for the collaboration and remind them gently of the upcoming or recent invoice date.",
        )
    elif risk <= 0.60:
        return (
            "clear_and_polite",
            "Clear & Polite (30-60% risk)",
            "Professional, clear, and polite payment reminder. Note the invoice number, due date, and amount, and inquire if payment is on track with their accounts department.",
        )
    else:
        return (
            "firm_and_specific",
            "Firm & Specific (>60% risk)",
            "Firm, direct, and urgent tone. Explicitly state the exact overdue or due date, exact outstanding amount, and request immediate confirmation of the payment transfer execution date today.",
        )


def generate_plain_words_reason(
    invoice: dict[str, Any],
    history_invoices: list[dict[str, Any]] | None = None,
    risk: float | None = None,
    predicted_days_late: float | None = None,
) -> str:
    """Generate a human-readable explanation of why this invoice was assigned its risk score.

    Strictly follows leakage prevention: evaluates only invoices issued prior to this invoice.
    """
    client_name = invoice.get("client", "Client")
    issue_date = invoice.get("issued_on", "9999-99-99")

    # Filter previous paid invoices for this client issued before this invoice
    prev_invoices = []
    if history_invoices:
        for inv in history_invoices:
            if (
                inv.get("client") == client_name
                and inv.get("issued_on", "") < issue_date
                and inv.get("paid_on") is not None
            ):
                prev_invoices.append(inv)

    n_prev = len(prev_invoices)
    if n_prev == 0:
        base_reason = "First-time client in your records with no prior payment history."
    else:
        late_count = 0
        delays = []
        for inv in prev_invoices:
            # Check days late
            due_on = inv.get("due_on", "")
            paid_on = inv.get("paid_on", "")
            if due_on and paid_on:
                try:
                    d_due = datetime.strptime(due_on, "%Y-%m-%d")
                    d_paid = datetime.strptime(paid_on, "%Y-%m-%d")
                    diff = (d_paid - d_due).days
                    delays.append(diff)
                    if diff > 7:
                        late_count += 1
                except Exception:
                    pass

        avg_delay = sum(delays) / len(delays) if delays else 0.0

        if late_count > 0:
            base_reason = (
                f"This client has paid late on {late_count} of their last {n_prev} invoices "
                f"(averaging {avg_delay:+.1f} days delay)."
            )
        else:
            base_reason = f"Client has paid all {n_prev} prior invoices on time."

    # Append model prediction context
    risk_val = risk if risk is not None else float(invoice.get("risk", 0.0) or 0.0)
    delay_val = (
        predicted_days_late
        if predicted_days_late is not None
        else float(invoice.get("predicted_days_late", 0.0) or 0.0)
    )

    if risk_val >= 0.60:
        prediction_text = f" TabPFN model flags high risk ({risk_val * 100:.1f}%) with ~{delay_val:+.1f} days expected delay."
    elif risk_val >= 0.30:
        prediction_text = f" TabPFN model estimates moderate delay risk ({risk_val * 100:.1f}%, ~{delay_val:+.1f} days delay)."
    else:
        prediction_text = (
            f" TabPFN model predicts payment on schedule ({risk_val * 100:.1f}% risk)."
        )

    return base_reason + prediction_text


def get_fallback_template(invoice: dict[str, Any], tone_key: str) -> tuple[str, str]:
    """Generate high-quality deterministic email template when Gemma is unavailable or times out."""
    inv_id = invoice.get("id", "INV")
    client = invoice.get("client", "Client")
    amount = float(invoice.get("amount", 0.0))
    currency = invoice.get("currency", "USD")
    due_on = invoice.get("due_on", "due date")

    formatted_amount = f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"

    if tone_key == "friendly_nudge":
        subject = f"Friendly check-in: Invoice {inv_id} ({formatted_amount})"
        body = (
            f"Hi {client} team,\n\n"
            f"Hope you are having a productive week!\n\n"
            f"Just sending a quick, friendly note regarding invoice {inv_id} for {formatted_amount}, "
            f"which is due on {due_on}.\n\n"
            f"Please let me know if you need any additional project details or another copy of the invoice. "
            f"Thank you for the wonderful collaboration!\n\n"
            f"Best regards,\n[Your Name]"
        )
    elif tone_key == "clear_and_polite":
        subject = f"Invoice {inv_id} due on {due_on} ({formatted_amount})"
        body = (
            f"Hello {client} Accounts Team,\n\n"
            f"I hope this note finds you well.\n\n"
            f"This is a gentle reminder that invoice {inv_id} in the amount of {formatted_amount} "
            f"is scheduled for payment on {due_on}.\n\n"
            f"Could you please confirm if this invoice is approved and on track for disbursement?\n\n"
            f"Thank you for your prompt assistance,\n[Your Name]"
        )
    else:  # firm_and_specific
        subject = f"Action Required: Outstanding Invoice {inv_id} ({formatted_amount}) due {due_on}"
        body = (
            f"Dear {client} Accounts Payable,\n\n"
            f"I am writing regarding invoice {inv_id} for {formatted_amount}, which was due on {due_on}.\n\n"
            f"According to our records, payment for this invoice remains outstanding. "
            f"Please review the attached invoice details and confirm today when the transfer will be completed.\n\n"
            f"If payment has already been sent, please reply with the remittance confirmation so we can update our ledger.\n\n"
            f"Sincerely,\n[Your Name]"
        )

    return subject, body


def draft_chase_email_with_gemma(
    invoice: dict[str, Any],
    risk: float,
    reason: str,
    tone_key: str,
    tone_instruction: str,
    model_name: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = CHASE_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """Call local Gemma to draft a tailored reminder email using short context."""
    import httpx

    from payradar.extract import resolve_model

    chosen_model = resolve_model(model_name, base_url=base_url)

    inv_id = invoice.get("id", "INV")
    client = invoice.get("client", "Client")
    amount = float(invoice.get("amount", 0.0))
    currency = invoice.get("currency", "USD")
    due_on = invoice.get("due_on", "due date")
    formatted_amount = f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"

    system_prompt = (
        "You are an expert freelance finance assistant drafting invoice chaser emails.\n"
        "Generate a concise, professional reminder email.\n"
        "Always output exactly two labeled sections:\n"
        "SUBJECT: <single line subject>\n"
        "BODY:\n"
        "<email body ending with [Your Name]>\n"
        "Do not include markdown code fences or conversational greetings."
    )

    user_prompt = (
        f"Draft an invoice reminder email:\n"
        f"- Client: {client}\n"
        f"- Invoice ID: {inv_id}\n"
        f"- Amount: {formatted_amount}\n"
        f"- Due Date: {due_on}\n"
        f"- Risk Score: {risk * 100:.1f}%\n"
        f"- Background Reason: {reason}\n"
        f"- Desired Tone: {tone_instruction}\n"
    )

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
        },
    }

    with httpx.Client(timeout=timeout) as client_http:
        resp = client_http.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("message", {}).get("content", "").strip()

    # Parse SUBJECT and BODY
    subject_match = re.search(r"SUBJECT:\s*(.+)", raw_text, re.IGNORECASE)
    body_match = re.search(r"BODY:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)

    if subject_match and body_match:
        subject = subject_match.group(1).strip()
        body = body_match.group(1).strip()
    elif subject_match:
        subject = subject_match.group(1).strip()
        # Rest is body
        body = raw_text.split(subject)[-1].replace("BODY:", "").strip()
    else:
        # Fallback to splitting lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            subject = lines[0].replace("Subject:", "").strip()
            body = "\n\n".join(lines[1:])
        else:
            raise ValueError("Empty response from Gemma")

    return subject, body


def get_or_create_chase_draft(
    invoice: dict[str, Any],
    history_invoices: list[dict[str, Any]] | None = None,
    risk: float | None = None,
    predicted_days_late: float | None = None,
    force_regenerate: bool = False,
    use_fallback_only: bool = False,
    model_name: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve existing draft or generate a new draft for the given invoice.

    Protects user edits: if is_user_edited is True and force_regenerate is False,
    the existing user draft is strictly preserved.
    """
    inv_id = invoice["id"]
    existing = get_draft(inv_id, db_path=db_path)

    # Calculate risk and delay
    risk_val = float(risk if risk is not None else invoice.get("risk", 0.0) or 0.0)
    delay_val = float(
        predicted_days_late
        if predicted_days_late is not None
        else invoice.get("predicted_days_late", 0.0) or 0.0
    )
    tone_key, tone_label, tone_instruction = get_tone_category(risk_val)
    reason = generate_plain_words_reason(
        invoice,
        history_invoices=history_invoices,
        risk=risk_val,
        predicted_days_late=delay_val,
    )

    if existing and not force_regenerate:
        # Return existing draft
        return {
            "invoice_id": inv_id,
            "subject": existing["subject"],
            "body": existing["body"],
            "tone": existing["tone"],
            "tone_label": get_tone_category(existing["risk"])[1],
            "risk": existing["risk"],
            "reason": existing.get("reason") or reason,
            "is_user_edited": bool(existing.get("is_user_edited", 0)),
            "is_fallback": existing.get("model", "") == "Fallback Template"
            or existing.get("source") == "template",
            "source": existing.get("source")
            or ("template" if existing.get("model") == "Fallback Template" else "gemma"),
            "model": existing.get("model", "Gemma"),
            "created_at": existing.get("created_at"),
            "updated_at": existing.get("updated_at"),
        }

    # If existing was edited by user and caller requested regenerate, check force
    if existing and existing.get("is_user_edited") and not force_regenerate:
        return {
            "invoice_id": inv_id,
            "subject": existing["subject"],
            "body": existing["body"],
            "tone": existing["tone"],
            "tone_label": get_tone_category(existing["risk"])[1],
            "risk": existing["risk"],
            "reason": existing.get("reason") or reason,
            "is_user_edited": True,
            "is_fallback": existing.get("model", "") == "Fallback Template"
            or existing.get("source") == "template",
            "source": existing.get("source")
            or ("template" if existing.get("model") == "Fallback Template" else "gemma"),
            "model": existing.get("model", "Gemma"),
        }

    # Generate draft with Gemma, falling back to plain template
    is_fallback = False
    model_used = "Gemma"
    source = "gemma"
    if use_fallback_only:
        is_fallback = True
        model_used = "Fallback Template"
        source = "template"
        subject, body = get_fallback_template(invoice, tone_key=tone_key)
    else:
        try:
            subject, body = draft_chase_email_with_gemma(
                invoice=invoice,
                risk=risk_val,
                reason=reason,
                tone_key=tone_key,
                tone_instruction=tone_instruction,
                model_name=model_name,
            )
        except Exception:
            # Fall back gracefully to plain template
            is_fallback = True
            model_used = "Fallback Template"
            source = "template"
            subject, body = get_fallback_template(invoice, tone_key=tone_key)

    # Persist draft to SQLite
    save_draft(
        invoice_id=inv_id,
        subject=subject,
        body=body,
        tone=tone_key,
        risk=risk_val,
        reason=reason,
        is_user_edited=False,
        force=force_regenerate,
        model=model_used,
        source=source,
        db_path=db_path,
    )

    return {
        "invoice_id": inv_id,
        "subject": subject,
        "body": body,
        "tone": tone_key,
        "tone_label": tone_label,
        "risk": risk_val,
        "reason": reason,
        "is_user_edited": False,
        "is_fallback": is_fallback,
        "source": source,
        "model": model_used,
    }
