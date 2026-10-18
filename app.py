"""PayRadar: Local-first late-payment radar for freelancers.

Streamlit application featuring:
1. Inbox: Upload invoice (PDF/image), extract fields via Gemma vision with confidence flags, confirm to DB.
2. Radar: Open invoices sorted by risk, cached TabPFN scores, 8-week rolling cash flow chart, gap warnings, monthly cost setting.
3. Chase: Open invoice reminder drafting, tone tiers (<0.3, 0.3-0.6, >0.6), plain-words rationale, copyable draft.

Supports DEMO_MODE=1 for precomputed synthetic demo data.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from payradar.chase import (
    generate_plain_words_reason,
    get_tone_category,
)
from payradar.forecast import build_weekly_forecast, calculate_expected_pay_dates

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip() in ("1", "true", "True")
DEFAULT_REF_DATE = datetime.date(2026, 10, 15)

# Page configuration
st.set_page_config(
    page_title="PayRadar — Late-Payment Radar",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #4361ee;
    }
    .risk-high {
        color: #d90429;
        font-weight: 700;
    }
    .risk-mid {
        color: #e85d04;
        font-weight: 600;
    }
    .risk-low {
        color: #2b9348;
        font-weight: 600;
    }
    .gap-badge {
        background-color: #ffccd5;
        color: #c9184a;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .covered-badge {
        background-color: #d8f3dc;
        color: #2d6a4f;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .synthetic-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 16px;
        border-radius: 6px;
        border-left: 5px solid #ffeeba;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_system_health() -> dict[str, Any]:
    """Check availability of SQLite DB, TabPFN token, and Ollama server."""
    if DEMO_MODE:
        return {
            "db_ok": True,
            "tabpfn_ok": True,
            "ollama_ok": True,
            "ollama_model": "gemma4:e2b (Precomputed)",
        }

    health: dict[str, Any] = {
        "db_ok": True,
        "tabpfn_ok": bool(os.environ.get("TABPFN_TOKEN")),
        "ollama_ok": False,
        "ollama_model": None,
    }
    try:
        from payradar.db import init_db

        init_db()
    except Exception:
        health["db_ok"] = False

    try:
        from payradar.extract import get_available_models

        models = get_available_models()
        if models:
            health["ollama_ok"] = True
            pref = os.environ.get("GEMMA_MODEL", "gemma4:e2b")
            health["ollama_model"] = pref if pref in models else models[0]
    except Exception:
        health["ollama_ok"] = False

    return health


def render_sidebar(health: dict[str, Any]) -> str:
    """Render sidebar navigation, branding, and system status."""
    st.sidebar.title("🎯 PayRadar")
    st.sidebar.caption("Local-first late-payment radar for freelancers")

    if DEMO_MODE:
        st.sidebar.warning(
            "🟡 **DEMO MODE ACTIVE**\nPrecomputed JSON loaded. Live uploads & Gemma disabled."
        )

    page = st.sidebar.radio(
        "Navigation",
        ["📥 Inbox", "🎯 Radar", "✉️ Chase"],
        index=1,  # Default to Radar
    )

    st.sidebar.divider()
    st.sidebar.subheader("System Health")

    # DB Status
    if health["db_ok"]:
        st.sidebar.success("Database: Connected (SQLite)")
    else:
        st.sidebar.error("Database: Error connecting")

    # TabPFN Status
    if DEMO_MODE:
        st.sidebar.success("Scoring Engine: TabPFN-3.5 (precomputed)")
    elif health["tabpfn_ok"]:
        st.sidebar.success("Scoring Engine: TabPFN-3.5 Active")
    else:
        st.sidebar.warning("TabPFN Token: Unset\n(Using HistGradientBoosting baseline)")

    # Ollama Status
    if health["ollama_ok"]:
        st.sidebar.success(f"Ollama: Online ({health['ollama_model']})")
    else:
        st.sidebar.warning("Ollama: Offline\n(Using plain template fallbacks)")

    st.sidebar.divider()
    st.sidebar.caption("Hacktoberfest Open-Source AI Challenge\nWeek 2 (Side Hustle) • Local-first")
    return page


# ==============================================================================
# PAGE 1: INBOX
# ==============================================================================
def render_inbox_page(health: dict[str, Any]) -> None:
    st.markdown('<div class="main-header">📥 Invoice Inbox</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Drop in invoices (PDF or image). Gemma extracts the fields offline. Review, edit, and confirm to track on Radar.</div>',
        unsafe_allow_html=True,
    )

    if DEMO_MODE:
        st.info(
            "🟡 **DEMO MODE**: Document uploads and live Gemma extraction are disabled in hosted demo mode. "
            "You can view sample extracted invoices below to inspect the review and confirmation workflow."
        )

        sample_demo_invoices = [
            {
                "id": "INV-2026-DEMO-01",
                "client": "Apex Innovations",
                "issued_on": "2026-10-01",
                "due_on": "2026-10-31",
                "amount": 6863.46,
                "currency": "USD",
                "terms_days": 30,
                "project_type": "consulting",
                "confidence": 0.94,
            },
            {
                "id": "INV-2026-DEMO-02",
                "client": "Vanguard Media",
                "issued_on": "2026-10-10",
                "due_on": "2026-11-09",
                "amount": 8511.39,
                "currency": "USD",
                "terms_days": 30,
                "project_type": "branding",
                "confidence": 0.91,
            },
        ]
        chosen_sample = st.selectbox(
            "Select a pre-extracted sample invoice to preview:",
            sample_demo_invoices,
            format_func=lambda x: f"{x['id']} — {x['client']} (${x['amount']:,.2f})",
        )
        if chosen_sample:
            st.session_state["extracted_invoice"] = chosen_sample

    else:
        uploaded_file = st.file_uploader(
            "Choose an invoice file (PDF, PNG, JPG, WEBP)",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            help="Files are processed entirely on your machine. Nothing leaves local storage.",
        )

        col_prev, col_act = st.columns([1, 1])

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            with col_prev:
                st.subheader("Document Preview")
                try:
                    from payradar.extract import render_doc_to_image_bytes

                    img_bytes = render_doc_to_image_bytes(
                        file_bytes, filename=uploaded_file.name, max_dimension=800
                    )
                    st.image(
                        img_bytes,
                        caption=f"Uploaded: {uploaded_file.name}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Could not render preview: {e}")

            with col_act:
                st.subheader("Offline Extraction")
                st.write(
                    "Click below to trigger local Gemma Vision extraction. "
                    "This action is run on demand and never blocks normal page renders."
                )

                if st.button("🚀 Extract Invoice Fields with Gemma", type="primary"):
                    if not health["ollama_ok"]:
                        st.error(
                            "Ollama is not running. Please start Ollama (`ollama serve`) and ensure "
                            f"`{os.environ.get('GEMMA_MODEL', 'gemma4:e2b')}` is installed. "
                            "You can also fill the fields manually below."
                        )
                    else:
                        with st.spinner("Extracting invoice fields with local Gemma vision..."):
                            try:
                                from payradar.extract import extract_invoice

                                extraction = extract_invoice(
                                    file_bytes=file_bytes,
                                    filename=uploaded_file.name,
                                    timeout=60.0,
                                )
                                st.session_state["extracted_invoice"] = extraction.model_dump()
                                st.success("Extraction complete! Review the fields below.")
                            except Exception as ex:
                                st.error(
                                    f"Gemma extraction error: {ex}. Please enter fields manually below."
                                )

    # Form to review & confirm extracted invoice
    extracted = st.session_state.get("extracted_invoice")
    if extracted:
        st.divider()
        st.subheader("Review & Confirm Extracted Fields")

        conf = float(extracted.get("confidence", 0.90))
        conf_color = "green" if conf >= 0.85 else ("orange" if conf >= 0.70 else "red")
        st.markdown(
            f"Extraction Confidence: <span style='color:{conf_color}; font-weight:bold;'>{conf * 100:.1f}%</span> "
            f"({'High Confidence' if conf >= 0.85 else 'Review Carefully'})",
            unsafe_allow_html=True,
        )

        with st.form("confirm_invoice_form"):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                inv_id = st.text_input("Invoice ID / Number *", value=str(extracted.get("id", "")))
                client = st.text_input("Client Name *", value=str(extracted.get("client", "")))
                amount = st.number_input(
                    "Total Amount *",
                    min_value=0.01,
                    value=float(extracted.get("amount", 1000.0)),
                    step=50.0,
                    format="%.2f",
                )
                currency = st.selectbox(
                    "Currency",
                    ["USD", "EUR", "GBP", "CAD", "AUD"],
                    index=0 if extracted.get("currency") == "USD" else 1,
                )
            with fcol2:
                issued_raw = extracted.get("issued_on") or datetime.date.today().strftime(
                    "%Y-%m-%d"
                )
                due_raw = extracted.get("due_on") or (
                    datetime.date.today() + datetime.timedelta(days=30)
                ).strftime("%Y-%m-%d")
                try:
                    issued_val = datetime.datetime.strptime(issued_raw, "%Y-%m-%d").date()
                except Exception:
                    issued_val = datetime.date.today()
                try:
                    due_val = datetime.datetime.strptime(due_raw, "%Y-%m-%d").date()
                except Exception:
                    due_val = issued_val + datetime.timedelta(days=30)

                issued_on = st.date_input("Issued Date *", value=issued_val)
                due_on = st.date_input("Due Date *", value=due_val)
                terms_days = st.number_input(
                    "Terms (Days)",
                    min_value=0,
                    max_value=180,
                    value=int(extracted.get("terms_days", 30)),
                )
                project_type = st.selectbox(
                    "Project Type",
                    ["consulting", "web_dev", "design", "branding", "mobile_app", "general"],
                    index=0,
                )

            submitted = st.form_submit_button(
                "✅ Confirm & Save to Radar",
                type="primary",
                disabled=DEMO_MODE,
                help="Disabled in demo mode" if DEMO_MODE else None,
            )
            if DEMO_MODE:
                st.caption(
                    "ℹ️ Confirm & Save is disabled in demo mode (view-only precomputed data)."
                )
            elif submitted:
                if not inv_id or not client or amount <= 0:
                    st.error("Please fill all required fields (ID, Client, Amount).")
                else:
                    invoice_record = {
                        "id": inv_id.strip(),
                        "client": client.strip(),
                        "issued_on": issued_on.strftime("%Y-%m-%d"),
                        "due_on": due_on.strftime("%Y-%m-%d"),
                        "amount": float(amount),
                        "currency": currency,
                        "terms_days": int(terms_days),
                        "project_type": project_type,
                        "paid_on": None,
                        "source_file": extracted.get("source_file", "manual_or_upload"),
                    }
                    from payradar.db import insert_invoice

                    insert_invoice(invoice_record)
                    st.success(
                        f"Invoice {inv_id} successfully saved to SQLite! It is now tracked on your Radar."
                    )


# ==============================================================================
# PAGE 2: RADAR
# ==============================================================================
def render_radar_page(health: dict[str, Any]) -> None:
    st.markdown(
        '<div class="main-header">🎯 Late-Payment & Cash Gap Radar</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Real-time risk scoring of open invoices and 8-week rolling cash flow gap detection.</div>',
        unsafe_allow_html=True,
    )

    # Visible Synthetic Banner
    st.markdown(
        '<div class="synthetic-banner">📊 <strong>SYNTHETIC DEMO DATA</strong>: '
        "All client habits, payment distributions, and forecasts shown are synthetic for hackathon demonstration. "
        "Predictions strictly evaluate data issued prior to each invoice (zero temporal leakage).</div>",
        unsafe_allow_html=True,
    )

    # Load open invoices and cached scores
    if DEMO_MODE:
        demo_dir = PROJECT_ROOT / "demo"
        with open(demo_dir / "invoices_history.json", "r", encoding="utf-8") as f:
            all_invoices = json.load(f)
        with open(demo_dir / "scores.json", "r", encoding="utf-8") as f:
            scores_data = json.load(f)
        scores_by_id = {s["invoice_id"]: s for s in scores_data}

        open_invoices = []
        for inv in all_invoices:
            if inv.get("paid_on") is None:
                rec = dict(inv)
                s = scores_by_id.get(rec["id"], {})
                rec["risk"] = s.get("risk", 0.1)
                rec["predicted_days_late"] = s.get("predicted_days_late", 0.0)
                rec["model"] = s.get("model", "TabPFN-3.5")
                open_invoices.append(rec)
        open_invoices.sort(key=lambda x: x.get("risk", 0.0), reverse=True)

        with open(demo_dir / "settings.json", "r", encoding="utf-8") as f:
            demo_settings = json.load(f)
        monthly_costs = float(demo_settings.get("monthly_costs", 5000.0))
    else:
        from payradar.db import get_scored_open_invoices, get_setting, init_db

        init_db()
        open_invoices = get_scored_open_invoices()
        cost_str = get_setting("monthly_costs", "5000.0")
        monthly_costs = float(cost_str or 5000.0)

    # Handling empty DB
    if not open_invoices:
        st.warning(
            "No open invoices found in the database. "
            "Upload an invoice in the **Inbox** or run `uv run python scripts/make_demo_data.py` to generate synthetic demo data."
        )
        return

    # Calculate expected pay dates and 8-week forecast
    enriched_invoices = calculate_expected_pay_dates(open_invoices, today=DEFAULT_REF_DATE)
    forecast_df = build_weekly_forecast(
        enriched_invoices,
        monthly_costs=monthly_costs,
        today=DEFAULT_REF_DATE,
        weeks_ahead=8,
    )

    # Summary KPIs
    total_open_count = len(enriched_invoices)
    total_open_amount = sum(float(inv.get("amount", 0.0)) for inv in enriched_invoices)
    high_risk_count = sum(
        1 for inv in enriched_invoices if float(inv.get("risk", 0.0) or 0.0) >= 0.60
    )
    gap_weeks_count = int(forecast_df["is_gap_week"].sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Open Invoices", f"{total_open_count} invoices")
    kpi2.metric("Total Outstanding", f"${total_open_amount:,.2f}")
    kpi3.metric(
        "High Late Risk (>60%)",
        f"{high_risk_count} invoices",
        delta=f"{high_risk_count / total_open_count * 100:.0f}% of total"
        if total_open_count
        else None,
        delta_color="inverse",
    )
    kpi4.metric(
        "8-Week Cash Gaps",
        f"{gap_weeks_count} of 8 weeks",
        delta=f"{gap_weeks_count} gap weeks" if gap_weeks_count > 0 else "Fully covered",
        delta_color="inverse" if gap_weeks_count > 0 else "normal",
    )

    st.write("")

    # Re-score action bar
    col_act, col_cost = st.columns([2, 1])
    with col_act:
        st.write("**Scoring Engine:** Cached TabPFN predictions (zero render latency).")
        if DEMO_MODE:
            st.button(
                "🔄 Re-score Open Invoices",
                disabled=True,
                help="Re-scoring is disabled in DEMO_MODE",
            )
            st.caption(
                "ℹ️ Re-scoring is disabled in demo mode (using precomputed TabPFN-3.5 scores)."
            )
        else:
            if st.button("🔄 Re-score Open Invoices (Run TabPFN)", type="primary"):
                with st.spinner("Running TabPFN inference to score all open invoices..."):
                    try:
                        from scripts.predict_open import score_open_invoices

                        score_open_invoices(today=DEFAULT_REF_DATE, fast=True)
                        st.success("Successfully updated scores table!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Scoring error: {ex}")

    with col_cost:
        with st.expander("⚙️ Monthly Operating Cost Setting", expanded=False):
            new_cost = st.number_input(
                "Monthly Fixed Costs ($)",
                min_value=500.0,
                max_value=100000.0,
                value=monthly_costs,
                step=500.0,
            )
            if st.button("Save Cost Setting"):
                if not DEMO_MODE:
                    from payradar.db import set_setting

                    set_setting("monthly_costs", str(new_cost))
                    st.success("Updated monthly cost setting!")
                    st.rerun()
                else:
                    st.info("Cost setting update simulated in DEMO_MODE.")

    st.divider()

    # SECTION 1: 8-Week Rolling Cash Flow & Gap Radar
    st.subheader("📊 8-Week Expected Cash-In vs Operating Costs")
    st.caption(
        "Forecast is strictly based on invoices issued so far. "
        f"Weekly threshold is prorated to ${monthly_costs * 12 / 52:,.2f}/week. "
        "Red bars highlight cash gap weeks where expected cash-in falls below weekly costs."
    )

    # Prepare DataFrame for Altair chart
    chart_data = forecast_df.copy()
    chart_data["Week Label"] = chart_data.apply(
        lambda r: f"Wk {int(r['week_num'])} ({r['start_date'][5:]})", axis=1
    )
    chart_data["Status"] = chart_data["is_gap_week"].apply(lambda g: "Gap Week" if g else "Covered")

    # Altair Bar Chart with Threshold Rule
    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Week Label:N", title="Week (Start Date)", sort=None),
            y=alt.Y("expected_cash_in:Q", title="Expected Cash-In ($)"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Covered", "Gap Week"], range=["#2a9d8f", "#e63946"]),
                legend=alt.Legend(title="Cash Flow Status"),
            ),
            tooltip=[
                alt.Tooltip("Week Label:N", title="Week"),
                alt.Tooltip("start_date:N", title="Start"),
                alt.Tooltip("end_date:N", title="End"),
                alt.Tooltip("expected_cash_in:Q", title="Expected Cash ($)", format="$,.2f"),
                alt.Tooltip("weekly_cost:Q", title="Weekly Cost ($)", format="$,.2f"),
                alt.Tooltip("net_cash_flow:Q", title="Net Flow ($)", format="$,.2f"),
                alt.Tooltip("Status:N", title="Status"),
            ],
        )
    )

    rule = (
        alt.Chart(chart_data)
        .mark_rule(color="#e76f51", strokeDash=[6, 4], size=2.5)
        .encode(
            y="weekly_cost:Q",
            tooltip=[
                alt.Tooltip("weekly_cost:Q", title="Weekly Cost Threshold ($)", format="$,.2f")
            ],
        )
    )

    st.altair_chart(bar_chart + rule, use_container_width=True)

    # Weekly breakdown table
    with st.expander("📋 View 8-Week Cash Flow Breakdown Table"):
        display_forecast = chart_data[
            [
                "week_num",
                "start_date",
                "end_date",
                "expected_cash_in",
                "weekly_cost",
                "net_cash_flow",
                "Status",
            ]
        ].copy()
        display_forecast.columns = [
            "Week #",
            "Period Start",
            "Period End",
            "Cash In ($)",
            "Cost ($)",
            "Net Cash Flow ($)",
            "Status",
        ]
        st.dataframe(
            display_forecast.style.format(
                {
                    "Cash In ($)": "${:,.2f}",
                    "Cost ($)": "${:,.2f}",
                    "Net Cash Flow ($)": "${:+,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # SECTION 2: Open Invoices Sorted by Late-Payment Risk
    st.subheader(f"📋 Open Invoices Ranked by Late-Payment Risk ({len(enriched_invoices)} Total)")

    table_rows = []
    for inv in enriched_invoices:
        risk_val = float(inv.get("risk", 0.0) or 0.0)
        delay_val = float(inv.get("predicted_days_late", 0.0) or 0.0)
        risk_tier = (
            "🔴 HIGH" if risk_val >= 0.60 else ("🟠 MEDIUM" if risk_val >= 0.30 else "🟢 LOW")
        )

        table_rows.append(
            {
                "Invoice ID": inv.get("id"),
                "Client": inv.get("client"),
                "Amount": float(inv.get("amount", 0.0)),
                "Due Date": inv.get("due_on"),
                "Predicted Delay": f"{delay_val:+.1f} days",
                "Expected Pay Date": inv.get("expected_pay_date"),
                "Late Risk": risk_val,
                "Risk Tier": risk_tier,
                "Payment Status": inv.get("pay_status", "on schedule"),
            }
        )

    df_table = pd.DataFrame(table_rows)

    st.dataframe(
        df_table.style.format(
            {
                "Amount": "${:,.2f}",
                "Late Risk": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# PAGE 3: CHASE
# ==============================================================================
def render_chase_page(health: dict[str, Any]) -> None:
    st.markdown(
        '<div class="main-header">✉️ Smart Chaser Email Drafts</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Select an open invoice to inspect late-payment rationale and generate tailored reminder emails. Draft only — you review and copy.</div>',
        unsafe_allow_html=True,
    )

    # Load open invoices and history
    if DEMO_MODE:
        demo_dir = PROJECT_ROOT / "demo"
        with open(demo_dir / "invoices_history.json", "r", encoding="utf-8") as f:
            all_invoices = json.load(f)
        with open(demo_dir / "scores.json", "r", encoding="utf-8") as f:
            scores_data = json.load(f)
        scores_by_id = {s["invoice_id"]: s for s in scores_data}

        with open(demo_dir / "drafts.json", "r", encoding="utf-8") as f:
            demo_drafts = json.load(f)
        drafts_by_id = {d["invoice_id"]: d for d in demo_drafts}

        open_invoices = []
        for inv in all_invoices:
            if inv.get("paid_on") is None:
                rec = dict(inv)
                s = scores_by_id.get(rec["id"], {})
                rec["risk"] = s.get("risk", 0.1)
                rec["predicted_days_late"] = s.get("predicted_days_late", 0.0)
                open_invoices.append(rec)
        open_invoices.sort(key=lambda x: x.get("risk", 0.0), reverse=True)
    else:
        from payradar.db import get_all_invoices, get_scored_open_invoices, init_db

        init_db()
        all_invoices = get_all_invoices()
        open_invoices = get_scored_open_invoices()
        drafts_by_id = {}

    if not open_invoices:
        st.warning(
            "No open invoices found. Add invoices in the Inbox to start tracking payment risk."
        )
        return

    # Invoice Selector
    inv_options = {
        f"{inv['id']} | {inv['client']} — ${float(inv['amount']):,.2f} ({float(inv.get('risk', 0.0) or 0.0) * 100:.1f}% risk)": inv
        for inv in open_invoices
    }
    selected_label = st.selectbox("Select Open Invoice to Chase:", list(inv_options.keys()))
    selected_invoice = inv_options[selected_label]
    inv_id = selected_invoice["id"]

    risk_val = float(selected_invoice.get("risk", 0.0) or 0.0)
    delay_val = float(selected_invoice.get("predicted_days_late", 0.0) or 0.0)
    tone_key, tone_label, tone_instruction = get_tone_category(risk_val)

    # Compute plain words reason
    reason_text = generate_plain_words_reason(
        selected_invoice,
        history_invoices=all_invoices,
        risk=risk_val,
        predicted_days_late=delay_val,
    )

    # Display Risk & Plain-Words Rationale Card
    st.write("")
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        with st.container(border=True):
            st.markdown("#### 📄 Invoice Summary")
            st.markdown(f"**ID:** `{selected_invoice['id']}`")
            st.markdown(f"**Client:** {selected_invoice['client']}")
            st.markdown(f"**Amount:** ${float(selected_invoice['amount']):,.2f}")
            st.markdown(f"**Due Date:** {selected_invoice['due_on']}")
            st.markdown(f"**Assigned Tone:** {tone_label}")

    with col_r2:
        with st.container(border=True):
            st.markdown("#### 🔍 Risk Rationale (Plain Words)")
            st.markdown(f"{reason_text}")
            st.caption(f"**Tone Strategy:** {tone_instruction}")

    st.divider()

    # Draft Management: Check existing draft or generate
    if DEMO_MODE:
        draft = drafts_by_id.get(inv_id)
        if not draft:
            from payradar.chase import get_or_create_chase_draft

            draft = get_or_create_chase_draft(
                selected_invoice,
                history_invoices=all_invoices,
                risk=risk_val,
                predicted_days_late=delay_val,
                use_fallback_only=True,
            )
    else:
        from payradar.chase import get_or_create_chase_draft
        from payradar.db import get_draft

        existing_row = get_draft(inv_id)
        if existing_row:
            draft = {
                "invoice_id": existing_row["invoice_id"],
                "subject": existing_row["subject"],
                "body": existing_row["body"],
                "tone": existing_row["tone"],
                "tone_label": get_tone_category(existing_row["risk"])[1],
                "risk": existing_row["risk"],
                "reason": existing_row.get("reason") or reason_text,
                "is_user_edited": bool(existing_row["is_user_edited"]),
                "model": existing_row.get("model", "Gemma"),
                "source": dict(existing_row).get("source")
                or ("template" if existing_row.get("model") == "Fallback Template" else "gemma"),
            }
        else:
            draft = get_or_create_chase_draft(
                selected_invoice,
                history_invoices=all_invoices,
                risk=risk_val,
                predicted_days_late=delay_val,
                force_regenerate=False,
            )

    # Action buttons for generation / regeneration
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("🤖 Generate / Re-draft with Gemma", type="secondary"):
            if DEMO_MODE:
                st.info("Live Gemma generation is simulated in DEMO_MODE.")
            else:
                if draft and draft.get("is_user_edited"):
                    st.warning(
                        "⚠️ This draft has been edited by you. Re-drafting will preserve your custom text unless explicitly confirmed."
                    )
                with st.spinner("Drafting reminder email with Gemma..."):
                    from payradar.chase import get_or_create_chase_draft

                    draft = get_or_create_chase_draft(
                        selected_invoice,
                        history_invoices=all_invoices,
                        risk=risk_val,
                        predicted_days_late=delay_val,
                        force_regenerate=True,
                    )
                    st.success("Fresh draft generated!")
                    st.rerun()

    # Draft Editor
    st.subheader("Edit & Copy Reminder Draft")

    # Source Indicator Badge
    source_type = draft.get(
        "source", "template" if draft.get("model") == "Fallback Template" else "gemma"
    )
    if DEMO_MODE:
        source_label = (
            "Source: Gemma 4 (precomputed)"
            if source_type == "gemma"
            else "Source: Plain Template (precomputed)"
        )
        icon = "🤖" if source_type == "gemma" else "📋"
        bg_col = "#d8f3dc" if source_type == "gemma" else "#f1f3f5"
        txt_col = "#2d6a4f" if source_type == "gemma" else "#495057"
        st.markdown(
            f'<div style="margin-bottom: 8px;"><span style="background-color: {bg_col}; color: {txt_col}; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">{icon} {source_label}</span></div>',
            unsafe_allow_html=True,
        )
    elif source_type == "gemma":
        st.markdown(
            '<div style="margin-bottom: 8px;"><span style="background-color: #d8f3dc; color: #2d6a4f; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">🤖 Source: Gemma 4 (Live LLM)</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin-bottom: 8px;"><span style="background-color: #f1f3f5; color: #495057; border: 1px solid #ced4da; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">📋 Source: Plain Template (Fallback)</span></div>',
            unsafe_allow_html=True,
        )

    if draft.get("is_user_edited"):
        st.info(
            "✏️ **User-Edited Draft**: Your custom edits are protected and will never be overwritten automatically."
        )

    edit_subject = st.text_input("Email Subject Line", value=draft.get("subject", ""))
    edit_body = st.text_area("Email Body", value=draft.get("body", ""), height=220)

    save_col, copy_col = st.columns([1, 1])
    with save_col:
        if st.button("💾 Save Custom Draft"):
            if not DEMO_MODE:
                from payradar.db import save_draft

                save_draft(
                    invoice_id=inv_id,
                    subject=edit_subject,
                    body=edit_body,
                    tone=tone_key,
                    risk=risk_val,
                    reason=reason_text,
                    is_user_edited=True,
                    force=True,
                    source=source_type,
                )
                st.success("Draft saved! Your edits are locked and protected from overwrite.")
                st.rerun()
            else:
                st.success("Draft edits saved (simulated in DEMO_MODE).")

    with copy_col:
        st.caption("Click into the block below to copy with standard Ctrl+C / Cmd+C:")
        full_email_text = f"Subject: {edit_subject}\n\n{edit_body}"
        st.code(full_email_text, language="text")

    st.markdown(
        """
        <div style="margin-top: 2rem; padding: 12px; background-color: #f1f3f5; border-radius: 6px; font-size: 0.85rem; color: #495057;">
            🔒 <strong>Local-First Guarantee</strong>: PayRadar never sends emails automatically. No credentials or communications leave your machine.
            Copy the finalized text directly into your preferred email client (Gmail, Outlook, Apple Mail, etc.).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    health = check_system_health()
    selected_page = render_sidebar(health)

    if selected_page == "📥 Inbox":
        render_inbox_page(health)
    elif selected_page == "🎯 Radar":
        render_radar_page(health)
    elif selected_page == "✉️ Chase":
        render_chase_page(health)


if __name__ == "__main__":
    main()
