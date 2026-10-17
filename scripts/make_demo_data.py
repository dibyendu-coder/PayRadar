"""Synthetic demo data generator for PayRadar.

Generates:
1. 25 synthetic clients with 5 distinct payment habits (always late, pays on 30th, big invoices slower, reliable, erratic).
2. ~400 realistic invoices spanning 3 years saved to SQLite and demo/invoices_history.json.
3. 20 rendered ground-truth invoices (PDF & PNG) across 5 distinct visual layouts with metadata JSON.

Notice: ALL DATA IS 100% SYNTHETIC and labelled synthetic everywhere.
"""

from __future__ import annotations

import json
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from payradar.db import init_db, insert_invoice

# Fixed seed for perfect reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
GT_DIR = DEMO_DIR / "ground_truth"
GT_INVOICES_DIR = GT_DIR / "invoices"
GT_META_DIR = GT_DIR / "metadata"

CLIENT_PROFILES = [
    # 1. Always late (delays 15 - 45 days past due)
    {"name": "Apex Innovations", "archetype": "always_late", "industry": "tech"},
    {"name": "Vanguard Media", "archetype": "always_late", "industry": "marketing"},
    {"name": "Echo Dynamics", "archetype": "always_late", "industry": "hardware"},
    {"name": "Summit Global Logistics", "archetype": "always_late", "industry": "supply_chain"},
    {"name": "Cascade Retail Brands", "archetype": "always_late", "industry": "ecommerce"},
    # 2. Pays on the 30th (batches payments to end of month)
    {"name": "Orion Enterprise", "archetype": "pays_on_30th", "industry": "enterprise"},
    {"name": "Beacon Health Group", "archetype": "pays_on_30th", "industry": "healthcare"},
    {"name": "Atlas Financial Tech", "archetype": "pays_on_30th", "industry": "fintech"},
    {"name": "Zenith Corp", "archetype": "pays_on_30th", "industry": "manufacturing"},
    {"name": "Prime Legal Solutions", "archetype": "pays_on_30th", "industry": "legal"},
    # 3. Big invoices slower (small: 0-3d late, large >= $3500: 20-40d late)
    {"name": "Quantum Ventures", "archetype": "big_invoices_slower", "industry": "vc"},
    {"name": "Starlight Systems", "archetype": "big_invoices_slower", "industry": "software"},
    {"name": "Pinnacle Advisory", "archetype": "big_invoices_slower", "industry": "consulting"},
    {"name": "Cobalt BioLabs", "archetype": "big_invoices_slower", "industry": "biotech"},
    {"name": "Horizon CleanTech", "archetype": "big_invoices_slower", "industry": "energy"},
    # 4. Reliable (pays on time, -3 to +2 days from due date)
    {"name": "Nexus Software Labs", "archetype": "reliable", "industry": "saas"},
    {"name": "Pulse Digital Studio", "archetype": "reliable", "industry": "design"},
    {"name": "Crestline Partners", "archetype": "reliable", "industry": "private_equity"},
    {"name": "Solaria Renewable Energy", "archetype": "reliable", "industry": "cleantech"},
    {"name": "Acuity AI Analytics", "archetype": "reliable", "industry": "ai"},
    # 5. Erratic startups (variable, -4 to +28 days)
    {"name": "RocketByte Labs", "archetype": "erratic", "industry": "crypto"},
    {"name": "Nova Intelligence", "archetype": "erratic", "industry": "ai"},
    {"name": "SparkStudio Interactive", "archetype": "erratic", "industry": "gaming"},
    {"name": "Kite Urban Mobility", "archetype": "erratic", "industry": "transport"},
    {"name": "Drift Protocol", "archetype": "erratic", "industry": "web3"},
]

PROJECT_TYPES = [
    "web_dev",
    "mobile_app",
    "ui_ux_design",
    "cloud_infrastructure",
    "data_engineering",
    "ai_consulting",
]


def calculate_paid_on(
    issued_on: date,
    due_on: date,
    amount: float,
    archetype: str,
    cutoff_date: date,
) -> str | None:
    """Calculate synthetic paid_on date based on client habit archetype."""
    # If due_on is within the last 15 days or in future relative to cutoff, some open invoices remain
    days_past_due = (cutoff_date - due_on).days
    if days_past_due < 0 and random.random() < 0.85:
        return None  # Future invoice, unpaid
    if 0 <= days_past_due <= 25 and random.random() < 0.50:
        return None  # Recently due invoice, still unpaid

    delay = 0
    if archetype == "always_late":
        delay = random.randint(14, 45)
        pay_date = due_on + timedelta(days=delay)
    elif archetype == "pays_on_30th":
        # Pays on the 28th-30th of the due month or next month
        target_year = due_on.year
        target_month = due_on.month
        if due_on.day > 25:
            if target_month == 12:
                target_month = 1
                target_year += 1
            else:
                target_month += 1
        day = random.choice([28, 29, 30])
        try:
            pay_date = date(target_year, target_month, day)
        except ValueError:
            pay_date = date(target_year, target_month, 28)
        if pay_date < due_on:
            pay_date = due_on + timedelta(days=random.randint(1, 5))
    elif archetype == "big_invoices_slower":
        if amount >= 3500.0:
            delay = random.randint(18, 42)
        else:
            delay = random.randint(-2, 4)
        pay_date = due_on + timedelta(days=delay)
    elif archetype == "reliable":
        delay = random.randint(-4, 3)
        pay_date = due_on + timedelta(days=delay)
    else:  # erratic
        delay = random.randint(-5, 32)
        pay_date = due_on + timedelta(days=delay)

    # Don't mark paid in the future beyond cutoff_date
    if pay_date > cutoff_date:
        return None

    return pay_date.strftime("%Y-%m-%d")


def generate_synthetic_invoices(
    total_target: int = 400,
    cutoff_date: date = date(2026, 10, 15),
) -> list[dict]:
    """Generate ~400 synthetic invoices spanning Nov 2023 to Oct 2026."""
    start_date = date(2023, 11, 1)
    total_days = (cutoff_date - start_date).days
    invoices = []
    inv_counter = 1

    # Distribute ~16 invoices per client over 36 months
    invoices_per_client = total_target // len(CLIENT_PROFILES)

    for client in CLIENT_PROFILES:
        # Generate timestamps spaced over 3 years
        offsets = sorted(random.sample(range(1, total_days), invoices_per_client))
        for offset in offsets:
            issued_on = start_date + timedelta(days=offset)
            terms = random.choice([14, 30, 30, 30, 45, 60])
            due_on = issued_on + timedelta(days=terms)

            # Amount distribution: common range $800 to $12,500
            base_amt = random.choice([1200, 2400, 3500, 4800, 6500, 8500, 11000])
            amount = round(base_amt + random.uniform(-250, 450), 2)

            proj_type = random.choice(PROJECT_TYPES)
            paid_on = calculate_paid_on(
                issued_on, due_on, amount, client["archetype"], cutoff_date
            )

            inv_id = f"INV-{issued_on.year}-{inv_counter:04d}"
            inv_counter += 1

            invoices.append(
                {
                    "id": inv_id,
                    "client": client["name"],
                    "issued_on": issued_on.strftime("%Y-%m-%d"),
                    "due_on": due_on.strftime("%Y-%m-%d"),
                    "amount": amount,
                    "currency": "USD",
                    "terms_days": terms,
                    "project_type": proj_type,
                    "paid_on": paid_on,
                    "source_file": f"{inv_id}.pdf",
                    "synthetic": True,
                }
            )

    # Sort chronologically by issued_on
    invoices.sort(key=lambda x: x["issued_on"])
    return invoices


# --- ReportLab Layout Renderers for 20 Ground Truth Documents ---


def render_layout_classic(doc_path: Path, data: dict):
    """Layout 1: Classic Minimalist Clean."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1E293B'))
    label_style = ParagraphStyle('Label', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#64748B'))
    val_style = ParagraphStyle('Val', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', textColor=colors.HexColor('#0F172A'))

    elements.append(Paragraph("<b>INVOICE</b>", title_style))
    elements.append(Paragraph("<font size=8 color='#94A3B8'>[SYNTHETIC TEST DOCUMENT]</font>", styles['Normal']))
    elements.append(Spacer(1, 15))

    header_table = Table([
        [Paragraph("<b>Billed To:</b>", label_style), Paragraph("<b>Invoice Details:</b>", label_style)],
        [Paragraph(f"<b>{data['client']}</b><br/>Accounts Payable<br/>accounting@{data['client'].lower().replace(' ', '')}.com", val_style),
         Paragraph(f"<b>Invoice ID:</b> {data['id']}<br/><b>Issue Date:</b> {data['issued_on']}<br/><b>Due Date:</b> {data['due_on']}<br/><b>Terms:</b> Net {data['terms_days']} days", val_style)]
    ], colWidths=[280, 250])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    elements.append(header_table)
    elements.append(Spacer(1, 25))

    # Items Table
    items = [
        ["Description", "Category", "Amount"],
        [f"Professional Services ({data['project_type'].replace('_', ' ').title()})", data['project_type'], f"${data['amount']:,.2f}"],
        ["", "Subtotal:", f"${data['amount']:,.2f}"],
        ["", "Total Due:", f"${data['amount']:,.2f}"]
    ]
    t = Table(items, colWidths=[280, 120, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (1,-1), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (1,-1), (-1,-1), colors.HexColor('#F8FAFC')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(f"Payment due within {data['terms_days']} days. Thank you for your partnership.", label_style))

    doc.build(elements)


def render_layout_corporate_blue(doc_path: Path, data: dict):
    """Layout 2: Corporate Blue Grid."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    banner_data = [[
        Paragraph("<font color='white' size=18><b>CONSULTING INVOICE</b></font><br/><font color='#BFDBFE' size=8>SYNTHETIC DEMO RECORD</font>", styles['Normal']),
        Paragraph(f"<font color='white' size=14><b>{data['id']}</b></font><br/><font color='#DBEAFE'>Due: {data['due_on']}</font>", styles['Normal'])
    ]]
    banner = Table(banner_data, colWidths=[360, 180])
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E40AF')),
        ('TOPPADDING', (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 16),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 20))

    meta = [
        ["Client (Bill To)", "Issue Date", "Due Date", "Payment Terms"],
        [data['client'], data['issued_on'], data['due_on'], f"{data['terms_days']} Days"]
    ]
    meta_t = Table(meta, colWidths=[200, 110, 110, 120])
    meta_t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E40AF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EFF6FF')),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    elements.append(meta_t)
    elements.append(Spacer(1, 25))

    lines = [
        ["Service Milestone", "Units", "Rate", "Total"],
        [f"Milestone Deliverables - {data['project_type'].upper()}", "1.0", f"${data['amount']:,.2f}", f"${data['amount']:,.2f}"],
        ["", "", "Total Balance Due:", f"${data['amount']:,.2f}"]
    ]
    lines_t = Table(lines, colWidths=[260, 70, 100, 110])
    lines_t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (2,-1), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (2,-1), (-1,-1), colors.HexColor('#DBEAFE')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(lines_t)
    doc.build(elements)


def render_layout_modern_dark(doc_path: Path, data: dict):
    """Layout 3: Modern Charcoal Agency Layout."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter, rightMargin=45, leftMargin=45, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    header = [
        [Paragraph(f"<font size=20 color='#09090B'><b>STUDIO INVOICE</b></font><br/><font size=9 color='#71717A'>REF: {data['id']} (SYNTHETIC)</font>", styles['Normal']),
         Paragraph(f"<font size=10 color='#71717A'>TOTAL DUE</font><br/><font size=18 color='#09090B'><b>${data['amount']:,.2f}</b></font>", styles['Normal'])]
    ]
    t_head = Table(header, colWidths=[330, 190])
    t_head.setStyle(TableStyle([('ALIGN', (1,0), (1,0), 'RIGHT')]))
    elements.append(t_head)
    elements.append(Spacer(1, 15))

    card_data = [
        ["CUSTOMER", "DATES & TERMS"],
        [f"{data['client']}\nProject: {data['project_type']}",
         f"Issued: {data['issued_on']}\nDue: {data['due_on']}\nTerms: {data['terms_days']} Days Net"]
    ]
    t_card = Table(card_data, colWidths=[260, 260])
    t_card.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#71717A')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F4F5')),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t_card)
    elements.append(Spacer(1, 20))

    t_items = Table([
        ["Item Description", "Qty", "Price", "Amount"],
        [f"Scope Execution: {data['project_type'].replace('_', ' ').title()}", "1", f"${data['amount']:,.2f}", f"${data['amount']:,.2f}"],
        ["", "", "Net Total:", f"${data['amount']:,.2f}"]
    ], colWidths=[270, 50, 100, 100])
    t_items.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor('#18181B')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (2,-1), (-1,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_items)
    doc.build(elements)


def render_layout_compact_freelance(doc_path: Path, data: dict):
    """Layout 4: Compact Freelance Statement."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter, rightMargin=50, leftMargin=50, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>FREELANCE BILLING STATEMENT</b>", styles['Heading2']))
    elements.append(Paragraph("<font size=8 color='#6B7280'>[SYNTHETIC TESTING DATASET]</font>", styles['Normal']))
    elements.append(Spacer(1, 10))

    info = [
        [f"Invoice Number: {data['id']}", f"Issue Date: {data['issued_on']}"],
        [f"Client Name: {data['client']}", f"Payment Due Date: {data['due_on']}"],
        [f"Engagement: {data['project_type']}", f"Standard Terms: Net {data['terms_days']}"]
    ]
    t_info = Table(info, colWidths=[250, 250])
    t_info.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_info)
    elements.append(Spacer(1, 20))

    body = [
        ["Description of Delivered Work", "Total Amount"],
        [f"Contracted Sprint: {data['project_type'].replace('_', ' ').title()}", f"${data['amount']:,.2f}"],
        ["Amount Due (USD):", f"${data['amount']:,.2f}"]
    ]
    t_body = Table(body, colWidths=[360, 140])
    t_body.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E7EB')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#9CA3AF')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_body)
    doc.build(elements)


def render_layout_enterprise_boxed(doc_path: Path, data: dict):
    """Layout 5: Formal Enterprise Boxed Layout."""
    doc = SimpleDocTemplate(str(doc_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    header = Table([
        [Paragraph("<font size=16 color='#047857'><b>ENTERPRISE VENDOR INVOICE</b></font>", styles['Normal']),
         Paragraph(f"<b>INVOICE NO:</b> {data['id']}<br/><font size=8 color='#065F46'>SYNTHETIC RECORD</font>", styles['Normal'])]
    ], colWidths=[320, 210])
    header.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.HexColor('#047857')), ('PADDING', (0,0), (-1,-1), 8)]))
    elements.append(header)
    elements.append(Spacer(1, 15))

    details = [
        ["BILL TO (CLIENT):", "INVOICE METRICS:"],
        [f"Organization: {data['client']}\nStatus: Active Client\nTerms: {data['terms_days']} Days",
         f"Invoice Date: {data['issued_on']}\nPayment Deadline: {data['due_on']}\nCurrency: {data['currency']}"]
    ]
    t_det = Table(details, colWidths=[265, 265])
    t_det.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D1FAE5')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#A7F3D0')),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    elements.append(t_det)
    elements.append(Spacer(1, 20))

    charges = [
        ["Service Item", "Category", "Rate", "Total"],
        [f"Core Engineering Deliverables", data['project_type'], f"${data['amount']:,.2f}", f"${data['amount']:,.2f}"],
        ["TOTAL PAYABLE DUE", "", "", f"${data['amount']:,.2f}"]
    ]
    t_charges = Table(charges, colWidths=[230, 110, 90, 100])
    t_charges.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#047857')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#6EE7B7')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#ECFDF5')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_charges)
    doc.build(elements)


LAYOUT_RENDERERS = [
    render_layout_classic,
    render_layout_corporate_blue,
    render_layout_modern_dark,
    render_layout_compact_freelance,
    render_layout_enterprise_boxed,
]


def render_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 150):
    """Convert generated PDF page 1 to PNG image using PyMuPDF."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[0]
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(png_path))
    doc.close()


def generate_ground_truth_suite(count: int = 20) -> list[dict]:
    """Render 20 ground-truth documents (PDF + PNG) across 5 layout templates with JSON metadata."""
    GT_INVOICES_DIR.mkdir(parents=True, exist_ok=True)
    GT_META_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth_records = []
    # Select diverse clients across archetypes
    selected_clients = CLIENT_PROFILES[:count]

    base_dates = [
        ("2026-08-10", 30), ("2026-08-15", 14), ("2026-08-20", 30), ("2026-08-25", 45),
        ("2026-09-01", 30), ("2026-09-05", 30), ("2026-09-10", 14), ("2026-09-15", 30),
        ("2026-09-20", 60), ("2026-09-25", 30), ("2026-09-28", 30), ("2026-10-01", 14),
        ("2026-10-03", 30), ("2026-10-05", 30), ("2026-10-07", 30), ("2026-10-08", 45),
        ("2026-10-10", 14), ("2026-10-11", 30), ("2026-10-12", 30), ("2026-10-14", 30),
    ]

    for idx in range(count):
        client = selected_clients[idx]
        issue_str, terms = base_dates[idx]
        issue_dt = datetime.strptime(issue_str, "%Y-%m-%d").date()
        due_dt = issue_dt + timedelta(days=terms)
        due_str = due_dt.strftime("%Y-%m-%d")

        amount = round(random.choice([1500, 2750, 3800, 4950, 6200, 7500, 9800]) + random.uniform(50, 200), 2)
        proj = PROJECT_TYPES[idx % len(PROJECT_TYPES)]
        inv_id = f"INV-GT-{idx+1:03d}"

        record = {
            "id": inv_id,
            "client": client["name"],
            "issued_on": issue_str,
            "due_on": due_str,
            "amount": amount,
            "currency": "USD",
            "terms_days": terms,
            "project_type": proj,
            "synthetic": True,
            "source_file": f"inv_{idx+1:02d}.pdf",
        }

        pdf_path = GT_INVOICES_DIR / f"inv_{idx+1:02d}.pdf"
        png_path = GT_INVOICES_DIR / f"inv_{idx+1:02d}.png"
        meta_path = GT_META_DIR / f"inv_{idx+1:02d}.json"

        # Rotate among the 5 distinct layout templates
        renderer = LAYOUT_RENDERERS[idx % len(LAYOUT_RENDERERS)]
        renderer(pdf_path, record)
        render_pdf_to_png(pdf_path, png_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        ground_truth_records.append(record)

    # Save manifest
    manifest_path = GT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_records, f, indent=2)

    return ground_truth_records


def main():
    print("=" * 60)
    print("Generating Synthetic Demo Dataset for PayRadar")
    print("=" * 60)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Clients Profile
    clients_path = DEMO_DIR / "clients.json"
    with open(clients_path, "w", encoding="utf-8") as f:
        json.dump(CLIENT_PROFILES, f, indent=2)
    print(f"[OK] Saved {len(CLIENT_PROFILES)} client profiles to {clients_path.relative_to(PROJECT_ROOT)}")

    # 2. Generate ~400 Synthetic Invoices
    invoices = generate_synthetic_invoices(total_target=400)
    history_path = DEMO_DIR / "invoices_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(invoices, f, indent=2)
    print(f"[OK] Generated {len(invoices)} historical invoices saved to {history_path.relative_to(PROJECT_ROOT)}")

    # 3. Seed SQLite Database
    db_path = PROJECT_ROOT / "payradar.db"
    init_db(db_path)
    for inv in invoices:
        insert_invoice(inv, db_path=db_path)
    print(f"[OK] Seeded {len(invoices)} invoices into SQLite database at {db_path.name}")

    # 4. Generate 20 Ground Truth Documents in 5+ Layouts
    gt_docs = generate_ground_truth_suite(count=20)
    print(f"[OK] Rendered {len(gt_docs)} ground-truth documents (PDF + PNG) across 5 layouts in {GT_DIR.relative_to(PROJECT_ROOT)}")
    print("Synthetic dataset generation complete!")


if __name__ == "__main__":
    main()
