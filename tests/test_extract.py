"""Unit tests for invoice extraction, normalization and document rendering."""

from pathlib import Path

from payradar.extract import (
    InvoiceExtraction,
    clean_currency_amount,
    normalize_date_string,
    render_doc_to_image_bytes,
)


def test_normalize_date_string():
    assert normalize_date_string("2024-05-12") == "2024-05-12"
    assert normalize_date_string("12/05/2024") == "2024-05-12"
    assert normalize_date_string("May 12, 2024") == "2024-05-12"
    assert normalize_date_string("12 May 2024") == "2024-05-12"
    assert normalize_date_string("2024/05/12") == "2024-05-12"


def test_clean_currency_amount():
    assert clean_currency_amount(1250.50) == 1250.50
    assert clean_currency_amount("$1,250.50") == 1250.50
    assert clean_currency_amount("€ 3,400.00") == 3400.00
    assert clean_currency_amount("£500") == 500.00
    assert clean_currency_amount("Amount: $4,567.89 total") == 4567.89


def test_invoice_extraction_model():
    data = {
        "id": "INV-001",
        "client": "Apex Innovations",
        "issued_on": "2024-01-01",
        "due_on": "2024-01-31",
        "amount": 2500.0,
        "currency": "USD",
        "terms_days": 30,
        "project_type": "web_dev",
        "confidence": 0.95,
        "source_file": "inv_001.pdf",
    }
    obj = InvoiceExtraction(**data)
    assert obj.id == "INV-001"
    assert obj.client == "Apex Innovations"
    assert obj.amount == 2500.0
    assert obj.terms_days == 30


def test_render_doc_to_image_bytes():
    gt_pdf = Path("demo/ground_truth/invoices/inv_01.pdf")
    if gt_pdf.exists():
        img_bytes = render_doc_to_image_bytes(gt_pdf)
        assert len(img_bytes) > 1000
        # PNG signature check: \x89PNG
        assert img_bytes[:4] == b"\x89PNG"
