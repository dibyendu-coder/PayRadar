"""Extract structured invoice data using Gemma Vision via Ollama.

Non-negotiable requirements:
- Raw invoices never leave the machine (local Ollama inference).
- Offline batch extraction, never inside a UI request.
- Default to gemma4:e2b via GEMMA_MODEL, fallback to gemma4:e4b or gemma3:4b.
- Uses pymupdf for PDF to image rendering.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from PIL import Image
from pydantic import BaseModel, Field

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:e2b")
FALLBACK_MODELS = ["gemma4:e2b", "gemma4:e4b", "gemma3:4b"]


class InvoiceExtraction(BaseModel):
    id: str = Field(description="Invoice number or identifier")
    client: str = Field(description="Client or customer name receiving the invoice")
    issued_on: str = Field(description="Invoice date in YYYY-MM-DD format")
    due_on: str = Field(description="Due date in YYYY-MM-DD format")
    amount: float = Field(description="Total monetary amount due")
    currency: str = Field(default="USD", description="Currency code (USD, EUR, GBP, etc.)")
    terms_days: int = Field(default=30, description="Payment terms in days")
    project_type: str = Field(default="general", description="Category of services or goods")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")
    source_file: str | None = None


INVOICE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "client": {"type": "string"},
        "issued_on": {"type": "string"},
        "due_on": {"type": "string"},
        "amount": {"type": "number"},
        "currency": {"type": "string"},
        "terms_days": {"type": "integer"},
        "project_type": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["id", "client", "issued_on", "due_on", "amount"],
}


def get_available_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """Retrieve list of locally installed models from Ollama."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def resolve_model(preferred: str | None = None, base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Resolve an available Ollama model following the fallback chain."""
    available = get_available_models(base_url)
    if not available:
        return preferred or DEFAULT_MODEL

    candidates = [preferred] if preferred else []
    candidates.extend([DEFAULT_MODEL] + FALLBACK_MODELS)

    # Match exact or tag prefix (e.g. gemma4:e2b matches gemma4:e2b)
    for cand in candidates:
        if not cand:
            continue
        for avail in available:
            if avail == cand or avail.startswith(cand.split(":")[0]):
                return avail
    return available[0]


def resize_image_bytes(img_bytes: bytes, max_dimension: int = 1024) -> bytes:
    """Downsample image bytes so that max(width, height) <= max_dimension, preserving aspect ratio."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            w, h = img.size
            if max(w, h) <= max_dimension:
                return img_bytes

            scale = max_dimension / max(w, h)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            if resized.mode in ("RGBA", "P"):
                resized = resized.convert("RGB")
            out_buf = io.BytesIO()
            resized.save(out_buf, format="PNG", optimize=True)
            return out_buf.getvalue()
    except Exception:
        return img_bytes


def render_doc_to_image_bytes(
    doc: str | Path | bytes,
    filename: str | None = None,
    dpi: int = 150,
    max_dimension: int = 1024,
) -> bytes:
    """Render a PDF page or image file/bytes to PNG image bytes resized for 4GB VRAM target."""
    if isinstance(doc, bytes):
        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix == ".pdf" or doc.startswith(b"%PDF"):
            pdf_doc = pymupdf.open(stream=doc, filetype="pdf")
            if len(pdf_doc) == 0:
                raise ValueError("PDF has no pages")
            page = pdf_doc[0]
            mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            raw_bytes = pix.tobytes("png")
            pdf_doc.close()
        else:
            raw_bytes = doc
        return resize_image_bytes(raw_bytes, max_dimension=max_dimension)

    path = Path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        raw_bytes = path.read_bytes()
        return resize_image_bytes(raw_bytes, max_dimension=max_dimension)

    if suffix == ".pdf":
        pdf_doc = pymupdf.open(str(path))
        if len(pdf_doc) == 0:
            raise ValueError(f"PDF has no pages: {path}")
        page = pdf_doc[0]
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        raw_bytes = pix.tobytes("png")
        pdf_doc.close()
        return resize_image_bytes(raw_bytes, max_dimension=max_dimension)

    raise ValueError(f"Unsupported document format: {suffix}")


def normalize_date_string(date_str: str | None, default_date: str | None = None) -> str:
    """Normalize various date formats into standard ISO YYYY-MM-DD."""
    if not date_str:
        return default_date or datetime.now().strftime("%Y-%m-%d")

    clean = date_str.strip()
    # Try common formats directly
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Regex fallback for embedded YYYY-MM-DD
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", clean)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return default_date or datetime.now().strftime("%Y-%m-%d")


def clean_currency_amount(val: Any) -> float:
    """Extract numeric amount from string or number."""
    if isinstance(val, (int, float)):
        return float(val)
    if not val:
        return 0.0
    s = str(val).replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0
    return 0.0


def extract_invoice(
    file_path: str | Path | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    model_name: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 120.0,
) -> InvoiceExtraction:
    """Extract structured invoice fields from a PDF or image file using Gemma Vision.

    Operates on either a file path or in-memory file bytes.
    Images are scaled down to <= 1024px and context length is set to 2048 to fit 4GB VRAM.
    """
    if file_bytes is not None:
        source_name = filename or "uploaded_invoice"
        image_bytes = render_doc_to_image_bytes(
            file_bytes, filename=source_name, max_dimension=1024
        )
        doc_stem = Path(source_name).stem
    elif file_path is not None:
        path = Path(file_path)
        source_name = path.name
        doc_stem = path.stem
        image_bytes = render_doc_to_image_bytes(path, max_dimension=1024)
    else:
        raise ValueError("Either file_path or file_bytes must be provided")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    chosen_model = resolve_model(model_name, base_url=base_url)

    prompt = (
        "You are an expert document extraction system. Examine this invoice image carefully.\n"
        "Extract the following invoice fields into a valid JSON object:\n"
        "- id: The invoice identifier or invoice number (e.g., 'INV-2024-001').\n"
        "- client: The client or company being billed ('Bill To' or 'Customer', NOT the issuer).\n"
        "- issued_on: Date the invoice was issued (in YYYY-MM-DD format).\n"
        "- due_on: Date payment is due (in YYYY-MM-DD format).\n"
        "- amount: Total monetary amount due as a floating point number.\n"
        "- currency: 3-letter currency code (e.g. 'USD', 'EUR', 'GBP').\n"
        "- terms_days: Number of days in the payment term (e.g. 30 for Net 30).\n"
        "- project_type: Project type or service domain (e.g. 'web_dev', 'design', 'consulting').\n"
        "- confidence: Your extraction confidence from 0.0 to 1.0.\n\n"
        "Output ONLY the JSON object matching the schema."
    )

    payload = {
        "model": chosen_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 2048,
        },
    }

    try:
        from payradar.telemetry import trace_gemma_call

        with trace_gemma_call(operation_name="extract", model_name=chosen_model) as tracker:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                tracker.record_response(data)
                raw_text = data.get("message", {}).get("content", "{}")
    except Exception as exc:
        raise RuntimeError(f"Ollama inference error with {chosen_model}: {exc}") from exc

    # Parse JSON output
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try extracting JSON object if enclosed in markdown
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = {}

    # Normalize fields
    inv_id = str(parsed.get("id") or doc_stem).strip()
    client = str(parsed.get("client") or "Unknown Client").strip()
    # Strip common prefixes like 'Bill To:', 'Client:', etc.
    client = re.sub(
        r"^(Bill\s+To:?|Client:?|Customer:?)\s*", "", client, flags=re.IGNORECASE
    ).strip()

    issued_on = normalize_date_string(parsed.get("issued_on"))
    due_on_raw = parsed.get("due_on")
    terms_days = int(parsed.get("terms_days") or 30)

    if due_on_raw:
        due_on = normalize_date_string(due_on_raw)
    else:
        # Fallback due_on calculation from issued_on + terms_days
        try:
            issue_dt = datetime.strptime(issued_on, "%Y-%m-%d")
            due_on = (issue_dt + timedelta(days=terms_days)).strftime("%Y-%m-%d")
        except Exception:
            due_on = issued_on

    amount = clean_currency_amount(parsed.get("amount", 0.0))
    currency = str(parsed.get("currency") or "USD").strip().upper()
    project_type = str(parsed.get("project_type") or "general").strip().lower()

    conf = parsed.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0 else 0.85

    return InvoiceExtraction(
        id=inv_id,
        client=client,
        issued_on=issued_on,
        due_on=due_on,
        amount=amount,
        currency=currency,
        terms_days=terms_days,
        project_type=project_type,
        confidence=confidence,
        source_file=source_name,
    )


def extract_batch(
    file_paths: list[str | Path],
    model_name: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> list[InvoiceExtraction]:
    """Offline batch extraction for a list of document files."""
    results = []
    for fp in file_paths:
        try:
            extraction = extract_invoice(fp, model_name=model_name, base_url=base_url)
            results.append(extraction)
        except Exception:
            # Return partial fallback record with 0 confidence
            p = Path(fp)
            results.append(
                InvoiceExtraction(
                    id=p.stem,
                    client="Extraction Error",
                    issued_on=datetime.now().strftime("%Y-%m-%d"),
                    due_on=datetime.now().strftime("%Y-%m-%d"),
                    amount=0.0,
                    confidence=0.0,
                    source_file=p.name,
                )
            )
    return results
