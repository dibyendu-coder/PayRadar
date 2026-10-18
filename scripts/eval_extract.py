"""Evaluate Gemma Vision extraction accuracy on 20 synthetic ground-truth invoices.

Evaluates field-level accuracy:
- Client name
- Total amount
- Issue date
- Due date
- Full document match (all 4 fields correct)

Outputs detailed table and writes results/extraction_eval.json.
Notice: All evaluation is performed on 100% synthetic invoice documents.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from payradar.extract import extract_invoice

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
GT_DIR = DEMO_DIR / "ground_truth"
GT_META_DIR = GT_DIR / "metadata"
GT_INVOICES_DIR = GT_DIR / "invoices"
RESULTS_DIR = PROJECT_ROOT / "results"


def normalize_str(s: str | None) -> str:
    """Normalize string for robust comparison."""
    if not s:
        return ""
    return "".join(c.lower() for c in s if c.isalnum())


def is_client_match(extracted: str, truth: str) -> bool:
    """Check if client name matches with normalization."""
    ext_norm = normalize_str(extracted)
    tru_norm = normalize_str(truth)
    if not ext_norm or not tru_norm:
        return False
    return ext_norm == tru_norm or tru_norm in ext_norm or ext_norm in tru_norm


def is_amount_match(extracted: float, truth: float, tolerance_pct: float = 0.01) -> bool:
    """Check if monetary amount matches within tolerance (1% or $0.50)."""
    if abs(extracted - truth) <= 0.50:
        return True
    if truth > 0 and abs(extracted - truth) / truth <= tolerance_pct:
        return True
    return False


def is_date_match(extracted: str, truth: str) -> bool:
    """Check if date strings match in YYYY-MM-DD."""
    return extracted.strip() == truth.strip()


def run_evaluation(
    limit: int = 20,
    model_name: str | None = None,
    doc_format: str = "pdf",
) -> dict:
    """Run extraction benchmark on ground-truth documents and return accuracy metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(GT_META_DIR.glob("inv_*.json"))
    if not meta_files:
        print("[ERROR] No ground-truth metadata files found. Run scripts/make_demo_data.py first.")
        sys.exit(1)

    eval_targets = meta_files[:limit]
    total_docs = len(eval_targets)

    print("=" * 80)
    print("PAYRADAR EXTRACTION EVALUATION [SYNTHETIC DATASET]")
    print(
        f"Evaluating {total_docs} ground-truth invoices using Gemma Vision ({model_name or os.environ.get('GEMMA_MODEL', 'gemma4:e2b')})"
    )
    print(f"Target format: .{doc_format}")
    print("=" * 80)

    client_correct = 0
    amount_correct = 0
    issued_correct = 0
    due_correct = 0
    all_correct = 0
    total_latency = 0.0
    confidences = []

    doc_results = []

    for i, meta_file in enumerate(eval_targets, start=1):
        with open(meta_file, "r", encoding="utf-8") as f:
            truth = json.load(f)

        file_stem = meta_file.stem
        invoice_file = GT_INVOICES_DIR / f"{file_stem}.{doc_format}"
        if not invoice_file.exists():
            print(f"[WARN] Document {invoice_file.name} not found, skipping.")
            continue

        start_time = time.time()
        try:
            extracted = extract_invoice(invoice_file, model_name=model_name)
            latency = time.time() - start_time
        except Exception as e:
            latency = time.time() - start_time
            print(f"[{i:02d}/{total_docs:02d}] {invoice_file.name} -> EXTRACTION ERROR: {e}")
            doc_results.append(
                {
                    "file": invoice_file.name,
                    "error": str(e),
                    "client_match": False,
                    "amount_match": False,
                    "issued_match": False,
                    "due_match": False,
                    "all_match": False,
                    "latency_sec": latency,
                }
            )
            continue

        c_match = is_client_match(extracted.client, truth["client"])
        a_match = is_amount_match(extracted.amount, truth["amount"])
        i_match = is_date_match(extracted.issued_on, truth["issued_on"])
        d_match = is_date_match(extracted.due_on, truth["due_on"])
        doc_all = c_match and a_match and i_match and d_match

        if c_match:
            client_correct += 1
        if a_match:
            amount_correct += 1
        if i_match:
            issued_correct += 1
        if d_match:
            due_correct += 1
        if doc_all:
            all_correct += 1

        total_latency += latency
        confidences.append(extracted.confidence)

        status_symbols = f"C:{'OK' if c_match else 'FAIL'} A:{'OK' if a_match else 'FAIL'} I:{'OK' if i_match else 'FAIL'} D:{'OK' if d_match else 'FAIL'}"
        print(
            f"[{i:02d}/{total_docs:02d}] {invoice_file.name} ({latency:.1f}s) | {status_symbols} | Client: '{extracted.client}' | Amt: ${extracted.amount:,.2f}"
        )

        doc_results.append(
            {
                "file": invoice_file.name,
                "ground_truth": {
                    "client": truth["client"],
                    "amount": truth["amount"],
                    "issued_on": truth["issued_on"],
                    "due_on": truth["due_on"],
                },
                "extracted": {
                    "client": extracted.client,
                    "amount": extracted.amount,
                    "issued_on": extracted.issued_on,
                    "due_on": extracted.due_on,
                    "confidence": extracted.confidence,
                },
                "matches": {
                    "client": c_match,
                    "amount": a_match,
                    "issued_on": i_match,
                    "due_on": d_match,
                    "all_fields": doc_all,
                },
                "latency_sec": round(latency, 2),
            }
        )

    client_acc = (client_correct / total_docs) * 100 if total_docs > 0 else 0
    amount_acc = (amount_correct / total_docs) * 100 if total_docs > 0 else 0
    issued_acc = (issued_correct / total_docs) * 100 if total_docs > 0 else 0
    due_acc = (due_correct / total_docs) * 100 if total_docs > 0 else 0
    full_doc_acc = (all_correct / total_docs) * 100 if total_docs > 0 else 0
    avg_latency = total_latency / total_docs if total_docs > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print("\n" + "=" * 80)
    print("EXTRACTION ACCURACY BENCHMARK SUMMARY (SYNTHETIC)")
    print("=" * 80)
    print(f"Total Evaluated Documents : {total_docs}")
    print(f"Client Accuracy           : {client_acc:.1f}% ({client_correct}/{total_docs})")
    print(f"Amount Accuracy           : {amount_acc:.1f}% ({amount_correct}/{total_docs})")
    print(f"Issue Date Accuracy       : {issued_acc:.1f}% ({issued_correct}/{total_docs})")
    print(f"Due Date Accuracy         : {due_acc:.1f}% ({due_correct}/{total_docs})")
    print(f"All 4 Fields Exact Match  : {full_doc_acc:.1f}% ({all_correct}/{total_docs})")
    print(f"Mean Confidence Score     : {avg_conf:.2f}")
    print(f"Mean Extraction Latency   : {avg_latency:.2f} s / invoice")
    print("=" * 80)

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Synthetic Ground Truth",
        "synthetic": True,
        "total_documents": total_docs,
        "metrics": {
            "client_accuracy_pct": round(client_acc, 2),
            "amount_accuracy_pct": round(amount_acc, 2),
            "issue_date_accuracy_pct": round(issued_acc, 2),
            "due_date_accuracy_pct": round(due_acc, 2),
            "full_document_accuracy_pct": round(full_doc_acc, 2),
            "mean_confidence": round(avg_conf, 3),
            "mean_latency_seconds": round(avg_latency, 2),
        },
        "details": doc_results,
    }

    out_file = RESULTS_DIR / "extraction_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to {out_file.relative_to(PROJECT_ROOT)}")

    # Write results/extract_eval.md including seconds per document
    chosen_model = model_name or os.environ.get("GEMMA_MODEL", "gemma4:e2b")
    md_content = f"""# Gemma Vision Invoice Extraction Evaluation [SYNTHETIC DATASET]

> **Evaluation Date**: {summary["evaluation_timestamp"]}
> **Model**: `{chosen_model}` via local Ollama API
> **Notice**: All evaluated documents and ground-truth values are 100% synthetically generated for Hacktoberfest Week 2.

## 1. Summary Metrics

| Metric | Target | Result (Synthetic) |
|---|---|---|
| **Client Name Accuracy** | >90% | **{client_acc:.1f}%** ({client_correct}/{total_docs}) |
| **Total Amount Accuracy** | >90% | **{amount_acc:.1f}%** ({amount_correct}/{total_docs}) |
| **Issue Date Accuracy** | >90% | **{issued_acc:.1f}%** ({issued_correct}/{total_docs}) |
| **Due Date Accuracy** | >90% | **{due_acc:.1f}%** ({due_correct}/{total_docs}) |
| **All 4 Fields Exact Match** | >85% | **{full_doc_acc:.1f}%** ({all_correct}/{total_docs}) |
| **Mean Confidence Score** | - | **{avg_conf:.2f}** |
| **Mean Latency per Doc** | <45s | **{avg_latency:.2f} seconds** |

## 2. Per-Document Evaluation Breakdown (Seconds per Document)

| # | Document File | Latency (s) | All Fields Match | Client Match | Amount Match | Issued Date Match | Due Date Match | Extracted Client | Extracted Amount |
|---|---|---|---|---|---|---|---|---|---|
"""
    for idx, d in enumerate(doc_results, start=1):
        m = d.get("matches", {})
        all_ok = "✅ PASS" if m.get("all_fields") else "❌ FAIL"
        c_ok = "✅ PASS" if m.get("client") else "❌ FAIL"
        a_ok = "✅ PASS" if m.get("amount") else "❌ FAIL"
        i_ok = "✅ PASS" if m.get("issued_on") else "❌ FAIL"
        due_ok = "✅ PASS" if m.get("due_on") else "❌ FAIL"
        ext = d.get("extracted", {})
        ext_client = ext.get("client", "Error")
        ext_amt = f"${ext.get('amount', 0.0):,.2f}" if "amount" in ext else "Error"
        lat = d.get("latency_sec", 0.0)
        md_content += f"| {idx:02d} | `{d['file']}` | **{lat:.2f}s** | {all_ok} | {c_ok} | {a_ok} | {i_ok} | {due_ok} | {ext_client} | {ext_amt} |\n"

    md_file = RESULTS_DIR / "extract_eval.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Markdown report saved to {md_file.relative_to(PROJECT_ROOT)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma Vision invoice extraction")
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of documents to evaluate (default 20)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Ollama model name (default gemma4:e2b)"
    )
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png"], help="Document format to test"
    )
    args = parser.parse_args()

    run_evaluation(limit=args.limit, model_name=args.model, doc_format=args.format)


if __name__ == "__main__":
    main()
