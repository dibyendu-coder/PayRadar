# Gemma Vision Invoice Extraction Evaluation [SYNTHETIC DATASET]

> **Evaluation Date**: 2026-10-18 15:18:00
> **Model**: `gemma4:e2b` via local Ollama API
> **Notice**: All evaluated documents and ground-truth values are 100% synthetically generated for Hacktoberfest Week 2.

## 1. Summary Metrics

| Metric | Target | Result (Synthetic) |
|---|---|---|
| **Client Name Accuracy** | >90% | **100.0%** (20/20) |
| **Total Amount Accuracy** | >90% | **100.0%** (20/20) |
| **Issue Date Accuracy** | >90% | **100.0%** (20/20) |
| **Due Date Accuracy** | >90% | **100.0%** (20/20) |
| **All 4 Fields Exact Match** | >85% | **100.0%** (20/20) |
| **Mean Confidence Score** | - | **1.00** |
| **Mean Latency per Doc** | <45s | **19.10 seconds** |

## 2. Per-Document Evaluation Breakdown (Seconds per Document)

| # | Document File | Latency (s) | All Fields Match | Client Match | Amount Match | Issued Date Match | Due Date Match | Extracted Client | Extracted Amount |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `inv_01.pdf` | **19.13s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Apex Innovations Accounts Payable | $2,868.25 |
| 02 | `inv_02.pdf` | **19.29s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Vanguard Media | $6,306.04 |
| 03 | `inv_03.pdf` | **20.69s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Echo Dynamics | $7,632.39 |
| 04 | `inv_04.pdf` | **21.69s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Summit Global Logistics | $5,113.60 |
| 05 | `inv_05.pdf` | **18.02s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Cascade Retail Brands | $9,949.61 |
| 06 | `inv_06.pdf` | **18.45s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Orion Enterprise Accounts Payable accounting@orionenterprise.com | $2,914.28 |
| 07 | `inv_07.pdf` | **19.00s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Beacon Health Group | $7,663.21 |
| 08 | `inv_08.pdf` | **21.12s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Atlas Financial Tech | $9,968.11 |
| 09 | `inv_09.pdf` | **17.47s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Zenith Corp | $6,317.01 |
| 10 | `inv_10.pdf` | **17.91s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Prime Legal Solutions | $7,604.27 |
| 11 | `inv_11.pdf` | **18.04s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Quantum Ventures | $6,266.83 |
| 12 | `inv_12.pdf` | **18.08s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Starlight Systems | $9,932.15 |
| 13 | `inv_13.pdf` | **20.78s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Pinnacle Advisory | $6,280.32 |
| 14 | `inv_14.pdf` | **17.57s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Cobalt BioLabs | $6,272.48 |
| 15 | `inv_15.pdf` | **19.01s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Horizon CleanTech | $3,978.17 |
| 16 | `inv_16.pdf` | **21.15s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Nexus Software Labs | $5,017.44 |
| 17 | `inv_17.pdf` | **18.99s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Pulse Digital Studio | $2,907.54 |
| 18 | `inv_18.pdf` | **19.12s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Crestline Partners | $5,013.64 |
| 19 | `inv_19.pdf` | **17.12s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Solaria Renewable Energy | $6,316.82 |
| 20 | `inv_20.pdf` | **19.37s** | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS | Acuity AI Analytics | $1,618.01 |
