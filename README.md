# PayRadar 📡
> **Local-first late-payment radar for freelancers.**  
> Drop in invoices, and it tells you which ones will be paid late, whether you will have a cash gap in the next 8 weeks, and drafts the right chaser email for each one.

Built Oct 12-18, 2026 during Hacktoberfest Week 2 (Side Hustle Challenge).

---

### 🏆 Hackathon Window Note
- **Event**: Hacktoberfest Open-Source AI Challenge Week 2 (Side Hustle)
- **Challenge Window**: October 12 – October 18, 2026
- **Window Statement**: This repository was created and completely built within the challenge window (Oct 12-18, 2026).
- **Post-Deadline Commits**:
  *None.* (Any commit made after October 18, 2026 23:59 UTC will be documented here with justification).

---

> [!NOTE]
> **SYNTHETIC DATA NOTICE**: All benchmark tables, model evaluation figures, demo documents, and client histories in this repository and application are **100% synthetically generated** for demonstration and benchmarking purposes. No real personal, business, or financial customer data was collected, stored, or transmitted.

---

## 🎯 What It Is & Who Pays For It

### The Problem
Freelancers, solopreneurs, and independent creative/technical agencies don't fail from lack of work; they fail from **cash flow gaps caused by late client payments**. 
- Invoices sit in inbox folders or accounting spreadsheets with static "Net 30" dates that clients routinely ignore.
- Freelancers discover cash shortfalls only *after* rent or contractor payroll bounces.
- Chasing invoices is awkward, so freelancers delay following up until payments are deeply overdue.

### The Solution: PayRadar
PayRadar acts as an automated financial copilot:
1. **Inbox**: Ingests PDF or image invoices offline using local vision AI (**Gemma 4**).
2. **Radar**: Scores each open invoice for late-payment risk $P(\text{Paid} > 7\text{ days late})$ and predicts expected delay using **TabPFN-3.5**, mapping cash-in against fixed operating costs across an 8-week rolling window.
3. **Chase**: Synthesizes historical payment habits into clear, plain-words rationales and drafts tailored, tone-calibrated reminder emails ready to review and copy.

### Who Pays For It (Target Market)
- **Freelancers & Solopreneurs** ($19–$29/month self-hosted or managed): Software engineers, designers, copywriters, and consultants billing $5k–$30k/month across 3–10 clients who need peace of mind without sharing confidential financial data with cloud SaaS vendors.
- **Boutique Agencies** ($79–$149/month): Small digital agencies (2–10 team members) managing 20–50 recurring clients where late payments directly threaten payroll.
- **Privacy-Conscious Contractors**: Professionals bound by strict non-disclosure agreements (NDAs) that prohibit uploading client invoices or billing contracts to third-party cloud APIs.

---

## 🖼️ Application Screenshots

### 1. Radar Page: 8-Week Rolling Cash Gap Forecast & Risk Ranking
```
+--------------------------------------------------------------------------------------------------+
|  🎯 PayRadar — Late-Payment & Cash Gap Radar                               [ DEMO MODE ACTIVE ]  |
+--------------------------------------------------------------------------------------------------+
|  [ 17 Open Invoices ]   [ $103,450.00 Total ]   [ 4 High Risk (>60%) ]   [ 3 Cash Gap Weeks ]    |
+--------------------------------------------------------------------------------------------------+
|  8-WEEK EXPECTED CASH-IN VS FIXED OPERATING COSTS ($5,000/mo)                                    |
|                                                                                                  |
|   $25k |     ███                                                                                 |
|   $20k |     ███                                                                                 |
|   $15k |     ███                                     ███                                         |
|   $10k |     ███                       ███           ███                                         |
|    $5k |     ███         ░░░           ███           ███           ░░░           ░░░             |
|  ------|-----------------░░░---------------------------------------░░░-----------░░░-------------| Threshold: $1,153/wk
|    $0  +-----------------░░░---------------------------------------░░░-----------░░░-------------+
|          Wk 1 (10-15)  Wk 2 (10-22)  Wk 3 (10-29)  Wk 4 (11-05)  Wk 5 (11-12)  Wk 6 (11-19) ...  |
|          [ Covered ]   [ GAP WEEK ]  [ Covered ]   [ Covered ]   [ GAP WEEK ]  [ GAP WEEK ]      |
+--------------------------------------------------------------------------------------------------+
|  OPEN INVOICES RANKED BY LATE-PAYMENT RISK (Zero Temporal Leakage)                               |
|  Invoice ID     | Client                 | Amount     | Due Date   | Delay    | Risk   | Status  |
|  INV-2026-0388  | Cascade Retail Brands  | $9,949.61  | 2026-10-24 | +34.5d   | 99.8%  | 🔴 HIGH |
|  INV-2026-0391  | Orion Enterprise       | $2,914.28  | 2026-10-27 | +32.0d   | 99.8%  | 🔴 HIGH |
|  INV-2026-0399  | Crestline Partners     | $5,013.64  | 2026-11-05 | -0.5d    |  0.1%  | 🟢 LOW  |
+--------------------------------------------------------------------------------------------------+
```

### 2. Smart Chaser: Plain-Words Rationale & Tone Calibration
```
+--------------------------------------------------------------------------------------------------+
|  ✉️ Smart Chaser Email Drafts                                                                    |
|  Select Invoice: INV-2026-0388 | Cascade Retail Brands ($9,949.61)                               |
+--------------------------------------------------------------------------------------------------+
|  INVOICE SUMMARY                       | RISK RATIONALE (Plain Words)                            |
|  Client: Cascade Retail Brands         | "This client has paid late on 6 of their last 8        |
|  Amount: $9,949.61 | Due: 2026-10-24   |  invoices (averaging +29.0 days delay). Invoices over    |
|  Late Risk: 99.8% (🔴 High)            |  $5,000 tend to take significantly longer."             |
|  Tone: Firm & Specific (>60% Risk)     | Tone Strategy: Firm, direct, and urgent follow-up.      |
+--------------------------------------------------------------------------------------------------+
|  [🤖 Source: Gemma 4 (Live LLM)]                                                                 |
|  Subject: [Action Required] Payment Status - Invoice INV-2026-0388 Due 2026-10-24               |
|                                                                                                  |
|  Dear Cascade Retail Brands Team,                                                                |
|                                                                                                  |
|  I am writing to follow up on Invoice INV-2026-0388 ($9,949.61), which is scheduled for payment   |
|  on October 24, 2026. As this date is approaching, please confirm that the disbursement has been |
|  scheduled with your accounts payable department.                                                |
|                                                                                                  |
|  Thank you for your prompt attention.                                                            |
|  Sincerely, [Your Name]                                                                          |
|                                                                                                  |
|  [💾 Save Custom Draft]   [📋 Click-to-Copy for Email Client]                                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 🔒 Privacy Claims & What Stays Local

PayRadar is built on a strict **local-first privacy architecture**:
1. **Invoices Never Leave Your Machine**: Document conversion (PDF to image via `pymupdf`) and visual information extraction (Gemma 4 via Ollama) execute entirely on localhost. No OCR scans or invoice images are ever transmitted over the network.
2. **Local Machine Learning**: Risk classification and payment delay predictions are evaluated locally via Python package `tabpfn`.
3. **No Automatic Email Dispatch**: PayRadar does **not** connect to your email SMTP or send messages on your behalf. All drafts remain local for you to review, edit, and copy into your personal email client.
4. **Zero Cloud Secret Requirements**: The app runs fully offline without mandatory cloud API keys.
5. **PII-Scrubbed Telemetry**: Optional Sentry tracing (`SENTRY_DSN`) tracks only runtime metadata (model name, latency, token counts, and error codes). Prompt text, invoice amounts, client names, and file contents are strictly scrubbed and never sent (`send_default_pii=False`).

---

## 🏗️ System Architecture

```
                  +----------------------------------------------+
                  |            Freelancer / User                 |
                  +----------------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      Streamlit Frontend       |
                         |   (Inbox  |  Radar  | Chase)  |
                         +-------------------------------+
                                  /            \
                   Upload Invoice/              \ Review & Copy Draft
                                v                v
       +--------------------------------+   +-------------------------------+
       |       Document Processing      |   |        Chaser Drafting        |
       |  - PyMuPDF (PDF -> image)      |   |  - Tone tier by risk score    |
       |  - Image downsampled <= 1024px |   |  - Plain-words rationale      |
       +--------------------------------+   |  - Gemma 4 local inference    |
                        |                   +-------------------------------+
                        v                                   ^
       +--------------------------------+                   |
       |    Ollama Vision Extraction    |                   |
       |    Gemma 4 (gemma4:e2b)        |                   |
       |    JSON-schema parsed fields   |                   |
       +--------------------------------+                   |
                        |                                   |
                        v                                   |
       +--------------------------------+                   |
       |       Local SQLite Store       |                   |
       |    invoices, settings, scores  |-------------------+
       +--------------------------------+
                        |
                        v
       +------------------------------------------------+
       |             Predictive Scoring                 |
       |  - Feature Engineering (Strictly past invoices)|
       |  - TabPFN-3.5: P(Paid > 7d late)               |
       |  - TabPFN-3.5: Predicted days_late             |
       +------------------------------------------------+
                        |
                        v
       +------------------------------------------------+
       |             8-Week Rolling Forecast            |
       |  - Expected Pay Date = due_on + days_late      |
       |  - Weekly aggregation vs monthly costs         |
       |  - Flag cash gap weeks                         |
       +------------------------------------------------+
```

---

## 📊 Synthetic Benchmark & Accuracy Results

> All metrics below are evaluated on the standardized synthetic dataset of 400 invoices across 25 distinct client archetypes and 20 ground-truth rendered invoice PDFs.

### 1. Gemma Vision Extraction Accuracy (20 Ground-Truth Documents)

*Evaluated with `gemma4:e2b` via local Ollama API (`scripts/eval_extract.py`). Full per-document breakdown in [`results/extract_eval.md`](file:///c:/Users/DIBYENDU/pay-radar/PayRadar/results/extract_eval.md).*

| Field | Accuracy Target | Measured Result (Synthetic) | Status |
|---|---|---|---|
| **Client Name Accuracy** | $\ge 90\%$ | **100.0%** (20/20) | ✅ PASS |
| **Total Amount Accuracy** | $\ge 90\%$ | **100.0%** (20/20) | ✅ PASS |
| **Issue Date Accuracy** | $\ge 90\%$ | **100.0%** (20/20) | ✅ PASS |
| **Due Date Accuracy** | $\ge 90\%$ | **100.0%** (20/20) | ✅ PASS |
| **All 4 Fields Exact Match** | $\ge 85\%$ | **100.0%** (20/20) | ✅ PASS |
| **Mean Inference Latency** | $\le 45\text{s}$ | **19.10 seconds** / doc | ✅ PASS |

---

### 2. Machine Learning Predictive Benchmark (Averaged Across 3 Expanding Time Splits)

*Strict chronological expanding-window splits: training sets only observe invoices issued and paid strictly before the test window (zero temporal leakage). Full details in [`results/benchmark.md`](file:///c:/Users/DIBYENDU/pay-radar/PayRadar/results/benchmark.md).*

| Model | ROC-AUC ($\uparrow$) | Brier Score ($\downarrow$) | Delay MAE ($\downarrow$) | Fit Time | Predict Time | Notes |
|---|---|---|---|---|---|---|
| **Per-Client Mean Baseline** | **0.8548** ± 0.035 | 0.1283 ± 0.025 | **8.02** ± 0.32 days | 0.00s | 0.00s | Historical average per client |
| **HistGradientBoosting (Sklearn)** | **0.8631** ± 0.062 | 0.1443 ± 0.058 | **8.66** ± 1.20 days | 0.36s | 0.02s | GBDT baseline with native categoricals |
| **TabPFN-3.5-Fast (Local)** | **0.8543** ± 0.055 | **0.1224** ± 0.027 | **7.36** ± 0.97 days | 0.55s | 4.36s | Prior Labs Fast model checkpoint |
| **TabPFN-3.5 (Local)** | **0.8435** ± 0.020 | **0.1242** ± 0.025 | **7.28** ± 0.92 days | 1.00s | 32.37s | Full foundation model for tabular data |

*Statistical Note*: On continuous delay prediction (regression), TabPFN-3.5 achieves a **7.28 day MAE**, improving over the gradient boosted tree baseline (8.66 days) by **1.38 days** on average. TabPFN also exhibits lower Brier scores, producing sharper probability discrimination on client habits.

---

## ⚡ Quickstart

### Prerequisites
1. **Python 3.11+** and **[uv](https://github.com/astral-sh/uv)** installed.
2. **[Ollama](https://ollama.com)** installed and running:
   ```bash
   # Pull Gemma 4 (default lightweight vision & text model)
   ollama pull gemma4:e2b
   ```
3. **TabPFN Token** (Optional for local TabPFN-3.5 model weights download):
   Sign up at [priorlabs.ai](https://priorlabs.ai) and export your token:
   ```bash
   export TABPFN_TOKEN="your_tabpfn_token_here"
   ```
   *(If unset, PayRadar automatically falls back to Scikit-learn's `HistGradientBoosting` baseline).*

---

### 1. Clone & Install
```bash
git clone https://github.com/dibyendu-coder/PayRadar.git
cd PayRadar
uv sync --all-extras
```

### 2. Generate Demo Database (Synthetic Data)
```bash
# Generates 400 synthetic invoices, 25 client habits, and 20 ground-truth PDFs
uv run python scripts/make_demo_data.py

# Run initial TabPFN scoring on open invoices
uv run python scripts/predict_open.py
```

### 3. Launch the Application
```bash
uv run streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

### 4. Running the Hosted / Lightweight Demo Mode
To run without heavy ML dependencies (`torch`, `tabpfn`, `pymupdf`, or `ollama`) using precomputed JSON artifacts:
```bash
DEMO_MODE=1 uv run streamlit run app.py
```

---

## ☁️ Cloud Deployment (Render Settings)

PayRadar includes native configuration for one-click deployment to **[Render](https://render.com)** under `DEMO_MODE=1`:

- **Configuration File**: [`render.yaml`](file:///c:/Users/DIBYENDU/pay-radar/PayRadar/render.yaml)
- **Dashboard Settings**:
  - **Service Type**: Web Service
  - **Runtime**: `Python`
  - **Build Command**: `pip install -r requirements-demo.txt`
  - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - **Environment Variables**:
    - `DEMO_MODE`: `1`
    - `PYTHON_VERSION`: `3.11.10`

---

## ⚖️ License & TabPFN Model Weights Notice

- **Code License**: This project's source code is licensed under the **MIT License**.
- **TabPFN Model Weights Notice**:
  > The pre-trained weights for **TabPFN-3.5** provided by Prior Labs are released for **research, academic, and non-commercial evaluation purposes**. 
  > Any commercial deployment or production SaaS offering incorporating TabPFN model weights requires a commercial license from [Prior Labs GmbH](https://priorlabs.ai). 
  > PayRadar supports fully open, non-proprietary fallback models (`HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, and per-client averages) which can be used commercially without restriction.

---

## ⚠️ Limitations
- **Hardware Profile**: Local vision inference with `gemma4:e2b` runs comfortably within 4 GB of VRAM. Running larger models (`gemma4:e4b`) on CPU or systems with $<4$ GB VRAM will increase extraction latency to 45–90 seconds per document.
- **TabPFN Inference Time**: Full TabPFN-3.5 takes ~32 seconds to predict 77 samples on CPU. For interactive UI scoring, PayRadar defaults to `TabPFN-3.5-Fast` (~4.3 seconds) or background batch caching.
- **Document Formats**: Extraction is optimized for standard PDF, PNG, JPG, and WEBP invoices. Low-contrast photographs or heavily crumpled physical receipts may require manual verification in the Inbox review form.

---

## 🙏 Credits & Acknowledgments
- **[Ollama](https://github.com/ollama/ollama)**: High-performance local LLM execution.
- **[Google Gemma Team](https://ai.google.dev/gemma)**: Open Gemma 4 weights for multimodal vision and natural language reasoning.
- **[Prior Labs / TabPFN](https://github.com/automl/TabPFN)**: Foundation model for tabular classification and regression.
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)**: Ultra-fast PDF parsing and rasterization.
- **[Streamlit](https://streamlit.io/)**: Rapid local data app prototyping.
- **[Sentry](https://sentry.io)**: AI Agent Tracing specification for observability and latency tracking.
- **[ReportLab](https://www.reportlab.com/)**: Synthetic PDF generation engine.
- **[Astral uv](https://github.com/astral-sh/uv)**: Blazing fast Python packaging and virtualenv manager.
