# PayRadar 📡

> **Local-first late-payment radar for freelancers.**
> Drop in invoices, and it tells you which ones will be paid late, whether you will have a cash gap in the next 8 weeks, and drafts the right chaser email for each one.

---

### 🏆 Hackathon Window Note
- **Event**: Hacktoberfest Open-Source AI Challenge Week 2 (Side Hustle)
- **Window**: October 12 – October 18, 2026
- **Status**: Created and finished inside this challenge window.
- **Post-Deadline Commits**: None. (Any commit made after October 18, 2026 23:59 UTC will be listed here).

---

> [!NOTE]
> **SYNTHETIC DATA NOTICE**: All benchmark figures, demo invoices, and client profiles used in tests, documentation, and the demo mode are 100% synthetically generated. No real personal or corporate financial data is included or transmitted.

---

## 🔒 Privacy & Non-Negotiables
- **Local-First AI**: Document extraction runs locally using **Gemma 4** (`gemma4:e2b` / `gemma4:e4b`) via [Ollama](https://ollama.ai).
- **Local ML Predictions**: Late payment probability and expected delay predictions run locally using **TabPFN-3.5**.
- **Zero Data Leakage**: Raw invoices never leave your machine. No emails are sent automatically—drafts are generated for user review and manual copying.
- **No Secrets in Git**: All optional keys and tokens come from environment variables (`TABPFN_TOKEN`, `SENTRY_DSN`, `ELEVENLABS_API_KEY`).

---

## 💻 Target Hardware & Stack
- **Target Hardware**: 4 GB VRAM, 15 GB RAM (runs smoothly with `gemma4:e2b`).
- **Python**: 3.11+ managed via `uv`.
- **Extraction**: `pymupdf` (PDF to image conversion) + Gemma Vision via Ollama API. Offline batch processing.
- **Storage**: SQLite local database.
- **Predictive ML**: TabPFN (TabPFN-3.5) with Scikit-learn baselines (`HistGradientBoostingClassifier`, per-client mean).
- **Interface**: Streamlit UI.

---

## 📦 Quickstart

### Prerequisites
1. Install [uv](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Install [Ollama](https://ollama.com) and pull Gemma 4:
   ```bash
   ollama pull gemma4:e2b
   ```

### Installation
```bash
git clone https://github.com/dibyendu-coder/PayRadar.git
cd PayRadar
uv sync --all-extras
```

### Launch Streamlit UI
```bash
# Standard local-first mode (with local Ollama and TabPFN)
uv run streamlit run app.py

# Hosted or lightweight demo mode (precomputed synthetic demo dataset)
DEMO_MODE=1 uv run streamlit run app.py
```

### Run Tests and Predictions
```bash
# Run full unit and integration test suite
uv run pytest

# Re-score open invoices via TabPFN-3.5
uv run python scripts/predict_open.py

# Export precomputed artifacts for DEMO_MODE
uv run python scripts/export_demo.py --templates
```

---

## 🖥️ Application Features (3 Pages)

### 1. 📥 Inbox
- **Document Ingestion**: Upload invoices in PDF, PNG, JPG, or WEBP format.
- **Offline Gemma Vision Extraction**: User-triggered on-demand extraction with confidence scores per field. Images are automatically downsampled to $\le 1024$px to respect the 4 GB VRAM ceiling.
- **Review & Confirm**: Validate extracted metadata (ID, Client, Amount, Currency, Dates, Terms) and save directly to SQLite.

### 2. 🎯 Radar
- **Open Invoice Risk Ranking**: Invoices sorted by late-payment probability $P(\text{Paid} > 7\text{ days late})$ and expected delay days.
- **Cached Predictions**: Instant page loads with zero LLM/TabPFN latency during render; re-scoring runs as an explicit background action.
- **8-Week Rolling Cash Flow & Gap Radar**: Altair interactive bar chart displaying weekly expected cash-in versus prorated monthly operating costs, with cash gap weeks visually flagged in red.
- **Configurable Settings**: Dynamic monthly operating cost slider with immediate forecast recalculation.

### 3. ✉️ Chase
- **Tone Strategy by Risk Tier**:
  - **$< 30\%$ Risk**: Friendly nudge (warm check-in, relationship maintenance).
  - **$30\% - 60\%$ Risk**: Clear & polite (professional status inquiry with terms).
  - **$> 60\%$ Risk**: Firm & specific (urgent notice detailing exact amounts and due dates).
- **Plain-Words Rationale**: Explains *why* the client was assigned their risk tier based strictly on historical invoices issued prior to the target invoice (e.g., *"This client has paid late on 6 of their last 8 invoices (averaging +29.0 days delay)"*).
- **Local-First Safety & Editor**: Edit subject and body directly. User edits are strictly locked and never overwritten. One-click copy container ready for your email client.
- **Graceful Fallbacks**: If Gemma is offline or times out, high-quality deterministic templates are used automatically without application errors.

---

## 📊 Benchmarks & Extraction Accuracy
*(Populated during evaluation phases - all evaluations performed on synthetic datasets)*

| Metric | Target | Result (Synthetic) |
|---|---|---|
| Extraction: Client Accuracy | >90% | **100.0%** (20/20) |
| Extraction: Amount Accuracy | >90% | **100.0%** (20/20) |
| Extraction: Issue Date Accuracy | >90% | **100.0%** (20/20) |
| Extraction: Due Date Accuracy | >90% | **100.0%** (20/20) |
| Extraction: All 4 Fields Exact Match | >85% | **100.0%** (20/20) |
| Extraction: Mean Latency | <30s | **19.90s** / invoice |
| Model ROC-AUC (TabPFN-3.5) | >0.75 | **0.8880** (beats HistGBM: 0.8590) |
| Model MAE (Days Late) | < 7 days | **6.60 days** (beats HistGBM: 8.92d) |

---

## 📚 Credits & Attributions
- [Ollama](https://github.com/ollama/ollama) - Local LLM runtime.
- [Google Gemma](https://github.com/google-deepmind/gemma) - Gemma 4 open weights.
- [TabPFN](https://github.com/automl/TabPFN) - Prior-data Fitted Network for tabular classification and regression.
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - High performance PDF parsing and rendering.
- [Streamlit](https://streamlit.io/) - Interactive application framework.
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms and evaluation.
- [ReportLab](https://www.reportlab.com/) - PDF generation for synthetic invoice suites.
