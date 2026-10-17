# PayRadar: agent brief

Local-first late-payment radar for freelancers. Drop in invoices, and it tells you
which ones will be paid late, whether you will have a cash gap in the next 8 weeks,
and drafts the right chaser email for each one.

Built for Hacktoberfest Open-Source AI Challenge Week 2 (Side Hustle).
Window: Oct 12-18, 2026. This repo must be NEW and finished inside that window.
README must say so, and list any commit made after the deadline.

## Non-negotiables
- Open pieces do the real work: Gemma 4 (Ollama) reads documents, TabPFN-3.5 (local package) predicts.
- Raw invoices never leave the machine. Nothing is sent anywhere automatically, including emails.
- Demo data is synthetic and labelled synthetic everywhere (UI, README, benchmark output).
- No secrets in git. Keys come from env vars (TABPFN_TOKEN, SENTRY_DSN, ELEVENLABS_API_KEY).
- Small commits with real messages. Judges read the history.
- Credit every library and any code you borrow in the README.

## Stack
Python 3.11+, uv. Ollama with `gemma4:e4b` (fallback `gemma4:e2b`, then `gemma3:4b`; check `ollama list`).
pymupdf (PDF to image), sqlite3, pandas, scikit-learn (baselines only), tabpfn (local, TabPFN-3.5),
Streamlit UI, pytest, GitHub Actions. Optional: sentry-sdk, elevenlabs.
Target hardware: 4 GB VRAM, 15 GB RAM. Default to gemma4:e2b via GEMMA_MODEL; extraction runs as an offline batch, never inside a UI request.

## Data model (SQLite)
invoices(id, client, issued_on, due_on, amount, currency, terms_days, project_type, paid_on NULL, source_file)
settings(key, value)  -- includes monthly_costs

## Modules
- `payradar/extract.py`   Gemma vision call, JSON-schema constrained output, returns Invoice fields + confidence.
- `payradar/features.py`  Features per invoice: log_amount, terms_days, issue_month, issue_dow, client,
                          project_type, client_prev_count, client_prev_mean_days_late, amount_vs_client_median.
                          Client history features MUST use only invoices issued earlier (no leakage).
- `payradar/model.py`     TabPFNClassifier -> P(paid > 7 days late). TabPFNRegressor -> days_late.
- `payradar/bench.py`     Time-based split (oldest 70% train, newest 30% test). Compare TabPFN vs
                          (a) per-client mean, (b) HistGradientBoosting. Report ROC-AUC and MAE.
                          Writes results/benchmark.json and results/benchmark.md.
- `payradar/forecast.py`  Expected pay date = due_on + predicted days_late. Weekly cash-in for 8 weeks
                          vs monthly_costs. Flag gap weeks.
- `payradar/chase.py`     Gemma drafts a reminder. Tone by risk: <0.3 friendly nudge, 0.3-0.6 clear
                          and polite, >0.6 firm. Draft only, user copies it.
- `app.py`                Streamlit: Inbox (upload, review extracted fields, confirm), Radar (open invoices
                          by risk + weekly cash chart), Chase (draft and copy).
- `scripts/make_demo_data.py`  Seeded generator: 25 clients with distinct habits (always late, pays on the
                          30th, big invoices slower, one reliable), ~400 invoices over 3 years, plus
                          ~20 rendered invoice images/PDFs in 5+ layouts with ground-truth JSON.
- `DEMO_MODE=1` loads precomputed JSON from `demo/`, disables upload and live Gemma. Used for the hosted demo.

## Phases and acceptance
P0 Scaffold: uv project, ruff, pytest, CI workflow, README stub with the window note.
P1 Data + extraction: demo data script; extract.py; `scripts/eval_extract.py` prints field-level
   accuracy on the 20 ground-truth docs (client, amount, issue date, due date).
P2 Models: features, model, bench, forecast. Tests for leakage (a feature for invoice N never sees N or later).
P3 UI + chase: all three pages work end to end on demo data locally.
P4 Ship: DEMO_MODE, Sentry tracing around every Gemma call (latency, tokens, errors), Render deploy,
   README with screenshots and the benchmark table.

## Definition of done
`uv run streamlit run app.py` works from a clean clone with only Ollama + a TabPFN token set up.
Benchmark table and extraction accuracy are in the README, labelled synthetic.
Hosted demo loads in DEMO_MODE. Tests and CI are green.

## First prompt to paste
Read AGENTS.md fully. Do Phase 0 and Phase 1 only. Before writing code, list the files you will create
and anything in the brief you think is wrong or missing. Then build it, commit after each working step,
and stop when `eval_extract.py` prints accuracy numbers.