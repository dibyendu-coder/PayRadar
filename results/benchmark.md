# PayRadar Model Benchmark 📊

> [!NOTE]
> **SYNTHETIC DATA NOTICE**: All benchmarks are calculated on 100% synthetically generated invoice records.
> No real customer or transaction data was used.

## Evaluation Setup
- **Split Strategy**: Chronological Time-Based Split (Oldest 70% Train, Newest 30% Test)
- **Train Set**: 268 invoices (2023-11-02 to 2025-10-10)
- **Test Set**: 115 invoices (2025-10-13 to 2026-09-12)
- **Classification Task**: $P(\text{Paid} > 7\text{ days late})$ (Metric: ROC-AUC)
- **Regression Task**: Continuous delay in days (Metric: Mean Absolute Error)

## Results Table

| Model | Classification ROC-AUC | Delay MAE (Days) | Fit Time (s) | Predict Time (s) | Status |
|---|---|---|---|---|---|
| **Per-Client Mean Baseline** | 0.8740 | 8.02 | 0.004s | 0.000s | Evaluated |
| **HistGradientBoosting (Sklearn)** | 0.8590 | 8.92 | 0.614s | 0.026s | Evaluated |
| **TabPFN-3.5 (Local)** | N/A | N/A | N/A | N/A | Requires TABPFN_TOKEN env var |
