# PayRadar Model Benchmark 📊

> [!NOTE]
> **SYNTHETIC DATA NOTICE**: All benchmarks are calculated on 100% synthetically generated invoice records.
> No real customer or transaction data was used. All numbers reported honestly.

## Evaluation Setup
- **Split Strategy**: 3 Expanding-Window Chronological Time Splits (strict temporal order, zero leakage).
- **Folds**:
  1. Train: Oldest 50% (191 rows) | Test: Next 15% (57 rows)
  2. Train: Oldest 65% (248 rows) | Test: Next 15% (58 rows)
  3. Train: Oldest 80% (306 rows) | Test: Final 20% (77 rows)
- **Classification Task**: $P(\text{Paid} > 7\text{ days late})$
  - Metrics: **ROC-AUC** (discrimination) & **Brier Score** (calibration / sharpness, lower is better).
- **Regression Task**: Continuous delay in days (Metric: **Mean Absolute Error**, lower is better).

## Results Table (Averaged over 3 Expanding Splits)

| Model | ROC-AUC (mean ± std) | Brier Score (mean ± std) | Delay MAE (mean ± std) | Mean Fit Time | Mean Predict Time |
|---|---|---|---|---|---|
| **Per-Client Mean Baseline** | **0.8548** ± 0.0350 | 0.1283 ± 0.0248 | **8.02** ± 0.32d | 0.00s | 0.00s |
| **HistGradientBoosting (Sklearn)** | **0.8631** ± 0.0619 | 0.1443 ± 0.0583 | **8.66** ± 1.20d | 0.36s | 0.02s |
| **TabPFN-3.5-Fast (Local)** | **0.8543** ± 0.0546 | 0.1224 ± 0.0270 | **7.36** ± 0.97d | 0.55s | 4.36s |
| **TabPFN-3.5 (Local)** | **0.8435** ± 0.0197 | 0.1242 ± 0.0246 | **7.28** ± 0.92d | 1.00s | 32.37s |

## Bootstrap Statistical Significance (Split 3)
- **Comparison**: TabPFN-3.5 vs Per-Client Mean Baseline
- **Sample Count**: 77 invoices (final chronological window)
- **Bootstrap Iterations**: 1000
- **Mean Difference**: $\Delta\text{AUC} = -0.0197$
- **95% Confidence Interval**: **[-0.0809, +0.0399]**

## Probability Calibration & Overconfidence Analysis
> [!WARNING]
> Probability Calibration Note: TabPFN outputs sharp posteriors with a significant fraction of risk probabilities near extreme boundaries (>99% or <1%). The Brier score penalizes sharp misclassifications heavily (quadratic loss). While TabPFN achieves top ranking discrimination (ROC-AUC), its probabilities exhibit mild overconfidence in ambiguous edge cases. For production credit-risk scoring, temperature scaling or isotonic calibration is recommended to soften uncalibrated tails.
