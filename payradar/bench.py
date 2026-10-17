"""Benchmark comparing TabPFN against per-client mean and HistGradientBoosting baselines.

Non-negotiable requirements:
- Time-based split: Oldest 70% train, newest 30% test.
- Baselines:
  1. Per-client empirical mean.
  2. Scikit-learn HistGradientBoosting (Classifier & Regressor).
  3. TabPFN-3.5 (TabPFNClassifier & TabPFNRegressor).
- Metrics: ROC-AUC (classification), MAE (regression), Fit time (s), Predict time (s).
- Outputs: results/benchmark.json and results/benchmark.md.
- Every output labelled "synthetic data".
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from payradar.db import get_all_invoices
from payradar.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_feature_dataset
from payradar.model import (
    LatePaymentClassifier,
    MissingTabPFNTokenError,
    PaymentDelayRegressor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def get_time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train/test split based on issued_on.

    Guarantees no future test invoice is issued before any training invoice.
    """
    sorted_df = df.sort_values(by=["issued_on", "id"]).reset_index(drop=True)
    n_train = int(len(sorted_df) * train_ratio)
    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_train:].copy().reset_index(drop=True)
    return train_df, test_df


# --- Baseline 1: Per-Client Mean Baseline ---


class PerClientMeanBaseline:
    """Predicts historical client empirical late rate and mean delay."""

    def __init__(self):
        self.client_clf_rates: dict[str, float] = {}
        self.client_reg_means: dict[str, float] = {}
        self.global_clf_rate: float = 0.0
        self.global_reg_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y_clf: np.ndarray, y_reg: np.ndarray) -> PerClientMeanBaseline:
        clients = X["client"].astype(str)
        self.global_clf_rate = float(np.mean(y_clf))
        self.global_reg_mean = float(np.mean(y_reg))

        df_train = pd.DataFrame({"client": clients, "y_clf": y_clf, "y_reg": y_reg})
        grouped = df_train.groupby("client")

        self.client_clf_rates = grouped["y_clf"].mean().to_dict()
        self.client_reg_means = grouped["y_reg"].mean().to_dict()
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.client_clf_rates.get(str(c), self.global_clf_rate) for c in X["client"]]
        )

    def predict_delay(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.client_reg_means.get(str(c), self.global_reg_mean) for c in X["client"]]
        )


# --- Benchmark Runner ---


def run_benchmark(db_path: str | Path | None = None) -> dict[str, Any]:
    """Execute complete benchmark comparing TabPFN with baselines on synthetic dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    raw_invoices = get_all_invoices(db_path=db_path)
    if not raw_invoices:
        # Fallback to demo invoices JSON if DB is not populated
        demo_json = PROJECT_ROOT / "demo" / "invoices_history.json"
        if demo_json.exists():
            with open(demo_json, "r", encoding="utf-8") as f:
                raw_invoices = json.load(f)

    # Filter to paid invoices for model training/testing
    paid_invoices = [inv for inv in raw_invoices if inv.get("paid_on") is not None]
    if len(paid_invoices) < 20:
        raise RuntimeError("Insufficient historical paid invoices for benchmark.")

    # 2. Build feature dataset
    df_all = build_feature_dataset(paid_invoices, include_targets=True)

    # 3. Time-based 70/30 split
    train_df, test_df = get_time_based_split(df_all, train_ratio=0.70)

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train_clf = train_df["is_late_7d"].values.astype(int)
    y_train_reg = train_df["days_late"].values.astype(float)

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test_clf = test_df["is_late_7d"].values.astype(int)
    y_test_reg = test_df["days_late"].values.astype(float)

    results_table = []

    # Model 1: Per-Client Mean Baseline
    t0 = time.time()
    client_baseline = PerClientMeanBaseline().fit(X_train, y_train_clf, y_train_reg)
    fit_time_base = time.time() - t0

    t0 = time.time()
    pred_risk_base = client_baseline.predict_risk(X_test)
    pred_delay_base = client_baseline.predict_delay(X_test)
    pred_time_base = time.time() - t0

    auc_base = float(roc_auc_score(y_test_clf, pred_risk_base))
    mae_base = float(mean_absolute_error(y_test_reg, pred_delay_base))

    results_table.append({
        "model": "Per-Client Mean Baseline",
        "classifier_auc": round(auc_base, 4),
        "regressor_mae": round(mae_base, 2),
        "fit_time_sec": round(fit_time_base, 3),
        "predict_time_sec": round(pred_time_base, 3),
        "status": "Evaluated",
    })

    # Model 2: HistGradientBoosting (Sklearn)
    # Preprocess categorical features for HistGBM
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_hgb = X_train.copy()
    X_test_hgb = X_test.copy()
    X_train_hgb[CATEGORICAL_COLUMNS] = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test_hgb[CATEGORICAL_COLUMNS] = encoder.transform(X_test[CATEGORICAL_COLUMNS])

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

    t0 = time.time()
    hgb_clf = HistGradientBoostingClassifier(categorical_features=cat_indices, random_state=42)
    hgb_clf.fit(X_train_hgb, y_train_clf)
    hgb_reg = HistGradientBoostingRegressor(categorical_features=cat_indices, random_state=42)
    hgb_reg.fit(X_train_hgb, y_train_reg)
    fit_time_hgb = time.time() - t0

    t0 = time.time()
    pred_risk_hgb = hgb_clf.predict_proba(X_test_hgb)[:, 1]
    pred_delay_hgb = hgb_reg.predict(X_test_hgb)
    pred_time_hgb = time.time() - t0

    auc_hgb = float(roc_auc_score(y_test_clf, pred_risk_hgb))
    mae_hgb = float(mean_absolute_error(y_test_reg, pred_delay_hgb))

    results_table.append({
        "model": "HistGradientBoosting (Sklearn)",
        "classifier_auc": round(auc_hgb, 4),
        "regressor_mae": round(mae_hgb, 2),
        "fit_time_sec": round(fit_time_hgb, 3),
        "predict_time_sec": round(pred_time_hgb, 3),
        "status": "Evaluated",
    })

    # Model 3: TabPFN-3.5
    tabpfn_token = os.environ.get("TABPFN_TOKEN")
    if tabpfn_token:
        try:
            device = os.environ.get("DEVICE", "cpu")
            t0 = time.time()
            tab_clf = LatePaymentClassifier(device=device)
            tab_clf.fit(X_train, y_train_clf)
            tab_reg = PaymentDelayRegressor(device=device)
            tab_reg.fit(X_train, y_train_reg)
            fit_time_tab = time.time() - t0

            t0 = time.time()
            pred_risk_tab = tab_clf.predict_risk(X_test)
            pred_delay_tab = tab_reg.predict(X_test)
            pred_time_tab = time.time() - t0

            auc_tab = float(roc_auc_score(y_test_clf, pred_risk_tab))
            mae_tab = float(mean_absolute_error(y_test_reg, pred_delay_tab))

            results_table.append({
                "model": "TabPFN-3.5 (Local)",
                "classifier_auc": round(auc_tab, 4),
                "regressor_mae": round(mae_tab, 2),
                "fit_time_sec": round(fit_time_tab, 3),
                "predict_time_sec": round(pred_time_tab, 3),
                "status": "Evaluated",
            })
        except Exception as e:
            results_table.append({
                "model": "TabPFN-3.5 (Local)",
                "classifier_auc": None,
                "regressor_mae": None,
                "fit_time_sec": None,
                "predict_time_sec": None,
                "status": f"Runtime Error: {e}",
            })
    else:
        results_table.append({
            "model": "TabPFN-3.5 (Local)",
            "classifier_auc": None,
            "regressor_mae": None,
            "fit_time_sec": None,
            "predict_time_sec": None,
            "status": "Requires TABPFN_TOKEN env var",
        })

    # Format Output JSON
    benchmark_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Synthetic Invoice Dataset",
        "synthetic": True,
        "split": {
            "strategy": "Time-based chronological (70/30)",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_date_range": [train_df["issued_on"].min(), train_df["issued_on"].max()],
            "test_date_range": [test_df["issued_on"].min(), test_df["issued_on"].max()],
        },
        "results": results_table,
    }

    # Save results/benchmark.json
    json_path = RESULTS_DIR / "benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Generate results/benchmark.md
    md_content = f"""# PayRadar Model Benchmark 📊

> [!NOTE]
> **SYNTHETIC DATA NOTICE**: All benchmarks are calculated on 100% synthetically generated invoice records.
> No real customer or transaction data was used.

## Evaluation Setup
- **Split Strategy**: Chronological Time-Based Split (Oldest 70% Train, Newest 30% Test)
- **Train Set**: {len(train_df)} invoices ({train_df['issued_on'].min()} to {train_df['issued_on'].max()})
- **Test Set**: {len(test_df)} invoices ({test_df['issued_on'].min()} to {test_df['issued_on'].max()})
- **Classification Task**: $P(\\text{{Paid}} > 7\\text{{ days late}})$ (Metric: ROC-AUC)
- **Regression Task**: Continuous delay in days (Metric: Mean Absolute Error)

## Results Table

| Model | Classification ROC-AUC | Delay MAE (Days) | Fit Time (s) | Predict Time (s) | Status |
|---|---|---|---|---|---|
"""
    for row in results_table:
        auc_str = f"{row['classifier_auc']:.4f}" if row['classifier_auc'] is not None else "N/A"
        mae_str = f"{row['regressor_mae']:.2f}" if row['regressor_mae'] is not None else "N/A"
        fit_str = f"{row['fit_time_sec']:.3f}s" if row['fit_time_sec'] is not None else "N/A"
        pred_str = f"{row['predict_time_sec']:.3f}s" if row['predict_time_sec'] is not None else "N/A"
        md_content += f"| **{row['model']}** | {auc_str} | {mae_str} | {fit_str} | {pred_str} | {row['status']} |\n"

    md_path = RESULTS_DIR / "benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print summary table to terminal
    print("\n" + "=" * 95)
    print("PAYRADAR MODEL BENCHMARK (100% SYNTHETIC DATA)")
    print("=" * 95)
    print(f"Split: Train={len(train_df)} rows ({train_df['issued_on'].min()} -> {train_df['issued_on'].max()})")
    print(f"       Test ={len(test_df)} rows ({test_df['issued_on'].min()} -> {test_df['issued_on'].max()})")
    print("-" * 95)
    print(f"{'Model':<32} | {'ROC-AUC':<9} | {'MAE (Days)':<10} | {'Fit (s)':<8} | {'Pred (s)':<8} | {'Status'}")
    print("-" * 95)
    for row in results_table:
        auc_str = f"{row['classifier_auc']:.4f}" if row['classifier_auc'] is not None else "N/A"
        mae_str = f"{row['regressor_mae']:.2f}" if row['regressor_mae'] is not None else "N/A"
        fit_str = f"{row['fit_time_sec']:.3f}s" if row['fit_time_sec'] is not None else "N/A"
        pred_str = f"{row['predict_time_sec']:.3f}s" if row['predict_time_sec'] is not None else "N/A"
        print(f"{row['model']:<32} | {auc_str:<9} | {mae_str:<10} | {fit_str:<8} | {pred_str:<8} | {row['status']}")
    print("=" * 95)
    print(f"[OK] Written to {json_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}\n")

    return benchmark_data


if __name__ == "__main__":
    run_benchmark()
