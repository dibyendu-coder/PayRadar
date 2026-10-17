"""Unit tests for time-based dataset splitting and benchmark baselines."""

import numpy as np
import pandas as pd

from payradar.bench import PerClientMeanBaseline, get_time_based_split


def test_time_split_never_leaks_future_invoices():
    """Verify that every test invoice is issued at or after every training invoice."""
    dates = [
        "2024-01-05",
        "2024-02-10",
        "2024-03-15",
        "2024-04-20",
        "2024-05-25",
        "2024-06-30",
        "2024-07-05",
        "2024-08-10",
        "2024-09-15",
        "2024-10-20",
    ]
    df = pd.DataFrame(
        {
            "id": [f"INV-{i:02d}" for i in range(1, 11)],
            "issued_on": dates,
            "amount": [1000.0 * i for i in range(1, 11)],
        }
    )

    train_df, test_df = get_time_based_split(df, train_ratio=0.70)

    assert len(train_df) == 7
    assert len(test_df) == 3

    max_train_date = train_df["issued_on"].max()
    min_test_date = test_df["issued_on"].min()

    assert max_train_date <= min_test_date
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))


def test_per_client_baseline_handles_known_and_unseen():
    """Verify PerClientMeanBaseline produces correct client means and handles new clients."""
    X_train = pd.DataFrame({"client": ["Acme", "Acme", "Beta", "Beta"]})
    y_clf = np.array([1, 1, 0, 0])
    y_reg = np.array([20.0, 10.0, 2.0, 4.0])

    baseline = PerClientMeanBaseline().fit(X_train, y_clf, y_reg)

    X_test = pd.DataFrame({"client": ["Acme", "Beta", "UnknownCorp"]})
    pred_risk = baseline.predict_risk(X_test)
    pred_delay = baseline.predict_delay(X_test)

    # Acme risk = 1.0, Beta = 0.0, UnknownCorp = global mean (0.5)
    assert pred_risk[0] == 1.0
    assert pred_risk[1] == 0.0
    assert pred_risk[2] == 0.5

    # Acme delay = 15.0, Beta = 3.0, UnknownCorp = global mean (9.0)
    assert pred_delay[0] == 15.0
    assert pred_delay[1] == 3.0
    assert pred_delay[2] == 9.0


def test_expanding_window_splits_no_leakage():
    """Verify 3 expanding-window splits expand training and never leak future invoices."""
    from payradar.bench import get_expanding_window_splits

    n_samples = 100
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame(
        {
            "id": [f"INV-{i:03d}" for i in range(n_samples)],
            "issued_on": dates,
            "amount": [100.0 * (i + 1) for i in range(n_samples)],
        }
    )

    splits = get_expanding_window_splits(df)
    assert len(splits) == 3

    prev_train_len = 0
    for idx, (train_df, test_df) in enumerate(splits, start=1):
        # 1. Training window expands chronologically
        assert len(train_df) > prev_train_len
        prev_train_len = len(train_df)

        # 2. Strict chronological order (no future leakage)
        max_train_date = train_df["issued_on"].max()
        min_test_date = test_df["issued_on"].min()
        assert max_train_date <= min_test_date

        # 3. Disjoint IDs
        assert set(train_df["id"]).isdisjoint(set(test_df["id"]))


def test_bootstrap_auc_diff_ci():
    """Verify bootstrap CI generates valid 95% interval on difference."""
    from payradar.bench import compute_bootstrap_auc_diff_ci

    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5)
    # Model 1 has high separation (AUC ~ 1.0)
    y_pred_m1 = np.array([0.1, 0.2, 0.15, 0.1, 0.25, 0.85, 0.9, 0.8, 0.75, 0.95] * 5)
    # Model 2 has poor/inverted predictions (AUC ~ 0.5 or lower)
    y_pred_m2 = np.array([0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.3, 0.9, 0.1, 0.4] * 5)

    ci_res = compute_bootstrap_auc_diff_ci(y_true, y_pred_m1, y_pred_m2, n_bootstraps=100, seed=42)
    assert ci_res["bootstraps_run"] > 0
    assert ci_res["ci_lower_95"] <= ci_res["mean_diff"] <= ci_res["ci_upper_95"]
    assert ci_res["mean_diff"] > 0
