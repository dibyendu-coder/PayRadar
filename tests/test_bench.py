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
    df = pd.DataFrame({
        "id": [f"INV-{i:02d}" for i in range(1, 11)],
        "issued_on": dates,
        "amount": [1000.0 * i for i in range(1, 11)],
    })

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
