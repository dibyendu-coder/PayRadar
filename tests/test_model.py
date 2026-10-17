"""Unit tests for TabPFN model wrappers and token handling."""

import os
import numpy as np
import pandas as pd
import pytest
from payradar.model import (
    LatePaymentClassifier,
    MissingTabPFNTokenError,
    PaymentDelayRegressor,
    verify_tabpfn_token,
)


def test_missing_token_raises_clear_error(monkeypatch):
    """Ensure missing TABPFN_TOKEN raises an informative MissingTabPFNTokenError."""
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(MissingTabPFNTokenError) as exc_info:
        verify_tabpfn_token()
    assert "TABPFN_TOKEN" in str(exc_info.value)

    clf = LatePaymentClassifier()
    with pytest.raises(MissingTabPFNTokenError):
        clf.fit(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]))


def test_device_defaults_to_cpu(monkeypatch):
    """Ensure default device is CPU to preserve 4 GB GPU for Ollama."""
    monkeypatch.delenv("DEVICE", raising=False)
    clf = LatePaymentClassifier()
    reg = PaymentDelayRegressor()
    assert clf.device == "cpu"
    assert reg.device == "cpu"


@pytest.mark.skipif(
    not os.environ.get("TABPFN_TOKEN"),
    reason="TABPFN_TOKEN is unset in environment; skipping live TabPFN fit test",
)
def test_tabpfn_fit_predict_tiny_fixture():
    """Live test with tiny fixture when TABPFN_TOKEN is provided."""
    X = pd.DataFrame({
        "log_amount": [7.0, 8.5, 6.2, 9.1],
        "terms_days": [30, 30, 14, 45],
        "issue_month": [1, 3, 5, 8],
        "issue_dow": [0, 2, 4, 1],
        "client": ["Acme", "Beta", "Acme", "Beta"],
        "project_type": ["web", "design", "web", "design"],
        "client_prev_count": [0, 0, 1, 1],
        "client_prev_mean_days_late": [0.0, 0.0, 15.0, 2.0],
        "amount_vs_client_median": [0.0, 0.0, -100.0, 500.0],
    })
    y_clf = np.array([0, 1, 0, 1])
    y_reg = np.array([0.0, 14.0, 2.0, 21.0])

    clf = LatePaymentClassifier(device="cpu")
    clf.fit(X, y_clf)
    probs = clf.predict_risk(X)
    assert len(probs) == 4
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    reg = PaymentDelayRegressor(device="cpu")
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    assert len(preds) == 4
