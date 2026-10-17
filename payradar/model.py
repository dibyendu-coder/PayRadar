"""TabPFN late-payment risk classifier and payment delay regressor.

Non-negotiables:
- Uses TabPFNClassifier to predict P(paid > 7 days late).
- Uses TabPFNRegressor to predict continuous days_late.
- Device defaults to "cpu" (override with DEVICE env var) to preserve 4 GB GPU VRAM for Ollama.
- Reads TABPFN_TOKEN from environment; never hardcodes tokens.
- Caches fitted models per run.
- Fails with a clear, actionable message if the token is missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor

DEFAULT_DEVICE = os.environ.get("DEVICE", "cpu")
CACHE_DIR = Path(".cache/models")


class MissingTabPFNTokenError(RuntimeError):
    """Raised when TABPFN_TOKEN is not configured in the environment."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message
            or (
                "TabPFN API token not found! TabPFN requires an API key to access model weights.\n"
                "Please obtain a token from Prior Labs (https://priorlabs.ai) and set it in your environment:\n"
                "  PowerShell: $env:TABPFN_TOKEN = '<your-key>'\n"
                "  Bash: export TABPFN_TOKEN='<your-key>'"
            )
        )


def verify_tabpfn_token() -> str:
    """Ensure TABPFN_TOKEN is present in the environment."""
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token:
        raise MissingTabPFNTokenError()
    return token


# In-memory session cache for fitted models
_MODEL_CACHE: dict[str, Any] = {}


class LatePaymentClassifier:
    """TabPFNClassifier predicting P(invoice paid > 7 days late)."""

    def __init__(self, device: str = DEFAULT_DEVICE, **kwargs: Any):
        self.device = os.environ.get("DEVICE", device)
        self.kwargs = kwargs
        self.model: TabPFNClassifier | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LatePaymentClassifier:
        verify_tabpfn_token()
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_in = X.copy()
        else:
            X_in = X

        # TabPFN expects pandas DataFrame or numpy array
        self.model = TabPFNClassifier(device=self.device, **self.kwargs)
        self.model.fit(X_in, np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        verify_tabpfn_token()
        proba = self.model.predict_proba(X)
        return proba

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 1D array of late-payment probabilities P(late > 7 days)."""
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        elif proba.shape[1] == 1:
            # Single class edge case
            return np.ones(len(X)) if self.model.classes_[0] == 1 else np.zeros(len(X))
        return proba[:, -1]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        risk = self.predict_risk(X)
        return (risk > 0.5).astype(int)


class PaymentDelayRegressor:
    """TabPFNRegressor predicting expected days_late for an invoice."""

    def __init__(self, device: str = DEFAULT_DEVICE, **kwargs: Any):
        self.device = os.environ.get("DEVICE", device)
        self.kwargs = kwargs
        self.model: TabPFNRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> PaymentDelayRegressor:
        verify_tabpfn_token()
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_in = X.copy()
        else:
            X_in = X

        self.model = TabPFNRegressor(device=self.device, **self.kwargs)
        self.model.fit(X_in, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        verify_tabpfn_token()
        preds = self.model.predict(X)
        return np.asarray(preds)


def get_fitted_models(
    X_train: pd.DataFrame,
    y_clf: pd.Series | np.ndarray,
    y_reg: pd.Series | np.ndarray,
    cache_key: str = "default",
    device: str = DEFAULT_DEVICE,
) -> tuple[LatePaymentClassifier, PaymentDelayRegressor]:
    """Retrieve or fit cached TabPFN classifier and regressor for a training dataset."""
    clf_key = f"{cache_key}_clf"
    reg_key = f"{cache_key}_reg"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clf_disk = CACHE_DIR / f"{clf_key}.joblib"
    reg_disk = CACHE_DIR / f"{reg_key}.joblib"

    # Check memory cache
    if clf_key in _MODEL_CACHE and reg_key in _MODEL_CACHE:
        return _MODEL_CACHE[clf_key], _MODEL_CACHE[reg_key]

    # Check disk cache
    if clf_disk.exists() and reg_disk.exists():
        try:
            clf = joblib.load(clf_disk)
            reg = joblib.load(reg_disk)
            _MODEL_CACHE[clf_key] = clf
            _MODEL_CACHE[reg_key] = reg
            return clf, reg
        except Exception:
            pass

    # Fit fresh models
    clf = LatePaymentClassifier(device=device)
    clf.fit(X_train, y_clf)

    reg = PaymentDelayRegressor(device=device)
    reg.fit(X_train, y_reg)

    _MODEL_CACHE[clf_key] = clf
    _MODEL_CACHE[reg_key] = reg

    try:
        joblib.dump(clf, clf_disk)
        joblib.dump(reg, reg_disk)
    except Exception:
        pass

    return clf, reg
