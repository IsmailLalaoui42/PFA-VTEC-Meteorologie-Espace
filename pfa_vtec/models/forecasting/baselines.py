"""Baselines: persistence + climatology + (optional) SARIMA placeholder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """y_hat(t+h) = y(t).

    With feature lag layout, "y(t)" is encoded in feature 'vtec_mean__lag<h>'.
    Predicting horizon h from features at time t means returning the lag h value.
    To stay general, we expect a designated input column 'vtec_mean__lag1' OR
    if horizon-specific, the caller picks the right lag.
    Here we just return the most recent target lag available in X.
    """
    name = "persistence"

    def __init__(self, horizon: int):
        self.horizon = horizon
        self._col = "vtec_mean__lag1"  # closest known past observation

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._col not in X.columns:
            raise ValueError(f"feature '{self._col}' missing")
        return X[self._col]


class ClimatologyForecaster(BaseForecaster):
    """Median by (month, hour) computed on train."""
    name = "climatology"

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.lut: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        t = X.index + pd.Timedelta(minutes=30 * self.horizon)
        key = pd.DataFrame({"month": t.month, "hour": t.hour}, index=X.index)
        df = key.assign(y=y.values)
        self.lut = df.groupby(["month", "hour"])["y"].median()
        # Fallback when a (month, hour) bin has all-NaN
        self._fallback = float(y.median(skipna=True))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.lut is None:
            raise RuntimeError("not fitted")
        t = X.index + pd.Timedelta(minutes=30 * self.horizon)
        key = pd.MultiIndex.from_arrays([t.month, t.hour], names=["month", "hour"])
        out = self.lut.reindex(key)
        out = out.fillna(self._fallback)
        out.index = X.index
        return out
