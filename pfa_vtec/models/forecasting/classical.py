"""Classical ML forecasters: Random Forest and Histogram Gradient Boosting."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from ..common import BaseForecaster
from ...io_paths import CFG


class RFForecaster(BaseForecaster):
    name = "rf"

    def __init__(self, horizon: int):
        self.horizon = horizon
        cfg = CFG["forecasting"]["rf"]
        self.model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)


class HGBForecaster(BaseForecaster):
    name = "hgb"

    def __init__(self, horizon: int):
        self.horizon = horizon
        cfg = CFG["forecasting"]["hgb"]
        self.model = HistGradientBoostingRegressor(
            max_iter=cfg["max_iter"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)
