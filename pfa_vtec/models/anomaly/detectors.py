"""Wrappers around scikit-learn anomaly detectors with a uniform interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ...io_paths import CFG


class IForestDetector:
    name = "iforest"

    def __init__(self):
        cfg = CFG["anomaly"]
        self.model = IsolationForest(
            n_estimators=cfg["iforest"]["n_estimators"],
            max_samples=cfg["iforest"]["max_samples"],
            contamination=cfg["contamination"],
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return a positive anomaly score (higher = more anomalous)."""
        return pd.Series(-self.model.score_samples(X), index=X.index, name=self.name)


class OCSVMDetector:
    name = "ocsvm"

    def __init__(self):
        cfg = CFG["anomaly"]
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=cfg["ocsvm"]["nu"])
        self.sub = cfg["ocsvm"]["subsample"]

    def fit(self, X: pd.DataFrame):
        # Subsample for tractability (OCSVM is O(n^2))
        if len(X) > self.sub:
            idx = np.random.default_rng(42).choice(len(X), size=self.sub, replace=False)
            X_sub = X.iloc[idx]
        else:
            X_sub = X
        Xs = self.scaler.fit_transform(X_sub)
        self.model.fit(Xs)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        Xs = self.scaler.transform(X)
        return pd.Series(-self.model.score_samples(Xs), index=X.index, name=self.name)
