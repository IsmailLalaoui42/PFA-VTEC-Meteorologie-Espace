"""Temporal split + walk-forward cross-validation with embargo."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..io_paths import CFG


def temporal_split(X: pd.DataFrame, y: pd.Series):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test).

    Boundaries come from config: train ends at CFG[split][train_end], val ends at val_end.
    """
    train_end = pd.Timestamp(CFG["split"]["train_end"], tz="UTC")
    val_end   = pd.Timestamp(CFG["split"]["val_end"],   tz="UTC")
    mask_tr  = (X.index <= train_end)
    mask_val = (X.index >  train_end) & (X.index <= val_end)
    mask_te  = (X.index >  val_end)
    return (
        X.loc[mask_tr],  y.loc[mask_tr],
        X.loc[mask_val], y.loc[mask_val],
        X.loc[mask_te],  y.loc[mask_te],
    )


def time_series_cv(n_samples: int, n_splits: int = 5, gap: int | None = None):
    gap = gap if gap is not None else CFG["split"]["embargo_steps"]
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)
