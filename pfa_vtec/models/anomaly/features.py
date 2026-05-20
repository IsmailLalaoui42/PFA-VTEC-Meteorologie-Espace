"""Anomaly-specific feature engineering.

The forecasting features (lags of vtec/drivers) are not ideal for anomaly
detection. Here we build a smaller set of *instant* descriptors capturing
the geomagnetic disturbance signature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data import load_merged


def build_anomaly_features() -> pd.DataFrame:
    """Build features adapted for anomaly detection on the 1h merged data.

    Columns:
      - Bz_mean, Bz_min        : instantaneous, with VTEC drivers
      - Vsw_mean, Pdyn_mean
      - epsilon_coupling, Bs
      - dBz_3h, dVsw_3h        : 3-h temporal differences
      - vtec_residual          : vtec_mean - climatology(month, hour)
    Rows with any NaN are dropped.
    """
    df = load_merged().resample("1h").mean()
    base = df[["Bz_mean", "Bz_min", "Vsw_mean", "Pdyn_mean",
               "epsilon_coupling", "Bs", "vtec_mean"]].copy()
    base["dBz_3h"]  = base["Bz_mean"].diff(3)
    base["dVsw_3h"] = base["Vsw_mean"].diff(3)
    # climatology median by (month, hour) on full available history
    key = pd.DataFrame({"m": base.index.month, "h": base.index.hour}, index=base.index)
    clim = base.groupby([key["m"], key["h"]])["vtec_mean"].transform("median")
    base["vtec_residual"] = base["vtec_mean"] - clim
    base = base.drop(columns=["vtec_mean"])
    return base.dropna()
