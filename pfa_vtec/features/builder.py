"""Single feature factory.

Build the design matrix X and the per-horizon targets y[h] from the merged
30-min dataset. Guarantees no temporal leakage:
    - rolling stats use ``closed="left"``
    - lags are strictly past (shift by +k)
    - cyclical encoding of (hour, doy, month) is computed on the timestamp itself

The factory is deterministic and cached as parquet in reports/features_cache.parquet.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..data import load_merged
from ..io_paths import CFG, REPORTS_DIR

CACHE_PATH = REPORTS_DIR / "features_cache.parquet"
HORIZON_META_PATH = REPORTS_DIR / "features_meta.json"


def _add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for k in lags:
            out[f"{c}__lag{k}"] = out[c].shift(k)
    return out


def _add_rolling_features(df: pd.DataFrame, cols: list[str], windows_hours: list[int]) -> pd.DataFrame:
    """Rolling mean / std with closed="left" (no leakage). df at 1h cadence."""
    out = df.copy()
    for h in windows_hours:
        steps = h  # 1h cadence
        for c in cols:
            roll = out[c].shift(1).rolling(window=steps, min_periods=max(2, steps // 2))
            out[f"{c}__r{h}h_mean"] = roll.mean()
            out[f"{c}__r{h}h_std"]  = roll.std()
    return out


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out.index
    hour = t.hour + t.minute / 60.0
    doy = t.dayofyear
    month = t.month
    out["sin_hour"]  = np.sin(2 * np.pi * hour / 24.0)
    out["cos_hour"]  = np.cos(2 * np.pi * hour / 24.0)
    out["sin_doy"]   = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"]   = np.cos(2 * np.pi * doy / 365.25)
    out["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    return out


def build_features(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[int, pd.Series], list[str]]:
    """Build (X, y_by_horizon, feature_names) from the merged dataframe.

    The merged dataset is at 30-min cadence; VTEC is hourly natively. We resample
    to 1h (OMNI mean) so that horizons map cleanly to whole-hour offsets.

    - Only rows where vtec_mean is not NaN are kept as candidates for training.
    - Rows where any base feature is NaN are dropped.
    - Future-horizon NaNs in y[h] are preserved (caller drops them).
    """
    if df is None:
        df = load_merged()
    # Resample to 1h: VTEC keeps its native cadence, OMNI gets a 2-row mean per hour.
    df = df.resample("1h").mean()
    target = "vtec_mean"
    horizons: list[int] = list(CFG["horizons"])
    drivers: list[str]  = list(CFG["features"]["drivers"])
    lags_target  = list(CFG["features"]["lags_target"])
    lags_drivers = list(CFG["features"]["lags_drivers"])
    rolling_hrs  = list(CFG["features"]["rolling_hours"])

    base = df[[target] + drivers].copy()
    feat = _add_lag_features(base, [target], lags_target)
    feat = _add_lag_features(feat, drivers, lags_drivers)
    feat = _add_rolling_features(feat, [target] + drivers, rolling_hrs)
    feat = _add_cyclical_features(feat)

    # Remove the *current* target from feature columns to avoid trivial leakage
    feat = feat.drop(columns=[target] + drivers)

    # Targets per horizon: y[h](t) = vtec_mean(t + h)
    y_by_h = {h: df[target].shift(-h) for h in horizons}

    # Keep rows where the closest VTEC lag exists (minimum signal for forecasting)
    # and where the 3-hour driver rolling stats exist (proxy for OMNI being present)
    must_have = ["vtec_mean__lag1", "Bz_mean__r3h_mean"]
    must_have = [c for c in must_have if c in feat.columns]
    valid = feat.dropna(subset=must_have).index
    X = feat.loc[valid].copy()
    # Median-impute remaining NaNs on the training feature matrix to make it RF-ready;
    # HGB handles NaN natively but RF does not.
    X = X.fillna(X.median(numeric_only=True))
    y_by_h = {h: y_by_h[h].loc[valid] for h in horizons}

    feat_names = list(X.columns)
    return X, y_by_h, feat_names


def build_and_cache() -> Path:
    """Build features and persist a parquet cache for reuse by trainers."""
    X, y_by_h, names = build_features()
    out = X.copy()
    for h, y in y_by_h.items():
        out[f"y_h{h}"] = y
    out.to_parquet(CACHE_PATH)
    import json
    meta = {"n_features": len(names), "feature_names": names,
            "horizons": list(y_by_h.keys()), "n_rows": int(len(out))}
    HORIZON_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return CACHE_PATH


def load_cached_features() -> tuple[pd.DataFrame, dict[int, pd.Series], list[str]]:
    if not CACHE_PATH.exists():
        build_and_cache()
    df = pd.read_parquet(CACHE_PATH)
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    horizons = sorted([int(c[3:]) for c in y_cols])
    X = df.drop(columns=y_cols)
    y_by_h = {h: df[f"y_h{h}"] for h in horizons}
    return X, y_by_h, list(X.columns)
