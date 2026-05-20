"""Loaders for the CSV outputs of the Phase 1 pipeline.

The Phase 1 monolithic script `01_data_engineering_PFA.py` already produced
clean CSVs in `output/`. These loaders consume them, parse the time index
as UTC tz-aware, and return DataFrames ready for feature engineering.
"""
from __future__ import annotations

import pandas as pd

from ..io_paths import MERGED_CSV, OMNI_CSV, VTEC_CSV, STORMS_CSV


def _read_time_indexed(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_merged() -> pd.DataFrame:
    """Merged OMNI 30-min + VTEC 30-min for Beni Mellal.

    Columns include: Bz_*, Vsw_*, Pdyn_*, Np_*, Bs, epsilon_coupling,
    vtec_mean, vtec_std, vtec_min, vtec_max, n_maps, coverage_pct.
    Index = UTC tz-aware at 30-min cadence.
    """
    return _read_time_indexed(MERGED_CSV)


def load_omni() -> pd.DataFrame:
    """Cleaned OMNI 30-min only (no VTEC)."""
    return _read_time_indexed(OMNI_CSV)


def load_vtec() -> pd.DataFrame:
    """VTEC GIM 30-min only (Beni Mellal point)."""
    return _read_time_indexed(VTEC_CSV)


def load_storms() -> pd.DataFrame:
    """Storm catalog (Bz < -10 nT events): start, end, Bz_min, duration_h.

    Times are UTC-aware. Returned as a plain DataFrame (no time index — the
    rows are events, not regular samples).
    """
    df = pd.read_csv(STORMS_CSV)
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.sort_values("start").reset_index(drop=True)
    return df
