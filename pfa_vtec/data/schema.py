"""Frozen schema of the merged dataset - used to assert invariants in tests."""
from __future__ import annotations

MERGED_COLUMNS: tuple[str, ...] = (
    "Bz_mean", "Bz_std", "Bz_min", "Bz_max",
    "Vsw_mean", "Vsw_std",
    "Pdyn_mean", "Pdyn_std",
    "Np_mean", "Np_std",
    "n_valid", "coverage_pct",
    "Bs", "epsilon_coupling",
    "vtec_mean", "vtec_std", "vtec_min", "vtec_max", "n_maps",
)

DRIVERS = ("Bz_mean", "Vsw_mean", "Pdyn_mean", "Np_mean", "epsilon_coupling", "Bs")
TARGET = "vtec_mean"
FREQ = "30min"
TIMEZONE = "UTC"


def assert_schema(df) -> None:
    missing = [c for c in MERGED_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f"merged dataset is missing columns: {missing}")
    if str(df.index.tz) not in ("UTC", "tzutc()", "datetime.timezone.utc"):
        raise AssertionError(f"index timezone must be UTC, got {df.index.tz!r}")
