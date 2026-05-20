"""Forecaster registry - one place to pick a model by name."""
from __future__ import annotations

from .baselines import ClimatologyForecaster, PersistenceForecaster
from .classical import HGBForecaster, RFForecaster

REGISTRY = {
    "persistence": PersistenceForecaster,
    "climatology": ClimatologyForecaster,
    "rf":          RFForecaster,
    "hgb":         HGBForecaster,
    # LSTM / 1D-CNN added in D4
}


def make(name: str, horizon: int):
    if name not in REGISTRY:
        raise KeyError(f"unknown model: {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name](horizon=horizon)
