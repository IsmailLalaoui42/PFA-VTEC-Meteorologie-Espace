"""Forecast metrics - RMSE, MAE, MAPE, R^2, and skill score vs persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align(y_true, y_pred):
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred, index=y_true.index) if not isinstance(y_pred, pd.Series) else y_pred
    mask = y_true.notna() & y_pred.notna()
    return y_true[mask].to_numpy(), y_pred[mask].to_numpy()


def rmse(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    if a.size == 0: return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    if a.size == 0: return float("nan")
    return float(np.mean(np.abs(a - b)))


def mape(y_true, y_pred, eps: float = 0.1) -> float:
    """MAPE protected against y close to zero by adding eps in the denominator."""
    a, b = _align(y_true, y_pred)
    if a.size == 0: return float("nan")
    return float(np.mean(np.abs((a - b) / (np.abs(a) + eps))) * 100)


def r2(y_true, y_pred) -> float:
    a, b = _align(y_true, y_pred)
    if a.size == 0: return float("nan")
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0: return float("nan")
    return float(1 - ss_res / ss_tot)


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    if rmse_baseline == 0 or np.isnan(rmse_baseline):
        return float("nan")
    return float(1 - rmse_model / rmse_baseline)


def summarize(y_true, y_pred, rmse_baseline: float | None = None) -> dict:
    rm = rmse(y_true, y_pred)
    out = {
        "rmse": rm,
        "mae":  mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }
    if rmse_baseline is not None:
        out["skill"] = skill_score(rm, rmse_baseline)
    return out
