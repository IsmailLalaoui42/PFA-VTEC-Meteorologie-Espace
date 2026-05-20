"""Event-level evaluation: collapse anomalies, match storm catalog with tolerance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...data import load_storms
from ...io_paths import CFG


@dataclass
class EventScore:
    precision: float
    recall: float
    f1: float
    n_detected_events: int
    n_catalog_events: int
    n_matched: int


def collapse_anomalies(scores: pd.Series, threshold: float,
                       gap_hours: int | None = None) -> pd.DataFrame:
    """Collapse consecutive point-anomalies into events.

    Returns a DataFrame [start, end, peak_score]. Two consecutive
    anomalies separated by less than `gap_hours` belong to the same event.
    """
    gap_hours = gap_hours if gap_hours is not None else CFG["anomaly"]["event_gap_hours"]
    above = scores > threshold
    if not above.any():
        return pd.DataFrame(columns=["start", "end", "peak_score"])

    # Make sure scores has a 1h freq for gap math
    idx = scores.index
    # Build a Series of contiguous group ids
    above_int = above.astype(int).to_numpy()
    # The "session" id increments when the time since the previous True > gap
    # OR when the current sample is True after a non-True one.
    sess = np.zeros(len(above_int), dtype=int)
    cur = 0
    last_true_pos = -1
    for i, b in enumerate(above_int):
        if b == 1:
            if last_true_pos == -1 or (i - last_true_pos) > gap_hours:
                cur += 1
            sess[i] = cur
            last_true_pos = i
    events = []
    for sid in range(1, cur + 1):
        mask = sess == sid
        ts = idx[mask]
        if len(ts) == 0:
            continue
        peak = float(scores[mask].max())
        events.append({"start": ts[0], "end": ts[-1], "peak_score": peak})
    return pd.DataFrame(events)


def match_against_storms(events: pd.DataFrame, tolerance_hours: int | None = None) -> EventScore:
    tol = tolerance_hours if tolerance_hours is not None else CFG["anomaly"]["match_tolerance_hours"]
    storms = load_storms()
    n_cat = len(storms)
    n_det = len(events)

    if n_det == 0:
        return EventScore(precision=float("nan"), recall=0.0, f1=0.0,
                          n_detected_events=0, n_catalog_events=n_cat, n_matched=0)

    storm_starts = storms["start"].to_numpy()
    storm_ends = storms["end"].to_numpy()
    tol_td = np.timedelta64(tol, "h")

    detected_matched = np.zeros(n_det, dtype=bool)
    catalog_matched = np.zeros(n_cat, dtype=bool)

    det_starts = events["start"].to_numpy()
    det_ends = events["end"].to_numpy()
    for i in range(n_det):
        ds, de = det_starts[i], det_ends[i]
        # Overlap with tolerance: det overlaps storm if (de + tol >= ss) and (ds - tol <= se)
        overlap = (de + tol_td >= storm_starts) & (ds - tol_td <= storm_ends)
        if overlap.any():
            detected_matched[i] = True
            for j in np.where(overlap)[0]:
                catalog_matched[j] = True

    n_matched_det = int(detected_matched.sum())
    n_matched_cat = int(catalog_matched.sum())
    precision = n_matched_det / n_det if n_det else float("nan")
    recall    = n_matched_cat / n_cat if n_cat else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return EventScore(
        precision=precision, recall=recall, f1=f1,
        n_detected_events=n_det, n_catalog_events=n_cat, n_matched=n_matched_det,
    )
