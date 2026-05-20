"""Orchestrate anomaly detection across detectors; write anomaly_comparison.csv."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console

from ...io_paths import CFG, MODELS_DIR, REPORTS_DIR
from .detectors import IForestDetector, OCSVMDetector
from .evaluate import collapse_anomalies, match_against_storms
from .features import build_anomaly_features

console = Console()

REGISTRY = {
    "iforest": IForestDetector,
    "ocsvm":   OCSVMDetector,
}


def run_detection(method: str = "all") -> Path:
    enabled = CFG["anomaly"]["enabled_methods"]
    enabled = [m for m in enabled if m in REGISTRY]
    if method != "all":
        enabled = [method]
        if method not in REGISTRY:
            console.print(f"[red]method {method!r} not implemented[/red]")
            return REPORTS_DIR / "anomaly_comparison.csv"

    X = build_anomaly_features()
    # Train/eval split: anomaly detection is unsupervised, but we want to train on a
    # 'quiet' period and score on the full dataset to expose events catalog-wide.
    train_end = pd.Timestamp(CFG["split"]["train_end"], tz="UTC")
    X_train = X.loc[X.index <= train_end]

    rows: list[dict] = []
    for name in enabled:
        det = REGISTRY[name]()
        det.fit(X_train)
        scores = det.score(X)
        # threshold at 99th percentile of training scores -> stable
        thresh = float(scores.loc[X_train.index].quantile(0.99))
        events = collapse_anomalies(scores, threshold=thresh)
        res = match_against_storms(events)
        rows.append({
            "method": name, "threshold": thresh,
            "precision": res.precision, "recall": res.recall, "f1": res.f1,
            "n_detected_events": res.n_detected_events,
            "n_catalog_events":  res.n_catalog_events,
            "n_matched":         res.n_matched,
        })
        console.print(
            f"  [cyan]{name:>8s}[/cyan]  evt={res.n_detected_events:5d}  "
            f"P={res.precision:.3f}  R={res.recall:.3f}  F1={res.f1:.3f}"
        )
        # Persist
        joblib.dump(det, MODELS_DIR / f"anom_{name}.joblib")
        scores.to_csv(REPORTS_DIR / f"anom_{name}_scores.csv", header=True)
        events.to_csv(REPORTS_DIR / f"anom_{name}_events.csv", index=False)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out_path = REPORTS_DIR / "anomaly_comparison.csv"
    df.to_csv(out_path, index=False)
    console.print(f"[green]wrote {out_path}[/green]")
    return out_path
