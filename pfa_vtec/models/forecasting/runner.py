"""Orchestrate training across models * horizons; persist comparison CSV."""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import joblib
import pandas as pd
from rich.console import Console

from ...evaluation import temporal_split, summarize
from ...features import load_cached_features
from ...io_paths import CFG, MODELS_DIR, REPORTS_DIR
from . import registry
from ..common import set_seeds

warnings.filterwarnings("ignore")
console = Console()


def run_training(model: str = "all", horizon: int | None = None) -> Path:
    set_seeds(42)
    X, y_by_h, _ = load_cached_features()

    enabled = CFG["forecasting"]["enabled_models"]
    enabled = [m for m in enabled if m in registry.REGISTRY]  # filter D2 subset
    if model != "all":
        enabled = [model]
        if model not in registry.REGISTRY:
            console.print(f"[red]model {model!r} not in D2 registry yet[/red]")
            return REPORTS_DIR / "forecast_comparison.csv"

    horizons = CFG["horizons"] if horizon is None else [horizon]
    rows: list[dict] = []

    for h in horizons:
        y_full = y_by_h[h]
        Xt, yt, Xv, yv, Xs, ys = temporal_split(X, y_full)
        # Drop val/test rows where y is NaN (future horizon may push past the last row)
        Xv = Xv.loc[yv.notna()]; yv = yv.dropna()
        Xs = Xs.loc[ys.notna()]; ys = ys.dropna()
        Xt = Xt.loc[yt.notna()]; yt = yt.dropna()
        console.print(f"[bold]horizon h={h}[/bold]  train={len(yt):,}  val={len(yv):,}  test={len(ys):,}")

        # Persistence baseline (no fit) gives the reference RMSE for skill
        pers = registry.make("persistence", horizon=h)
        pers.fit(Xt, yt)
        rmse_pers = summarize(ys, pers.predict(Xs))["rmse"]

        for name in enabled:
            t0 = time.time()
            mdl = registry.make(name, horizon=h)
            mdl.fit(Xt, yt)
            elapsed = time.time() - t0
            pred = mdl.predict(Xs)
            metrics = summarize(ys, pred, rmse_baseline=rmse_pers)
            rows.append({"model": name, "horizon": h, **metrics, "fit_seconds": round(elapsed, 2)})
            console.print(
                f"  [cyan]{name:>12s}[/cyan]  "
                f"rmse={metrics['rmse']:.3f}  mae={metrics['mae']:.3f}  "
                f"r2={metrics['r2']:.3f}  skill={metrics.get('skill', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

            # Persist non-trivial models
            if name not in ("persistence",):
                mpath = MODELS_DIR / f"{name}_h{h}.joblib"
                joblib.dump(mdl, mpath)
                meta = {"name": name, "horizon": h, **metrics}
                (MODELS_DIR / f"{name}_h{h}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows).sort_values(["horizon", "rmse"])
    out_path = REPORTS_DIR / "forecast_comparison.csv"
    df.to_csv(out_path, index=False)
    console.print(f"[green]wrote {out_path}[/green]")
    return out_path
