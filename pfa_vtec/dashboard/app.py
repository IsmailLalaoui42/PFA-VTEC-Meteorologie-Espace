"""Dash + Plotly editorial dashboard - entry point.

Run with ``python -m pfa_vtec dashboard``.
"""
from __future__ import annotations

import dash
from dash import Dash

from ..data import load_merged, load_storms
from ..viz.theme import register_plotly_template
from .callbacks import register
from .layout import build_layout

register_plotly_template(default=True)


def build_app() -> Dash:
    df = load_merged()
    storms = load_storms()
    # La periode affichee est la fenetre VTEC reellement exploitable (oct. 2015 -> sep. 2025),
    # et non la grille OMNI complete 2015-01 -> 2025-12.
    df_vtec = df.dropna(subset=["vtec_mean"])
    summary = dict(
        period_short=f"{df_vtec.index.min().strftime('%Y-%m')} -> {df_vtec.index.max().strftime('%Y-%m')}",
        n_rows=len(df),
        vtec_cov=df["vtec_mean"].notna().mean() * 100,
        n_storms=len(storms),
        vtec_max=float(df["vtec_mean"].max(skipna=True)),
    )

    app = dash.Dash(
        __name__,
        title="PFA VTEC and space weather",
        update_title=None,
        assets_folder="assets",
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.layout = build_layout(summary)
    register(app)
    return app


if __name__ == "__main__":
    build_app().run(host="127.0.0.1", port=8050, debug=False)
