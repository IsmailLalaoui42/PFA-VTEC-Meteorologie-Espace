"""Dashboard callbacks - tab switching + tab A/B/C/D interactive panels."""
from __future__ import annotations

from dash import Input, Output, State, dcc, html, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ..data import load_merged, load_storms
from ..io_paths import REPORTS_DIR, MODELS_DIR
from ..viz.theme import THEME
from .components import hairline, kpi_tile


def register(app):
    # Tab switch -> mounts the correct tab content
    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(value):
        if value == "tab-a":
            return _tab_a()
        if value == "tab-b":
            return _tab_b()
        if value == "tab-c":
            return _tab_c()
        if value == "tab-d":
            return _tab_d()
        return html.Div("?")

    # Tab A: date range -> redraw timeseries + KPIs
    @app.callback(
        [Output("ts-fig", "figure"), Output("ts-kpis", "children")],
        Input("ts-date-range", "value"),
    )
    def update_tab_a(range_idx):
        df = load_merged()
        df_v = df.dropna(subset=["vtec_mean"])
        n = len(df_v)
        if range_idx is None:
            i0, i1 = max(0, n - 30 * 48), n - 1
        else:
            i0, i1 = int(range_idx[0]), int(range_idx[1])
        i0 = max(0, min(i0, n - 1)); i1 = max(i0 + 1, min(i1, n - 1))
        sub = df_v.iloc[i0:i1 + 1]
        fig = _ts_figure(sub)
        kpis = _ts_kpis(sub)
        return fig, kpis


# ───────────────────────────────────────────────────────────
# Tab A - VTEC EXPLORER
# ───────────────────────────────────────────────────────────
def _tab_a() -> html.Div:
    df = load_merged().dropna(subset=["vtec_mean"])
    n = len(df)
    default_start = max(0, n - 30 * 48)
    default_end = n - 1
    return html.Div([
        dcc.Graph(id="ts-fig", figure=_ts_figure(df.iloc[default_start:default_end + 1]),
                  config={"displayModeBar": False}),
        html.Div(style={"padding": "6px 4px 18px 4px"}, children=[
            html.Div("PERIODE (BRUSH)", className="ui", style={
                "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
                "color": THEME["muted"], "letterSpacing": "0.12em", "marginBottom": "8px",
            }),
            dcc.RangeSlider(
                id="ts-date-range",
                min=0, max=n - 1, step=1,
                value=[default_start, default_end],
                marks=_slider_marks(df),
                tooltip={"placement": "bottom", "always_visible": False},
                allowCross=False,
            ),
        ]),
        html.Div(id="ts-kpis", style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "marginTop": "18px",
        }, children=_ts_kpis(df.iloc[default_start:default_end + 1])),
    ])


def _ts_figure(sub: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["vtec_mean"],
        mode="lines", line=dict(color=THEME["accent"], width=1.4),
        name="VTEC", hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} TECU<extra></extra>",
    ))
    fig.update_layout(
        height=440,
        margin=dict(l=60, r=16, t=10, b=30),
        showlegend=False,
        xaxis=dict(title=None),
        yaxis=dict(title="VTEC  -  TECU"),
    )
    return fig


def _ts_kpis(sub: pd.DataFrame) -> list:
    v = sub["vtec_mean"]
    return [
        kpi_tile("VTEC MEAN",   f"{v.mean():.1f}", "TECU"),
        kpi_tile("VTEC MIN",    f"{v.min():.1f}",  "TECU"),
        kpi_tile("VTEC MAX",    f"{v.max():.1f}",  "TECU"),
        kpi_tile("WINDOW SPAN", f"{(sub.index[-1] - sub.index[0]).days}", "d"),
    ]


def _slider_marks(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {}
    step = max(1, n // 8)
    marks = {}
    for i in range(0, n, step):
        marks[i] = {"label": df.index[i].strftime("%Y-%m"), "style": {
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "10px",
            "color": THEME["muted"],
        }}
    marks[n - 1] = {"label": df.index[-1].strftime("%Y-%m"), "style": {
        "fontFamily": "JetBrains Mono, monospace", "fontSize": "10px", "color": THEME["muted"],
    }}
    return marks


# ───────────────────────────────────────────────────────────
# Tab B, C, D - placeholders for D5/D6 (build skeleton only)
# ───────────────────────────────────────────────────────────
def _tab_b() -> html.Div:
    df = load_merged().resample("1h").mean()
    storms = load_storms()
    # restrict to 2024 for a clean year-view
    sub = df[df.index.year == 2024]
    sub_storms = storms[storms["start"].dt.year == 2024]

    # Bz with -10 nT rule and storm bands
    fig_bz = go.Figure()
    fig_bz.add_trace(go.Scatter(
        x=sub.index, y=sub["Bz_mean"],
        mode="lines", line=dict(color=THEME["ink"], width=0.6),
        name="Bz", hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} nT<extra></extra>",
    ))
    fig_bz.add_hline(y=-10, line_color=THEME["accent"], line_dash="dot", line_width=1)
    for _, row in sub_storms.iterrows():
        fig_bz.add_vrect(x0=row["start"], x1=row["end"], fillcolor=THEME["accent"],
                         opacity=0.10, line_width=0)
    fig_bz.update_layout(height=260, margin=dict(l=60, r=16, t=8, b=20),
                         yaxis=dict(title="Bz / nT"), showlegend=False)

    # epsilon coupling
    fig_eps = go.Figure()
    fig_eps.add_trace(go.Scatter(
        x=sub.index, y=sub["epsilon_coupling"],
        mode="lines", line=dict(color=THEME["accent"], width=0.6),
        name="epsilon"))
    for _, row in sub_storms.iterrows():
        fig_eps.add_vrect(x0=row["start"], x1=row["end"], fillcolor=THEME["accent"],
                          opacity=0.08, line_width=0)
    fig_eps.update_layout(height=200, margin=dict(l=60, r=16, t=8, b=24),
                          yaxis=dict(title="Newell eps  (a.u.)"), showlegend=False)

    # Storms table (top by intensity)
    tbl = sub_storms.copy()
    tbl["start"] = tbl["start"].dt.strftime("%Y-%m-%d  %H:%M")
    tbl["end"] = tbl["end"].dt.strftime("%H:%M")
    tbl["duration_h"] = tbl["duration_h"].round(2)
    tbl["Bz_min"] = tbl["Bz_min"].round(2)
    tbl = tbl.sort_values("Bz_min").reset_index(drop=True).head(20)

    return html.Div([
        html.Div("2024 - SAISON COMPLETE", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "marginBottom": "6px",
        }),
        dcc.Graph(figure=fig_bz, config={"displayModeBar": False}),
        dcc.Graph(figure=fig_eps, config={"displayModeBar": False}),
        html.Div("CATALOGUE 2024 - TOP 20 (Bz min le plus bas)", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "margin": "16px 0 6px 0",
        }),
        dash_table.DataTable(
            data=tbl.to_dict("records"),
            columns=[{"name": c.upper(), "id": c} for c in ["start", "end", "Bz_min", "duration_h"]],
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "JetBrains Mono, monospace", "fontSize": "12px",
                        "border": "none", "borderBottom": f"1px solid {THEME['rule']}",
                        "backgroundColor": THEME["bg"], "color": THEME["ink"],
                        "padding": "8px 14px"},
            style_header={"backgroundColor": THEME["bg"], "borderBottom": f"1.5px solid {THEME['ink']}",
                          "color": THEME["muted"], "fontWeight": "normal",
                          "fontFamily": "JetBrains Mono, monospace", "fontSize": "10px",
                          "letterSpacing": "0.15em", "padding": "8px 14px"},
        ),
    ], style={"padding": "12px 6px"})


def _tab_c() -> html.Div:
    """Forecast widget: load forecast_comparison.csv and let user pick model+horizon to see a 7-day forecast."""
    csv = REPORTS_DIR / "forecast_comparison.csv"
    if not csv.exists():
        return _placeholder_tab("FORECAST", "Aucun resultat de prevision",
                                "Lancez d'abord python -m pfa_vtec train.")
    df = pd.read_csv(csv)
    # Build a comparison table sorted by horizon then rmse
    df = df.sort_values(["horizon", "rmse"])
    # Plot RMSE per model per horizon
    fig = go.Figure()
    color_for = {"hgb": THEME["accent"], "rf": THEME["gray_50"],
                 "persistence": THEME["muted"], "climatology": THEME["gray_70"]}
    dash_for = {"persistence": "dash", "climatology": "dot"}
    for m in df["model"].unique():
        sub = df[df["model"] == m].sort_values("horizon")
        fig.add_trace(go.Scatter(
            x=sub["horizon"], y=sub["rmse"], mode="lines+markers",
            line=dict(color=color_for.get(m, THEME["ink"]), width=2 if m == "hgb" else 1.2,
                      dash=dash_for.get(m, "solid")),
            marker=dict(size=6, color=color_for.get(m, THEME["ink"])),
            name=m,
        ))
    fig.update_layout(height=360, margin=dict(l=60, r=16, t=20, b=40),
                      xaxis=dict(title="Horizon (h)", tickmode="array",
                                 tickvals=df["horizon"].unique().tolist()),
                      yaxis=dict(title="RMSE  /  TECU"))

    # Best model summary tile-strip
    best_per_h = df.sort_values("rmse").drop_duplicates("horizon").sort_values("horizon")

    return html.Div([
        html.Div("PREVISION - RMSE PAR MODELE x HORIZON  (test 2025)", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "marginBottom": "6px",
        }),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div("BEST MODEL PAR HORIZON", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "margin": "18px 0 6px 0",
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "0"},
                 children=[
                     kpi_tile(f"H = {row['horizon']} H", f"{row['rmse']:.2f}", "TECU")
                     for _, row in best_per_h.iterrows()
                 ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "0", "marginTop": "16px"},
                 children=[
                     kpi_tile(f"SKILL h={int(row['horizon'])}", f"{row['skill']*100:+.1f}", "%")
                     for _, row in best_per_h.iterrows()
                 ]),
    ], style={"padding": "12px 6px"})


def _tab_d() -> html.Div:
    """Anomaly detection: show IForest score series + events + confusion + P/R/F1."""
    cmp_csv = REPORTS_DIR / "anomaly_comparison.csv"
    scores_csv = REPORTS_DIR / "anom_iforest_scores.csv"
    if not (cmp_csv.exists() and scores_csv.exists()):
        return _placeholder_tab("ANOMALIES", "Aucun resultat de detection",
                                "Lancez d'abord python -m pfa_vtec detect.")

    cmp = pd.read_csv(cmp_csv)
    scores = pd.read_csv(scores_csv, parse_dates=["time"]).set_index("time")["iforest"]
    storms = load_storms()
    # restrict to test period 2025 for clarity
    sub = scores[scores.index.year == 2025]
    sub_storms = storms[storms["start"].dt.year == 2025]
    events = pd.read_csv(REPORTS_DIR / "anom_iforest_events.csv", parse_dates=["start", "end"])
    sub_events = events[events["start"].dt.year == 2025]

    fig = go.Figure()
    for _, row in sub_storms.iterrows():
        fig.add_vrect(x0=row["start"], x1=row["end"], fillcolor=THEME["gray_50"],
                      opacity=0.25, line_width=0)
    fig.add_trace(go.Scatter(x=sub.index, y=sub.values, mode="lines",
                             line=dict(color=THEME["ink"], width=0.6), name="iforest"))
    if len(sub_events):
        ev_y = sub.reindex(sub_events["start"], method="nearest").values
        fig.add_trace(go.Scatter(
            x=sub_events["start"], y=ev_y, mode="markers",
            marker=dict(color=THEME["accent"], size=9, symbol="diamond",
                        line=dict(color=THEME["bg"], width=1)),
            name="detection",
        ))
    fig.update_layout(height=360, margin=dict(l=60, r=16, t=20, b=40),
                      yaxis=dict(title="Score Isolation Forest"), showlegend=False)

    iforest = cmp[cmp["method"] == "iforest"].iloc[0]
    ocsvm   = cmp[cmp["method"] == "ocsvm"].iloc[0]

    return html.Div([
        html.Div("ANOMALIES - SCORE 2025 (test) + DETECTIONS", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "marginBottom": "6px",
        }),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div("COMPARAISON DETECTEURS (event-level, tolerance +/- 3 h)", className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["muted"], "letterSpacing": "0.18em", "margin": "18px 0 6px 0",
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "0"},
                 children=[
                     kpi_tile("IFOREST PRECISION", f"{iforest['precision']*100:.1f}", "%"),
                     kpi_tile("IFOREST RECALL",    f"{iforest['recall']*100:.1f}",    "%"),
                     kpi_tile("IFOREST F1",        f"{iforest['f1']:.3f}",            ""),
                     kpi_tile("OCSVM F1",          f"{ocsvm['f1']:.3f}",              ""),
                 ]),
    ], style={"padding": "12px 6px"})


def _placeholder_tab(eyebrow: str, title: str, sub: str) -> html.Div:
    return html.Div([
        html.Div(eyebrow, className="ui", style={
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "11px",
            "color": THEME["accent"], "letterSpacing": "0.18em", "marginBottom": "8px",
        }),
        html.H2(title, style={
            "fontFamily": "EB Garamond, Garamond, serif",
            "fontWeight": 400, "fontSize": "32px", "letterSpacing": "-0.01em",
            "color": THEME["ink"], "margin": "0 0 14px 0",
        }),
        hairline(),
        html.Div(sub, style={
            "marginTop": "20px", "color": THEME["muted"],
            "fontFamily": "JetBrains Mono, monospace", "fontSize": "12px",
        }),
    ], style={"padding": "20px 6px 60px 6px", "minHeight": "420px"})
