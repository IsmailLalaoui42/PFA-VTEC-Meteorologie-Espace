"""Top-level layout - chrome + tabs + tab content."""
from __future__ import annotations

from dash import dcc, html
import pandas as pd

from ..viz.theme import THEME
from .components import hairline, kpi_tile, page_chrome, section_header


def build_layout(df_summary: dict) -> html.Div:
    """Build the static layout shell. Callbacks fill the dynamic parts."""
    return html.Div(
        id="root",
        style={
            "fontFamily": "EB Garamond, Garamond, serif",
            "color": THEME["ink"],
            "background": THEME["bg"],
            "minHeight": "100vh",
        },
        children=[
            page_chrome(top_caption=f"BENI MELLAL  -  32.34N  -  6.36W  -  30 MIN  -  {df_summary['n_rows']:,} ROWS"),
            html.Div(
                style={"padding": "20px 32px 8px 32px"},
                children=section_header(
                    eyebrow="DASHBOARD / 04 ONGLETS",
                    title="Visualisation, prevision et detection d anomalies VTEC",
                ),
            ),
            html.Div(
                style={"padding": "0 32px 4px 32px"},
                children=[
                    _kpi_strip(df_summary),
                ],
            ),
            html.Div(
                style={"padding": "8px 32px 32px 32px"},
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="tab-a",
                        parent_className="tabs-parent",
                        className="tabs",
                        children=[
                            dcc.Tab(label="A. VTEC EXPLORER",  value="tab-a", className="tab", selected_className="tab--selected"),
                            dcc.Tab(label="B. STORM CATALOG", value="tab-b", className="tab", selected_className="tab--selected"),
                            dcc.Tab(label="C. FORECAST",      value="tab-c", className="tab", selected_className="tab--selected"),
                            dcc.Tab(label="D. ANOMALIES",     value="tab-d", className="tab", selected_className="tab--selected"),
                        ],
                    ),
                    html.Div(id="tab-content", style={"marginTop": "18px"}),
                ],
            ),
            _footer(df_summary),
        ],
    )


def _kpi_strip(s: dict) -> html.Div:
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "0", "marginBottom": "16px"},
        children=[
            kpi_tile("PERIODE",        s["period_short"]),
            kpi_tile("ROWS 30 MIN",    f"{s['n_rows']:,}".replace(",", " ")),
            kpi_tile("VTEC COVERAGE",  f"{s['vtec_cov']:.1f}", "%"),
            kpi_tile("STORMS BZ<-10",  f"{s['n_storms']:,}".replace(",", " "), "evt"),
            kpi_tile("VTEC MAX",       f"{s['vtec_max']:.1f}", "TECU"),
        ],
    )


def _footer(s: dict) -> html.Div:
    return html.Div([
        hairline(),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "padding": "12px 32px"},
            children=[
                html.Span("ISMAIL LALAOUI RACHIDI  -  ENSA BENI MELLAL  -  M242", style={
                    "fontFamily": "JetBrains Mono, monospace",
                    "fontSize": "10px",
                    "color": THEME["muted"],
                    "letterSpacing": "0.15em",
                }),
                html.Span(f"v0.1  -  {pd.Timestamp.now().strftime('%Y-%m-%d')}", style={
                    "fontFamily": "JetBrains Mono, monospace",
                    "fontSize": "10px",
                    "color": THEME["muted"],
                    "letterSpacing": "0.15em",
                }),
            ],
        ),
    ])
