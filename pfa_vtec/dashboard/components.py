"""Reusable UI primitives - hairline-styled, mono-numeral, no shadows."""
from __future__ import annotations

from dash import html

from ..viz.theme import THEME


def hairline(orientation: str = "h", color: str | None = None, weight: float = 0.6) -> html.Div:
    c = color or THEME["rule"]
    if orientation == "h":
        return html.Div(style={"height": f"{weight}px", "background": c, "width": "100%"})
    return html.Div(style={"width": f"{weight}px", "background": c, "height": "100%"})


def kpi_tile(label: str, value: str, unit: str = "", muted_label: bool = True) -> html.Div:
    """Editorial KPI tile - serif numeral, mono label, hairline under."""
    return html.Div(
        className="kpi-tile",
        children=[
            html.Div(label, className="kpi-label ui", style={
                "color": THEME["muted"] if muted_label else THEME["ink"],
                "fontFamily": "JetBrains Mono, monospace",
                "fontSize": "11px",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
            }),
            html.Div(
                children=[
                    html.Span(value, className="kpi-num", style={
                        "fontFamily": "EB Garamond, Garamond, serif",
                        "fontSize": "36px",
                        "fontFeatureSettings": "'tnum' 1, 'lnum' 1",
                        "color": THEME["ink"],
                    }),
                    html.Span(" " + unit if unit else "", style={
                        "fontFamily": "JetBrains Mono, monospace",
                        "fontSize": "12px",
                        "color": THEME["muted"],
                        "marginLeft": "6px",
                    }),
                ],
                style={"marginTop": "2px", "lineHeight": "1.0"},
            ),
            hairline(weight=0.6),
        ],
        style={
            "padding": "14px 16px 10px 0",
            "minWidth": "120px",
        },
    )


def section_header(eyebrow: str, title: str) -> html.Div:
    """Mono small-caps eyebrow + serif H1 + hairline."""
    return html.Div([
        html.Div(eyebrow, className="ui", style={
            "fontFamily": "JetBrains Mono, monospace",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "0.18em",
            "color": THEME["accent"],
            "marginBottom": "6px",
        }),
        html.H1(title, style={
            "fontFamily": "EB Garamond, Garamond, serif",
            "fontWeight": 400,
            "fontSize": "44px",
            "lineHeight": "1.08",
            "letterSpacing": "-0.01em",
            "color": THEME["ink"],
            "margin": "0 0 14px 0",
        }),
        hairline(),
    ], style={"marginBottom": "24px"})


def page_chrome(top_caption: str) -> html.Div:
    """Top navigation strip - hairline, mono small-caps, accent rule on the left."""
    return html.Div(
        className="chrome",
        children=[
            html.Div(style={
                "display": "flex",
                "alignItems": "baseline",
                "justifyContent": "space-between",
                "padding": "16px 32px 12px 32px",
            }, children=[
                html.Div([
                    html.Span("PFA", className="ui", style={
                        "fontFamily": "JetBrains Mono, monospace",
                        "fontSize": "11px",
                        "color": THEME["accent"],
                        "letterSpacing": "0.2em",
                        "marginRight": "10px",
                    }),
                    html.Span("VTEC and space weather", style={
                        "fontFamily": "EB Garamond, Garamond, serif",
                        "fontSize": "16px",
                        "color": THEME["ink"],
                    }),
                ]),
                html.Div(top_caption, className="ui", style={
                    "fontFamily": "JetBrains Mono, monospace",
                    "fontSize": "11px",
                    "color": THEME["muted"],
                    "letterSpacing": "0.12em",
                    "textTransform": "uppercase",
                }),
            ]),
            hairline(),
        ],
    )
