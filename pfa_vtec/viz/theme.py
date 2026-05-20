"""Editorial theme - one accent, hairlines instead of shadows, mono numerals.

OKLCH source palette (CSS variables in the dashboard mirror these):
    --bg:     oklch(98% 0.005 240)   -> #F6F7F8
    --bg2:    oklch(96% 0.008 240)   -> #EEF0F2
    --ink:    oklch(11% 0.012 240)   -> #16181C
    --muted:  oklch(45% 0.013 240)   -> #5F6470
    --rule:   oklch(88% 0.010 240)   -> #D7D9DD
    --accent: oklch(72% 0.18 38)     -> #E55A1F

Only the accent is colored. Secondary traces use grayscale tiers.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import plotly.graph_objects as go
import plotly.io as pio

from ..io_paths import CFG, ROOT

THEME: dict[str, str] = CFG["theme"]


def _register_local_fonts() -> None:
    """Tell matplotlib about the TTF files bundled in rapport/fonts/.

    Idempotent — safe to call repeatedly.
    """
    fonts_dir = ROOT / "rapport" / "fonts"
    if not fonts_dir.exists():
        return
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:
            pass


_register_local_fonts()


# ───────────────────────────────────────────────────────────
# Matplotlib editorial style
# ───────────────────────────────────────────────────────────
def apply_matplotlib_editorial() -> None:
    """Apply the editorial style to matplotlib globally.

    Use this in `viz/static.py` figure builders. Output PDF for the LaTeX
    rapport (vector) and PNG for the dashboard / slides.
    """
    plt.rcParams.update({
        "figure.facecolor":   THEME["bg"],
        "axes.facecolor":     THEME["bg"],
        "savefig.facecolor":  THEME["bg"],
        "savefig.edgecolor":  "none",
        "axes.edgecolor":     THEME["ink"],
        "axes.linewidth":     0.6,
        "axes.labelcolor":    THEME["ink"],
        "xtick.color":        THEME["ink"],
        "ytick.color":        THEME["ink"],
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.minor.width":  0.3,
        "ytick.minor.width":  0.3,
        "text.color":         THEME["ink"],
        "grid.color":         THEME["rule"],
        "grid.linewidth":     0.4,
        "grid.alpha":         1.0,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.frameon":     False,
        "legend.fontsize":    8.5,
        "font.family":        ["EB Garamond", "Garamond", "serif"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "normal",
        "axes.labelsize":     9.5,
        "figure.dpi":         140,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def hairline(ax, where="bottom", color=None, lw=0.6) -> None:
    """Add a clean hairline rule to an axis (no full spine box)."""
    c = color or THEME["ink"]
    ax.spines[where].set_color(c)
    ax.spines[where].set_linewidth(lw)


# ───────────────────────────────────────────────────────────
# Plotly editorial template
# ───────────────────────────────────────────────────────────
def plotly_editorial_template() -> go.layout.Template:
    """Build a Plotly Template named 'editorial'."""
    tpl = go.layout.Template()
    tpl.layout = dict(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(
            family="EB Garamond, Garamond, Cormorant Garamond, serif",
            color=THEME["ink"],
            size=14,
        ),
        title=dict(font=dict(size=20, color=THEME["ink"]), x=0, xanchor="left"),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(
            gridcolor=THEME["rule"],
            gridwidth=0.4,
            zeroline=False,
            linecolor=THEME["ink"],
            linewidth=0.6,
            ticks="outside",
            tickcolor=THEME["ink"],
            tickfont=dict(family="JetBrains Mono, Menlo, monospace", size=11),
            title=dict(font=dict(family="JetBrains Mono, monospace", size=11, color=THEME["muted"])),
        ),
        yaxis=dict(
            gridcolor=THEME["rule"],
            gridwidth=0.4,
            zeroline=False,
            linecolor=THEME["ink"],
            linewidth=0.6,
            ticks="outside",
            tickcolor=THEME["ink"],
            tickfont=dict(family="JetBrains Mono, Menlo, monospace", size=11),
            title=dict(font=dict(family="JetBrains Mono, monospace", size=11, color=THEME["muted"])),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME["rule"],
            borderwidth=0,
            font=dict(family="JetBrains Mono, monospace", size=11, color=THEME["muted"]),
        ),
        colorway=[
            THEME["accent"],   # primary / hero trace
            THEME["gray_50"],  # secondary
            THEME["gray_70"],  # tertiary
            THEME["gray_30"],
            THEME["muted"],
        ],
        hoverlabel=dict(
            bgcolor=THEME["bg2"],
            bordercolor=THEME["rule"],
            font=dict(family="JetBrains Mono, monospace", size=11, color=THEME["ink"]),
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=THEME["muted"],
            activecolor=THEME["accent"],
        ),
    )
    return tpl


def register_plotly_template(default: bool = True) -> None:
    pio.templates["editorial"] = plotly_editorial_template()
    if default:
        pio.templates.default = "editorial"


# Register on import so `import plotly.express as px` already inherits it
register_plotly_template(default=True)
