"""Extra editorial figures - storm catalog and anomaly events."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import load_storms, load_merged
from ..io_paths import FIGURES_DIR, REPORTS_DIR
from .theme import THEME, apply_matplotlib_editorial


def _save(fig, slug: str):
    pdf = FIGURES_DIR / f"fig-{slug}.pdf"
    png = FIGURES_DIR / f"fig-{slug}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=THEME["bg"])
    fig.savefig(png, bbox_inches="tight", facecolor=THEME["bg"], dpi=200)
    plt.close(fig)


def fig_storm_catalog_timeline() -> None:
    """Yearly count + duration distribution of catalogued storms."""
    apply_matplotlib_editorial()
    s = load_storms()
    s["year"] = s["start"].dt.year
    by_year = s.groupby("year").size()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: bar chart of yearly counts
    ax1.bar(by_year.index, by_year.values, color=THEME["accent"], width=0.7, edgecolor="none")
    ax1.set_xlabel("Annee", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    ax1.set_ylabel("Nombre d'evenements", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    for spine in ("top", "right"): ax1.spines[spine].set_visible(False)
    ax1.spines["bottom"].set_color(THEME["ink"]); ax1.spines["left"].set_color(THEME["ink"])
    ax1.spines["bottom"].set_linewidth(0.6); ax1.spines["left"].set_linewidth(0.6)
    ax1.grid(True, axis="y", color=THEME["rule"], lw=0.4)
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    # Right: histogram of duration_h, log y
    durations = s["duration_h"].clip(upper=24)
    ax2.hist(durations, bins=24, color=THEME["accent"], edgecolor="none")
    ax2.set_xlabel("Duree / h", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    ax2.set_ylabel("# evt", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    for spine in ("top", "right"): ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color(THEME["ink"]); ax2.spines["left"].set_color(THEME["ink"])
    ax2.spines["bottom"].set_linewidth(0.6); ax2.spines["left"].set_linewidth(0.6)
    ax2.grid(True, axis="y", color=THEME["rule"], lw=0.4)
    for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    _save(fig, "storm-catalog")


def fig_anomaly_events(method: str = "iforest") -> None:
    """Anomaly score timeseries with detected events highlighted + catalog overlay."""
    apply_matplotlib_editorial()
    scores_path = REPORTS_DIR / f"anom_{method}_scores.csv"
    events_path = REPORTS_DIR / f"anom_{method}_events.csv"
    if not scores_path.exists():
        return
    scores = pd.read_csv(scores_path, parse_dates=["time"]).set_index("time")[method]
    events = pd.read_csv(events_path, parse_dates=["start", "end"])
    storms = load_storms()
    # Restrict to a 1-year window for clarity (test year)
    yr = 2025
    scores = scores[scores.index.year == yr]
    events = events[(events["start"].dt.year == yr) | (events["end"].dt.year == yr)]
    storms = storms[(storms["start"].dt.year == yr)]

    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    # Storm catalog bands (cool gray, behind)
    for _, row in storms.iterrows():
        ax.axvspan(row["start"], row["end"], color=THEME["gray_70"], alpha=0.25, lw=0)
    # Anomaly score
    ax.plot(scores.index, scores.values, color=THEME["ink"], lw=0.6, alpha=0.8)
    # Detected events (accent dots at start of each)
    if len(events):
        ev_y = scores.reindex(events["start"], method="nearest").values
        ax.scatter(events["start"], ev_y, color=THEME["accent"], s=18,
                   edgecolor=THEME["bg"], linewidth=0.5, zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("Score d'anomalie", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(THEME["ink"]); ax.spines["left"].set_color(THEME["ink"])
    ax.spines["bottom"].set_linewidth(0.6); ax.spines["left"].set_linewidth(0.6)
    ax.grid(True, axis="y", color=THEME["rule"], lw=0.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    _save(fig, "anomaly-events")
