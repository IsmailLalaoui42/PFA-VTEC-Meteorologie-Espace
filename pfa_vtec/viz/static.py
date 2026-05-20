"""Editorial-grade static figures for the LaTeX rapport and slides.

Each figure writes both PDF (vector, for LaTeX) and PNG (raster, for slides).
Output: D:\\pfa\\reports\\figures\\fig-<slug>.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ..data import load_merged, load_storms
from ..io_paths import FIGURES_DIR
from .theme import THEME, apply_matplotlib_editorial


def _save(fig, slug: str) -> tuple[Path, Path]:
    apply_matplotlib_editorial()  # idempotent
    pdf = FIGURES_DIR / f"fig-{slug}.pdf"
    png = FIGURES_DIR / f"fig-{slug}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=THEME["bg"])
    fig.savefig(png, bbox_inches="tight", facecolor=THEME["bg"], dpi=200)
    plt.close(fig)
    return pdf, png


# ───────────────────────────────────────────────────────────
# Hero 1 - VTEC timeseries with storm overlay (2 years for clarity)
# ───────────────────────────────────────────────────────────
def fig_vtec_with_storms(years: tuple[int, int] = (2023, 2024)) -> None:
    apply_matplotlib_editorial()
    df = load_merged().resample("1h").mean()
    storms = load_storms()
    sub = df[(df.index.year >= years[0]) & (df.index.year <= years[1])]

    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    # Storm bands first (behind line)
    in_window = storms[(storms["start"] >= sub.index.min()) & (storms["end"] <= sub.index.max())]
    for _, row in in_window.iterrows():
        ax.axvspan(row["start"], row["end"], color=THEME["accent"], alpha=0.10, lw=0)

    ax.plot(sub.index, sub["vtec_mean"], color=THEME["accent"], lw=0.7, label="VTEC")

    ax.set_title(f"VTEC au-dessus de Beni Mellal  -  {years[0]}-{years[1]}",
                 loc="left", color=THEME["ink"], fontsize=13)
    ax.set_ylabel("VTEC  /  TECU", color=THEME["muted"])
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, axis="y", color=THEME["rule"], lw=0.4)
    for s in ax.spines.values(): s.set_visible(False)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(THEME["ink"]); ax.spines["bottom"].set_linewidth(0.6)
    # Tick label font
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    _save(fig, "vtec-timeseries-storms")


# ───────────────────────────────────────────────────────────
# Hero 2 - VTEC distribution by year (small multiples ridge)
# ───────────────────────────────────────────────────────────
def fig_vtec_distribution() -> None:
    apply_matplotlib_editorial()
    df = load_merged().resample("1h").mean().dropna(subset=["vtec_mean"])
    years = sorted(df.index.year.unique())
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    bins = np.linspace(0, max(80, df["vtec_mean"].quantile(0.999)), 60)
    for i, y in enumerate(years):
        sub = df.loc[df.index.year == y, "vtec_mean"]
        if len(sub) < 100: continue
        hist, edges = np.histogram(sub, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        # Stagger: each year shifted vertically by ~0.04
        offset = i * 0.025
        ax.fill_between(centers, offset, offset + hist * 1.0,
                        color=THEME["accent"] if y == 2024 else THEME["gray_70"],
                        alpha=0.45 if y == 2024 else 0.20, lw=0)
        ax.plot(centers, offset + hist * 1.0,
                color=THEME["accent"] if y == 2024 else THEME["ink"],
                lw=0.7 if y == 2024 else 0.4)
        ax.text(bins[-1] * 0.98, offset + 0.005, str(y),
                fontfamily="monospace", fontsize=9, color=THEME["muted"], ha="right", va="bottom")

    ax.set_xlim(0, bins[-1])
    ax.set_yticks([])
    ax.set_xlabel("VTEC  /  TECU", color=THEME["muted"])
    for s in ("top", "left", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(THEME["ink"]); ax.spines["bottom"].set_linewidth(0.6)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    _save(fig, "vtec-distribution-ridge")


# ───────────────────────────────────────────────────────────
# Hero 3 - Correlation matrix VTEC vs drivers
# ───────────────────────────────────────────────────────────
def fig_correlation_matrix() -> None:
    apply_matplotlib_editorial()
    df = load_merged().resample("1h").mean()
    cols = ["vtec_mean", "Bz_mean", "Vsw_mean", "Pdyn_mean", "Np_mean", "Bs", "epsilon_coupling"]
    sub = df[cols].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    # Build a one-accent diverging palette (cool gray -> warm accent)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("editorial-diverging", [
        "#3F4248",  # gray_30 (cool ink)
        THEME["bg"],
        THEME["accent"],
    ])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    nice = [c.replace("_mean", "").replace("_coupling", " (eps)") for c in cols]
    ax.set_xticklabels(nice, rotation=45, ha="right", fontfamily="monospace", color=THEME["muted"], fontsize=9)
    ax.set_yticklabels(nice, fontfamily="monospace", color=THEME["muted"], fontsize=9)
    # Annotate
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=THEME["ink"] if abs(v) < 0.6 else THEME["bg"],
                    fontfamily="monospace", fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "correlation-matrix")


# ───────────────────────────────────────────────────────────
# Hero 4 - Monthly climatology heatmap (month x hour)
# ───────────────────────────────────────────────────────────
def fig_climatology_heatmap() -> None:
    apply_matplotlib_editorial()
    df = load_merged().resample("1h").mean().dropna(subset=["vtec_mean"])
    pivot = df.groupby([df.index.month, df.index.hour])["vtec_mean"].median().unstack(level=1)

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("editorial-seq", [
        THEME["bg"],
        "#F3BC9F",         # soft accent
        THEME["accent"],
    ])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)], fontfamily="monospace",
                       color=THEME["muted"], fontsize=9)
    ax.set_yticks(range(12))
    months = ["jan", "fev", "mar", "avr", "mai", "jun", "jui", "aou", "sep", "oct", "nov", "dec"]
    ax.set_yticklabels(months, fontfamily="monospace", color=THEME["muted"], fontsize=9)
    ax.set_xlabel("Heure UTC", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    # Colorbar (hairline)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=THEME["muted"], labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
    cbar.set_label("VTEC mediane / TECU", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    _save(fig, "climatology-heatmap")


# ───────────────────────────────────────────────────────────
# Master orchestrator
# ───────────────────────────────────────────────────────────
def build_all_hero_figures() -> None:
    from rich.console import Console
    console = Console()
    pairs = [
        ("vtec-timeseries-storms", fig_vtec_with_storms),
        ("vtec-distribution-ridge", fig_vtec_distribution),
        ("correlation-matrix",      fig_correlation_matrix),
        ("climatology-heatmap",     fig_climatology_heatmap),
    ]
    for slug, fn in pairs:
        try:
            fn()
            console.print(f"[green]ok[/green]  fig-{slug}.{{pdf,png}}")
        except Exception as e:
            console.print(f"[red]FAIL[/red]  fig-{slug}  - {e}")


def build_forecast_comparison_figure() -> None:
    """Produce a comparison plot from reports/forecast_comparison.csv (D3+)."""
    from rich.console import Console
    console = Console()
    apply_matplotlib_editorial()
    csv = (FIGURES_DIR.parent / "forecast_comparison.csv")
    if not csv.exists():
        console.print(f"[yellow]skip[/yellow] {csv} missing - run `train` first")
        return
    df = pd.read_csv(csv)
    df["horizon_h"] = df["horizon"]  # already hours

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    models = df["model"].unique().tolist()
    # color order: accent for best, gray tiers for the rest, persistence dotted
    best_model = df.loc[df.groupby("horizon")["rmse"].idxmin(), "model"].mode().iat[0]
    palette = {}
    grays = [THEME["gray_70"], THEME["gray_50"], THEME["gray_30"], THEME["muted"]]
    gi = 0
    for m in models:
        if m == best_model:
            palette[m] = THEME["accent"]
        elif m == "persistence":
            palette[m] = THEME["muted"]
        else:
            palette[m] = grays[gi % len(grays)]; gi += 1

    for m in models:
        sub = df[df["model"] == m].sort_values("horizon_h")
        ls = "--" if m == "persistence" else "-"
        lw = 1.6 if m == best_model else 1.0
        ax.plot(sub["horizon_h"], sub["rmse"], marker="o", ms=4, ls=ls, lw=lw,
                color=palette[m], label=m)

    ax.set_xlabel("Horizon  /  h", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    ax.set_ylabel("RMSE  /  TECU", color=THEME["muted"], fontfamily="monospace", fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(THEME["ink"]); ax.spines["left"].set_color(THEME["ink"])
    ax.spines["bottom"].set_linewidth(0.6); ax.spines["left"].set_linewidth(0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace"); lbl.set_color(THEME["muted"]); lbl.set_fontsize(9)
    ax.grid(True, axis="y", color=THEME["rule"], lw=0.4)

    ax.text(0.0, 1.06, "FIG. 05", transform=ax.transAxes,
            color=THEME["accent"], fontfamily="monospace", fontsize=9, ha="left", va="bottom")
    ax.text(0.07, 1.06, " -  comparaison RMSE par modele et horizon  (test 2025)",
            transform=ax.transAxes, color=THEME["muted"],
            fontfamily="monospace", fontsize=9, ha="left", va="bottom")
    _save(fig, "forecast-comparison")
    console.print(f"[green]ok[/green]  fig-forecast-comparison.{{pdf,png}}")
