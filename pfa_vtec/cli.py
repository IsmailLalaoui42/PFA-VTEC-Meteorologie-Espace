"""Single CLI entrypoint for the package. Run via ``python -m pfa_vtec <cmd>``."""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="pfa-vtec",
    help="PFA VTEC and space weather - editorial-grade Phase 2 toolkit.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def smoke() -> None:
    """Verify the package wires up correctly."""
    from .data import load_merged, load_storms
    from .io_paths import CFG, FIGURES_DIR

    df = load_merged()
    storms = load_storms()
    console.print(f"[bold]CFG station[/bold]    : {CFG['station']['name']}")
    console.print(f"[bold]merged rows[/bold]    : {len(df):,}")
    console.print(f"[bold]storm events[/bold]   : {len(storms):,}")
    console.print(f"[bold]vtec coverage[/bold]  : {df['vtec_mean'].notna().mean()*100:.1f}%")
    console.print(f"[bold]figures dir[/bold]    : {FIGURES_DIR}")
    console.print("[green]smoke OK[/green]")


@app.command()
def dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Launch the Dash + Plotly editorial dashboard."""
    from .dashboard.app import build_app

    app_ = build_app()
    console.print(f"[bold]Dashboard[/bold] -> http://{host}:{port}")
    app_.run(host=host, port=port, debug=debug)


@app.command()
def features() -> None:
    """Build training features and persist a parquet cache."""
    from .features.builder import build_and_cache

    build_and_cache()


@app.command()
def train(
    model: str = typer.Option("all", help="all | persistence | climatology | rf | hgb | lstm | sarima"),
    horizon: int = typer.Option(0, help="0 = all configured horizons, else 1|3|6|24"),
) -> None:
    """Train forecasting model(s) and update reports/forecast_comparison.csv."""
    from .models.forecasting.runner import run_training

    run_training(model=model, horizon=horizon or None)


@app.command()
def detect(method: str = typer.Option("all", help="all | iforest | ocsvm | lstm_ae")) -> None:
    """Run anomaly detection and update reports/anomaly_comparison.csv."""
    from .models.anomaly.runner import run_detection

    run_detection(method=method)


@app.command()
def figures() -> None:
    """(Re)generate the editorial hero figures in reports/figures/ (PDF + PNG)."""
    from .viz.static import build_all_hero_figures

    build_all_hero_figures()


@app.command()
def report() -> None:
    """Compile the LaTeX rapport into reports/PFA_VTEC_LalaouiRachidi.pdf."""
    from .viz.report_compile import compile_rapport

    compile_rapport()


if __name__ == "__main__":
    app()
