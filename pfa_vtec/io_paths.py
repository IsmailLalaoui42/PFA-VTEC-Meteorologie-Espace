"""Centralised config + path resolution.

The single source of truth is `config/config.yaml` at the repo root.
Every other module imports `CFG` from here, never reads the YAML directly.
"""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """Turn a config-relative path string into an absolute Path."""
    return (ROOT / rel).resolve()


# Frequently-used paths
OUTPUT_DIR  = resolve(CFG["paths"]["output_dir"])
REPORTS_DIR = resolve(CFG["paths"]["reports_dir"])
FIGURES_DIR = resolve(CFG["paths"]["figures_dir"])
MODELS_DIR  = resolve(CFG["paths"]["models_dir"])
MERGED_CSV  = resolve(CFG["paths"]["merged_csv"])
OMNI_CSV    = resolve(CFG["paths"]["omni_csv"])
VTEC_CSV    = resolve(CFG["paths"]["vtec_csv"])
STORMS_CSV  = resolve(CFG["paths"]["storms_csv"])

for d in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
