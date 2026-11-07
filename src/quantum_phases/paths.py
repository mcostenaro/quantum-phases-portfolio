from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_DMRG_DIR = DATA_DIR / "dmrg"
DATA_DMRG_PROCESSED_DIR = DATA_DMRG_DIR / "processed"
DATA_LABELED_DIR = DATA_DIR / "labeled"
DATA_MAHLOW_DIR = DATA_DIR / "mahlow"
DATA_MAHLOW_PROCESSED_DIR = DATA_MAHLOW_DIR / "processed"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_REFERENCE_DIR = DATA_DIR / "reference"
RESULTS_ROOT = REPO_ROOT / "results"


def experiment_results_dir(family: str, script_name: str) -> Path:
    """Return the canonical output directory for one experiment script."""
    family = str(family).strip().replace("\\", "/").strip("/")
    script_name = str(script_name).strip().replace("\\", "/").strip("/")
    if not family:
        raise ValueError("family must not be empty.")
    if not script_name:
        raise ValueError("script_name must not be empty.")
    return RESULTS_ROOT / family / script_name
