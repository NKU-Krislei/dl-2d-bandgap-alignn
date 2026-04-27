"""
Shared utilities for the 2D band-gap project.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
RAW_DATA_DIR = DATA_DIR / "jarvis_dft_3d"
MPLCONFIG_DIR = RESULTS_DIR / ".mplconfig"

RANDOM_SEED = 42

BANDGAP_COLUMNS = (
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bandgap",
    "gap",
)

FAMILY_KEYWORDS = {
    "TMD": ("S2", "Se2", "Te2", "Mo", "W"),
    "h-BN": ("B", "N"),
    "Phosphorene": ("P",),
    "MXene": ("Ti", "V", "Nb", "Ta", "C", "N"),
    "Oxide": ("O",),
    "Halide": ("F", "Cl", "Br", "I"),
    "Nitride": ("N",),
    "Carbide": ("C",),
    "Selenide": ("Se",),
    "Telluride": ("Te",),
    "Sulfide": ("S",),
}


def ensure_directories() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR, MPLCONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()


def setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_split_paths() -> dict[str, Path]:
    return {
        "train": DATA_DIR / "train_id_prop.csv",
        "val": DATA_DIR / "val_id_prop.csv",
        "test": DATA_DIR / "test_id_prop.csv",
        "train_small": DATA_DIR / "train_small_id_prop.csv",
        "val_small": DATA_DIR / "val_small_id_prop.csv",
        "test_small": DATA_DIR / "test_small_id_prop.csv",
        "catalog": RESULTS_DIR / "material_catalog.csv",
        "stats": RESULTS_DIR / "dataset_stats.json",
        "download_manifest": RESULTS_DIR / "download_manifest.json",
        "predictions": RESULTS_DIR / "predictions.npz",
        "pretrain_benchmark": RESULTS_DIR / "pretrain_benchmark.json",
        "training_history": RESULTS_DIR / "training_history.json",
        "evaluation_report": RESULTS_DIR / "evaluation_report.json",
        "learning_curve": RESULTS_DIR / "learning_curve.json",
    }


def first_existing_bandgap_column(columns: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in BANDGAP_COLUMNS:
        if candidate in available:
            return candidate
    return None


def infer_family(formula: str, jid: str = "") -> str:
    text = f"{formula} {jid}".strip()
    if not text:
        return "Unknown"

    lowered = text.lower()
    if "graphene" in lowered:
        return "Graphene-like"
    if "bn" in lowered or ("b" in lowered and "n" in lowered):
        return "h-BN"
    if "phosphorene" in lowered:
        return "Phosphorene"
    if "mxene" in lowered:
        return "MXene"

    for family, tokens in FAMILY_KEYWORDS.items():
        if all(token in text for token in tokens):
            return family

    for family, tokens in FAMILY_KEYWORDS.items():
        if any(token in text for token in tokens):
            return family

    return "Other"


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
