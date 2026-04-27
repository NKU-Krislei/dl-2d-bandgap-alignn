"""
Step 2: Explore the downloaded data and create deterministic train/val/test splits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    DATA_DIR,
    FIGURES_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    ensure_directories,
    first_existing_bandgap_column,
    infer_family,
    safe_float,
    safe_int,
    set_random_seeds,
    setup_matplotlib,
    write_csv,
    write_json,
)


def load_jarvis_data() -> tuple[list[dict[str, str]], list[str], Path]:
    id_prop_path = RAW_DATA_DIR / "id_prop.csv"
    if not id_prop_path.exists():
        raise FileNotFoundError(
            f"Expected dataset metadata at {id_prop_path}. Run data_download.py first."
        )

    with id_prop_path.open() as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)

    return rows, header, RAW_DATA_DIR


def normalize_materials(rows: list[dict[str, str]], header: list[str]) -> list[dict[str, object]]:
    bandgap_column = first_existing_bandgap_column(header)
    if bandgap_column is None:
        raise KeyError(f"No supported band-gap column found in dataset header: {header[:20]}")

    materials: list[dict[str, object]] = []
    for row in rows:
        jid = row.get("jid") or row.get("id")
        if not jid:
            continue

        bandgap = safe_float(row.get(bandgap_column))
        if bandgap is None or bandgap < 0:
            continue

        formula = row.get("formula") or row.get("pretty_formula") or jid
        material = {
            "jid": jid,
            "target": float(bandgap),
            "formula": formula,
            "nsites": safe_int(row.get("nsites"), 0) or 0,
            "spacegroup": row.get("spacegroup", ""),
            "formation_energy": safe_float(
                row.get("optb88vdw_energy_per_atom") or row.get("formation_energy_peratom"),
                0.0,
            )
            or 0.0,
            "volume": safe_float(row.get("optb88vdw_volume") or row.get("volume"), 0.0) or 0.0,
            "family": infer_family(formula=formula, jid=jid),
            "structure_path": str(RAW_DATA_DIR / f"{jid}.json"),
        }
        if Path(material["structure_path"]).exists():
            materials.append(material)

    if not materials:
        raise RuntimeError("No valid materials with both band gaps and structure files were found.")

    return materials


def plot_bandgap_distribution(materials: list[dict[str, object]]) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(materials)
    bandgaps = df["target"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(bandgaps, bins=80, color="#3b82f6", alpha=0.85, edgecolor="white")
    axes[0, 0].axvline(bandgaps.mean(), color="#dc2626", linestyle="--", label=f"Mean = {bandgaps.mean():.2f}")
    axes[0, 0].axvline(np.median(bandgaps), color="#f59e0b", linestyle="--", label=f"Median = {np.median(bandgaps):.2f}")
    axes[0, 0].set_title("(a) Band Gap Distribution")
    axes[0, 0].set_xlabel("Band Gap (eV)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    categories = pd.cut(
        df["target"],
        bins=[-0.01, 0.01, 3.0, np.inf],
        labels=["Metal", "Semiconductor", "Insulator"],
    ).value_counts(sort=False)
    axes[0, 1].pie(
        categories.values,
        labels=[f"{name}\n(n={count})" for name, count in categories.items()],
        autopct="%1.1f%%",
        colors=["#ef4444", "#10b981", "#0ea5e9"],
        startangle=90,
    )
    axes[0, 1].set_title("(b) Electronic Type Split")

    axes[1, 0].scatter(df["nsites"], df["target"], c=df["target"], cmap="viridis", s=8, alpha=0.35)
    axes[1, 0].set_title("(c) Band Gap vs Unit Cell Size")
    axes[1, 0].set_xlabel("Number of Sites")
    axes[1, 0].set_ylabel("Band Gap (eV)")

    family_counts = df["family"].value_counts().head(8)
    axes[1, 1].barh(family_counts.index[::-1], family_counts.values[::-1], color="#6366f1")
    axes[1, 1].set_title("(d) Top Material Families")
    axes[1, 1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "data_exploration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_split(path: Path, records: list[dict[str, object]]) -> None:
    rows = [{"id": record["jid"], "target": f"{record['target']:.6f}"} for record in records]
    write_csv(path, rows, ["id", "target"])


def create_splits(materials: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = materials.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    for split_name, records in splits.items():
        _write_split(DATA_DIR / f"{split_name}_id_prop.csv", records)

    return splits


def prepare_small_dataset(materials: list[dict[str, object]], max_samples: int = 1000) -> dict[str, list[dict[str, object]]]:
    subset = materials[: min(max_samples, len(materials))]
    total = len(subset)
    n_train = max(1, int(total * 0.8))
    n_val = max(1, int(total * 0.1))
    splits = {
        "train_small": subset[:n_train],
        "val_small": subset[n_train : n_train + n_val],
        "test_small": subset[n_train + n_val :],
    }

    for split_name, records in splits.items():
        _write_split(DATA_DIR / f"{split_name}_id_prop.csv", records)

    return splits


def save_material_catalog(materials: list[dict[str, object]]) -> None:
    fieldnames = [
        "jid",
        "formula",
        "target",
        "family",
        "nsites",
        "formation_energy",
        "volume",
        "spacegroup",
        "structure_path",
    ]
    write_csv(RESULTS_DIR / "material_catalog.csv", materials, fieldnames)


def save_dataset_stats(
    materials: list[dict[str, object]],
    splits: dict[str, list[dict[str, object]]],
    small_splits: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    df = pd.DataFrame(materials)
    stats = {
        "total_materials": int(len(df)),
        "bandgap_mean": float(df["target"].mean()),
        "bandgap_std": float(df["target"].std(ddof=0)),
        "bandgap_min": float(df["target"].min()),
        "bandgap_max": float(df["target"].max()),
        "family_counts": df["family"].value_counts().to_dict(),
        "splits": {key: len(value) for key, value in splits.items()},
        "small_splits": {key: len(value) for key, value in small_splits.items()},
        "n_metal": int((df["target"] < 0.01).sum()),
        "n_semiconductor": int(((df["target"] >= 0.01) & (df["target"] < 3.0)).sum()),
        "n_insulator": int((df["target"] >= 3.0).sum()),
    }
    write_json(RESULTS_DIR / "dataset_stats.json", stats)
    return stats


def run(quick: bool = False) -> dict[str, object]:
    ensure_directories()
    set_random_seeds()
    rows, header, _ = load_jarvis_data()
    materials = normalize_materials(rows, header)
    splits = create_splits(materials)
    small_splits = prepare_small_dataset(materials)
    save_material_catalog(materials)
    stats = save_dataset_stats(materials, splits, small_splits)
    plot_bandgap_distribution(materials)

    if quick:
        print(f"⚡ Quick mode ready: {len(small_splits['train_small'])} small-train samples")
    print(f"📊 Saved dataset stats to {RESULTS_DIR / 'dataset_stats.json'}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and split the JARVIS dataset")
    parser.add_argument("--quick", action="store_true", help="Prepare quick-mode artefacts as well")
    args = parser.parse_args()

    try:
        run(quick=args.quick)
    except Exception as exc:
        print(f"❌ Data exploration failed: {exc}")
        raise


if __name__ == "__main__":
    main()
