"""
Step 3: Run pretrained ALIGNN inference and train classical baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    ensure_directories,
    read_json,
    set_random_seeds,
    write_json,
)


def install_dgl_graphbolt_stub() -> None:
    """Avoid DGL 2.1 import failure when PyTorch is newer than bundled GraphBolt."""
    import sys
    import types

    sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))


def load_catalog() -> pd.DataFrame:
    catalog_path = RESULTS_DIR / "material_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError("Missing material catalog. Run data_explore.py first.")
    return pd.read_csv(catalog_path)


def load_split(name: str) -> pd.DataFrame:
    split_path = DATA_DIR / f"{name}_id_prop.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return pd.read_csv(split_path)


def prepare_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = load_catalog().set_index("jid")
    train = load_split("train").rename(columns={"id": "jid"})
    test = load_split("test").rename(columns={"id": "jid"})

    train = train.join(catalog, on="jid", rsuffix="_meta")
    test = test.join(catalog, on="jid", rsuffix="_meta")
    train["target"] = train["target"].astype(float)
    test["target"] = test["target"].astype(float)
    train = _fill_missing_formulas(train)
    test = _fill_missing_formulas(test)
    return train, test


def _formula_from_structure(structure_path: str) -> str | None:
    try:
        atoms = json.loads(Path(structure_path).read_text())
    except Exception:
        return None

    elements = atoms.get("elements") or atoms.get("species") or []
    if not elements:
        return None

    counts: dict[str, int] = {}
    order: list[str] = []
    for element in elements:
        if element not in counts:
            counts[element] = 0
            order.append(element)
        counts[element] += 1
    return "".join(f"{element}{counts[element] if counts[element] > 1 else ''}" for element in order)


def _fill_missing_formulas(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    formulas = frame["formula"].astype(str)
    missing_mask = frame["formula"].isna() | formulas.str.strip().eq("") | formulas.str.lower().eq("nan")
    if missing_mask.any():
        frame.loc[missing_mask, "formula"] = frame.loc[missing_mask, "structure_path"].map(_formula_from_structure)
    return frame


def build_magpie_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    featurizer = ElementProperty.from_preset("magpie")

    def featurize(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        rows = []
        keep_indices = []
        for index, formula in frame["formula"].items():
            try:
                rows.append(featurizer.featurize(Composition(str(formula))))
                keep_indices.append(index)
            except Exception:
                continue
        filtered = frame.loc[keep_indices].reset_index(drop=True)
        return np.asarray(rows, dtype=float), filtered

    train_x, train_df = featurize(train_df)
    test_x, test_df = featurize(test_df)
    feature_names = featurizer.feature_labels()
    return train_x, test_x, feature_names, train_df, test_df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE_eV": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_eV": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    train_x, test_x, feature_names, train_df, test_df = build_magpie_features(train_df, test_df)
    y_train = train_df["target"].to_numpy(dtype=float)
    y_test = test_df["target"].to_numpy(dtype=float)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(train_x, y_train)
    rf_pred = rf.predict(test_x)

    ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    ridge.fit(train_x, y_train)
    ridge_pred = ridge.predict(test_x)

    top_indices = np.argsort(rf.feature_importances_)[::-1][:10]
    top_features = [
        {"feature": feature_names[index], "importance": float(rf.feature_importances_[index])}
        for index in top_indices
    ]

    return {
        "random_forest": {
            "method": "Random Forest + Magpie",
            "predictions": rf_pred.tolist(),
            "targets": y_test.tolist(),
            "ids": test_df["jid"].tolist(),
            **regression_metrics(y_test, rf_pred),
            "top_features": top_features,
        },
        "ridge": {
            "method": "Ridge + Magpie",
            "predictions": ridge_pred.tolist(),
            "targets": y_test.tolist(),
            "ids": test_df["jid"].tolist(),
            **regression_metrics(y_test, ridge_pred),
        },
    }


def _predict_single_material(model, atoms):
    if hasattr(model, "pred_from_atoms"):
        prediction = model.pred_from_atoms(atoms)
        return float(np.asarray(prediction).reshape(-1)[0])
    if hasattr(model, "predict"):
        prediction = model.predict(atoms)
        return float(np.asarray(prediction).reshape(-1)[0])
    raise AttributeError("Unsupported pretrained ALIGNN model interface.")


def run_pretrained_alignn(test_df: pd.DataFrame) -> dict[str, object] | None:
    install_dgl_graphbolt_stub()
    try:
        from alignn import pretrained
        from jarvis.core.atoms import Atoms
        from jarvis.db.jsonutils import loadjson
    except Exception as exc:
        print(f"⚠️ Skipping pretrained ALIGNN: missing dependency ({exc})")
        return None

    model = None
    model_name = "alignn_ffdb"
    load_errors: list[str] = []

    try:
        model = pretrained.get_alignn_ffdb_model()
    except Exception as exc:
        load_errors.append(f"get_alignn_ffdb_model failed: {exc}")

    if model is None:
        for candidate_name in (
            "jv_optb88vdw_bandgap_alignn",
            "jv_bandgap_alignn",
            "jv_mbj_bandgap_alignn",
        ):
            try:
                model = pretrained.get_figshare_model(model_name=candidate_name)
                model_name = candidate_name
                break
            except Exception as exc:
                load_errors.append(f"{candidate_name} failed: {exc}")

    if model is None:
        print("⚠️ Could not load a pretrained ALIGNN model.")
        for message in load_errors[:3]:
            print(f"   {message}")
        return None

    predictions = []
    targets = []
    ids = []
    for row in test_df.itertuples(index=False):
        structure_path = Path(getattr(row, "structure_path"))
        try:
            atoms = Atoms.from_dict(loadjson(str(structure_path)))
            predictions.append(_predict_single_material(model, atoms))
            targets.append(float(row.target))
            ids.append(row.jid)
        except Exception:
            continue

    if not predictions:
        print("⚠️ Pretrained ALIGNN loaded, but no predictions completed successfully.")
        return None

    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    return {
        "method": "ALIGNN (Pretrained)",
        "ids": ids,
        "targets": targets,
        "predictions": predictions,
        "model_name": model_name,
        **regression_metrics(y_true, y_pred),
    }


def save_predictions(
    test_df: pd.DataFrame,
    pretrained_result: dict[str, object] | None,
    baseline_results: dict[str, dict[str, object]],
) -> None:
    ids = test_df["jid"].to_numpy(dtype=str)
    targets = test_df["target"].to_numpy(dtype=float)

    aligned_pretrained = np.full(len(ids), np.nan, dtype=float)
    if pretrained_result is not None:
        lookup = dict(zip(pretrained_result["ids"], pretrained_result["predictions"]))
        aligned_pretrained = np.asarray([lookup.get(identifier, np.nan) for identifier in ids], dtype=float)

    rf_pred = np.asarray(baseline_results["random_forest"]["predictions"], dtype=float)
    ridge_pred = np.asarray(baseline_results["ridge"]["predictions"], dtype=float)

    np.savez(
        RESULTS_DIR / "predictions.npz",
        ids=ids,
        targets=targets,
        alignn_pretrained=aligned_pretrained,
        random_forest=rf_pred,
        ridge=ridge_pred,
    )

    benchmark = {
        key: {
            sub_key: value
            for sub_key, value in payload.items()
            if sub_key not in {"predictions", "targets", "ids"}
        }
        for key, payload in baseline_results.items()
    }
    if pretrained_result is not None:
        benchmark["alignn_pretrained"] = {
            key: value
            for key, value in pretrained_result.items()
            if key not in {"predictions", "targets", "ids"}
        }
    write_json(RESULTS_DIR / "pretrain_benchmark.json", benchmark)


def run(quick: bool = False) -> dict[str, object]:
    del quick
    ensure_directories()
    set_random_seeds()

    train_df, test_df = prepare_feature_matrices()
    baseline_results = run_baselines(train_df, test_df)
    pretrained_result = run_pretrained_alignn(test_df)
    save_predictions(test_df, pretrained_result, baseline_results)

    summary = {
        "baselines": {
            key: {k: v for k, v in value.items() if k not in {"predictions", "targets", "ids", "top_features"}}
            for key, value in baseline_results.items()
        },
        "pretrained": None
        if pretrained_result is None
        else {k: v for k, v in pretrained_result.items() if k not in {"predictions", "targets", "ids"}},
    }
    print(f"💾 Saved prediction bundle to {RESULTS_DIR / 'predictions.npz'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained ALIGNN and baseline regressors")
    parser.add_argument("--quick", action="store_true", help="Reserved for pipeline compatibility")
    args = parser.parse_args()

    try:
        run(quick=args.quick)
    except Exception as exc:
        print(f"❌ Prediction step failed: {exc}")
        raise


if __name__ == "__main__":
    main()
