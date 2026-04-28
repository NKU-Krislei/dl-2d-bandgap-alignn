"""
Step 5: Evaluate available models and generate analysis figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_directories,
    read_json,
    set_random_seeds,
    setup_matplotlib,
    write_json,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MAE_eV": float(np.mean(np.abs(residuals))),
        "RMSE_eV": float(np.sqrt(np.mean(residuals**2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot else 0.0,
    }


def load_prediction_frame() -> pd.DataFrame:
    prediction_path = RESULTS_DIR / "predictions.npz"
    catalog_path = RESULTS_DIR / "material_catalog.csv"
    if not prediction_path.exists() or not catalog_path.exists():
        return pd.DataFrame(
            columns=[
                "jid",
                "target",
                "alignn_pretrained",
                "alignn_self_trained",
                "random_forest",
                "ridge",
                "formula",
                "family",
            ]
        )

    bundle = np.load(prediction_path, allow_pickle=True)
    frame = pd.DataFrame(
        {
            "jid": bundle["ids"].astype(str),
            "target": bundle["targets"].astype(float),
            "alignn_pretrained": bundle["alignn_pretrained"].astype(float),
            "random_forest": bundle["random_forest"].astype(float),
            "ridge": bundle["ridge"].astype(float),
        }
    )
    self_trained_path = RESULTS_DIR / "self_trained_predictions.npz"
    if self_trained_path.exists():
        self_trained = np.load(self_trained_path, allow_pickle=True)
        self_frame = pd.DataFrame(
            {
                "jid": self_trained["ids"].astype(str),
                "alignn_self_trained": self_trained["predictions"].astype(float),
            }
        )
        frame = frame.merge(self_frame, on="jid", how="left")
    else:
        frame["alignn_self_trained"] = np.nan

    catalog = pd.read_csv(catalog_path)[["jid", "formula", "family"]]
    return frame.merge(catalog, on="jid", how="left")


def plot_prediction_panel(df: pd.DataFrame, column: str, title: str, filename: str) -> dict[str, float] | None:
    if column not in df.columns or "target" not in df.columns:
        return None
    valid = df[["target", column]].dropna()
    if valid.empty:
        return None

    setup_matplotlib()
    import matplotlib.pyplot as plt

    y_true = valid["target"].to_numpy(dtype=float)
    y_pred = valid[column].to_numpy(dtype=float)
    residuals = y_pred - y_true
    metrics = compute_metrics(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    max_val = float(max(y_true.max(), y_pred.max()) * 1.05)
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.35, color="#2563eb")
    axes[0].plot([0, max_val], [0, max_val], "--", color="#dc2626")
    axes[0].set_title(f"(a) {title}: Predicted vs Actual")
    axes[0].set_xlabel("DFT Band Gap (eV)")
    axes[0].set_ylabel("Predicted Band Gap (eV)")

    axes[1].hist(residuals, bins=40, color="#10b981", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.0, color="#dc2626", linestyle="--")
    axes[1].set_title("(b) Residual Distribution")
    axes[1].set_xlabel("Residual (eV)")
    axes[1].set_ylabel("Count")

    axes[2].scatter(y_true, np.abs(residuals), s=10, alpha=0.35, color="#7c3aed")
    axes[2].axhline(metrics["MAE_eV"], color="#f59e0b", linestyle="--", label=f"MAE = {metrics['MAE_eV']:.3f}")
    axes[2].set_title("(c) Absolute Error vs Band Gap")
    axes[2].set_xlabel("DFT Band Gap (eV)")
    axes[2].set_ylabel("Absolute Error (eV)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_training_history() -> bool:
    training_path = RESULTS_DIR / "training_history.json"
    payload = read_json(training_path, default=None)
    if not payload:
        return False

    train_loss = payload.get("train_loss", [])
    val_loss = payload.get("val_loss", [])
    val_mae = payload.get("val_mae", [])
    if not train_loss and not val_loss and not val_mae:
        return False

    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = np.arange(1, max(len(train_loss), len(val_loss), len(val_mae)) + 1)

    if train_loss:
        axes[0].plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train Loss", color="#2563eb")
    if val_loss:
        axes[0].plot(np.arange(1, len(val_loss) + 1), val_loss, label="Val Loss", color="#dc2626")
    axes[0].set_title("(a) Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if val_mae:
        axes[1].plot(np.arange(1, len(val_mae) + 1), val_mae, color="#10b981")
    else:
        axes[1].plot(epochs, np.zeros_like(epochs), color="#9ca3af")
    axes[1].set_title("(b) Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (eV)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_method_comparison(benchmark: dict[str, dict[str, float] | None]) -> None:
    benchmark = {
        key: payload
        for key, payload in benchmark.items()
        if payload is not None and not np.isnan(payload.get("MAE_eV", np.nan))
    }
    if not benchmark:
        return

    setup_matplotlib()
    import matplotlib.pyplot as plt

    methods = [payload.get("method", key) for key, payload in benchmark.items()]
    maes = [payload.get("MAE_eV", np.nan) for payload in benchmark.values()]
    rmses = [payload.get("RMSE_eV", np.nan) for payload in benchmark.values()]
    r2s = [payload.get("R2", np.nan) for payload in benchmark.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#2563eb", "#f97316", "#16a34a", "#7c3aed", "#f59e0b"]

    for axis, values, ylabel, title in zip(
        axes,
        (maes, rmses, r2s),
        ("MAE (eV)", "RMSE (eV)", "R²"),
        ("(a) MAE", "(b) RMSE", "(c) R²"),
    ):
        bars = axis.bar(np.arange(len(methods)), values, color=colors[: len(methods)])
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(methods, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        if values and not np.all(np.isnan(values)):
            y_max = float(np.nanmax(values))
            axis.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_family_evaluation(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty or "family" not in df.columns:
        return []
    metric_columns = ("alignn_self_trained", "alignn_pretrained", "random_forest", "ridge")
    rows = []
    for family, group in df.groupby("family"):
        family_row: dict[str, object] = {"family": family, "count": int(len(group))}
        for column in metric_columns:
            if column not in group.columns:
                family_row[f"{column}_MAE_eV"] = None
                continue
            valid = group[["target", column]].dropna()
            family_row[f"{column}_MAE_eV"] = (
                float(np.mean(np.abs(valid[column] - valid["target"]))) if not valid.empty else None
            )
        rows.append(family_row)

    result = sorted(rows, key=lambda item: item["count"], reverse=True)

    top = pd.DataFrame(result).head(6)
    if not top.empty:
        setup_matplotlib()
        import matplotlib.pyplot as plt

        plot_columns = [
            ("alignn_self_trained_MAE_eV", "ALIGNN (Self-Trained)", "#2563eb"),
            ("random_forest_MAE_eV", "Random Forest", "#f97316"),
            ("ridge_MAE_eV", "Ridge", "#16a34a"),
        ]
        if top.get("alignn_pretrained_MAE_eV", pd.Series(dtype=float)).notna().any():
            plot_columns.insert(1, ("alignn_pretrained_MAE_eV", "ALIGNN (Pretrained)", "#7c3aed"))
        plot_columns = [
            (column, label, color)
            for column, label, color in plot_columns
            if column in top.columns and top[column].notna().any()
        ]
        if not plot_columns:
            return result

        positions = np.arange(len(top))
        width = min(0.8 / len(plot_columns), 0.24)
        fig, ax = plt.subplots(figsize=(12, 6))
        offsets = (np.arange(len(plot_columns)) - (len(plot_columns) - 1) / 2) * width
        for offset, (column, label, color) in zip(offsets, plot_columns):
            bars = ax.bar(positions + offset, top[column].fillna(np.nan), width, label=label, color=color)
            for bar, value in zip(bars, top[column]):
                if pd.isna(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(value),
                    f"{float(value):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(top["family"], rotation=20, ha="right")
        ax.set_ylabel("MAE (eV)")
        ax.set_title("Per-Family Prediction Error")
        ax.legend()
        y_max = float(np.nanmax([top[column].to_numpy(dtype=float) for column, _, _ in plot_columns]))
        ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "per_family_performance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return result


def learning_curve_analysis() -> list[dict[str, object]]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except Exception as exc:
        print(f"⚠️ Skipping learning curve analysis: {exc}")
        existing = read_json(RESULTS_DIR / "learning_curve.json", default=[]) or []
        if existing:
            print("   Keeping existing learning-curve results.")
        return existing

    catalog = pd.read_csv(RESULTS_DIR / "material_catalog.csv").set_index("jid")
    train = pd.read_csv(DATA_DIR / "train_id_prop.csv").rename(columns={"id": "jid"}).join(catalog, on="jid", rsuffix="_meta")
    test = pd.read_csv(DATA_DIR / "test_id_prop.csv").rename(columns={"id": "jid"}).join(catalog, on="jid", rsuffix="_meta")
    train["target"] = train["target"].astype(float)
    test["target"] = test["target"].astype(float)

    featurizer = ElementProperty.from_preset("magpie")
    # Keep this diagnostic lightweight; full-test evaluation is already covered
    # by `predictions.npz`, while this curve only needs trend points.
    train = train.head(5000).copy()
    test = test.head(min(len(test), 2000)).copy()
    x_train, train = _featurize_valid_formulas(train, featurizer, Composition)
    x_test, test = _featurize_valid_formulas(test, featurizer, Composition)
    if len(train) == 0 or len(test) == 0:
        print("⚠️ Skipping learning curve analysis: no valid formulas after filtering")
        return []

    y_train = train["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)

    points = []
    for size in (100, 500, 1000, 2500, 5000):
        if len(train) < size:
            continue
        subset_x = x_train[:size]
        subset_y = y_train[:size]

        rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(subset_x, subset_y)
        ridge = Ridge(alpha=1.0)
        ridge.fit(subset_x, subset_y)

        rf_mae = float(np.mean(np.abs(rf.predict(x_test) - y_test)))
        ridge_mae = float(np.mean(np.abs(ridge.predict(x_test) - y_test)))
        points.append({"train_size": size, "random_forest_mae_eV": rf_mae, "ridge_mae_eV": ridge_mae})

    if points:
        setup_matplotlib()
        import matplotlib.pyplot as plt

        curve = pd.DataFrame(points)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve["train_size"], curve["random_forest_mae_eV"], marker="o", label="Random Forest")
        ax.plot(curve["train_size"], curve["ridge_mae_eV"], marker="o", label="Ridge")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("MAE (eV)")
        ax.set_title("Learning Curve for Baseline Models")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return points


def _featurize_valid_formulas(frame: pd.DataFrame, featurizer, composition_cls) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    keep_indices = []
    for index, formula in frame["formula"].items():
        if pd.isna(formula) or str(formula).strip().lower() in {"", "nan"}:
            continue
        try:
            rows.append(featurizer.featurize(composition_cls(str(formula))))
            keep_indices.append(index)
        except Exception:
            continue
    return np.asarray(rows, dtype=float), frame.loc[keep_indices].reset_index(drop=True)


def run(quick: bool = False, device: str = "cpu") -> dict[str, object]:
    del quick, device
    ensure_directories()
    set_random_seeds()

    frame = load_prediction_frame()
    pretrained_metrics = plot_prediction_panel(frame, "alignn_pretrained", "ALIGNN (Pretrained)", "eval_pretrained.png")
    self_trained_metrics = plot_prediction_panel(
        frame, "alignn_self_trained", "ALIGNN (Self-Trained)", "eval_self_trained.png"
    )
    rf_metrics = plot_prediction_panel(frame, "random_forest", "Random Forest", "eval_random_forest.png")
    ridge_metrics = plot_prediction_panel(frame, "ridge", "Ridge Regression", "eval_ridge.png")

    plot_training_history()
    plot_method_comparison(
        {
            "alignn_self_trained": {"method": "ALIGNN (Self-Trained)", **self_trained_metrics}
            if self_trained_metrics
            else None,
            "random_forest": {"method": "Random Forest + Magpie", **rf_metrics} if rf_metrics else None,
            "ridge": {"method": "Ridge + Magpie", **ridge_metrics} if ridge_metrics else None,
        }
    )
    family_rows = per_family_evaluation(frame)
    curve_rows = learning_curve_analysis()

    evaluation = {
        "alignn_pretrained": pretrained_metrics,
        "random_forest": rf_metrics,
        "ridge": ridge_metrics,
        "alignn_self_trained": self_trained_metrics,
        "per_family": family_rows,
        "learning_curve": curve_rows,
    }
    write_json(RESULTS_DIR / "evaluation_report.json", evaluation)
    write_json(RESULTS_DIR / "learning_curve.json", curve_rows)
    print(f"💾 Saved evaluation report to {RESULTS_DIR / 'evaluation_report.json'}")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--quick", action="store_true", help="Reserved for pipeline compatibility")
    parser.add_argument("--device", default="cpu", help="Reserved for pipeline compatibility")
    args = parser.parse_args()

    try:
        run(quick=args.quick, device=args.device)
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
