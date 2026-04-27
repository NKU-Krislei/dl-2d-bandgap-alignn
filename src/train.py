"""
Step 4: Train ALIGNN, preferring the official CLI and falling back gracefully.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from utils import (
    DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    ensure_directories,
    set_random_seeds,
    write_json,
)

TRAINING_ROOT = DATA_DIR / "alignn_training_data"
CLI_OUTPUT_DIR = RESULTS_DIR / "alignn_cli_run"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"


def load_records(split_name: str) -> list[dict[str, str]]:
    split_path = DATA_DIR / f"{split_name}_id_prop.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open() as handle:
        return list(csv.DictReader(handle))


def load_training_splits(quick: bool) -> dict[str, list[dict[str, str]]]:
    prefix = "_small" if quick else ""
    return {
        "train": load_records(f"train{prefix}"),
        "val": load_records(f"val{prefix}"),
        "test": load_records(f"test{prefix}"),
    }


def write_poscar_from_json(source_json: Path, destination: Path) -> None:
    from jarvis.core.atoms import Atoms
    from jarvis.db.jsonutils import loadjson

    atoms = Atoms.from_dict(loadjson(str(source_json)))
    if hasattr(atoms, "write_poscar"):
        atoms.write_poscar(str(destination))
        return

    try:
        from jarvis.io.vasp.inputs import Poscar

        Poscar(atoms=atoms).write_file(str(destination))
        return
    except Exception as exc:
        raise RuntimeError(f"Unable to export POSCAR for {source_json.name}: {exc}") from exc


def prepare_alignn_dataset(quick: bool) -> tuple[Path, dict[str, int]]:
    try:
        from jarvis.core.atoms import Atoms  # noqa: F401
        from jarvis.db.jsonutils import loadjson  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"ALIGNN dataset preparation requires jarvis-tools: {exc}") from exc

    splits = load_training_splits(quick=quick)
    dataset_root = TRAINING_ROOT / ("quick" if quick else "full")
    dataset_root.mkdir(parents=True, exist_ok=True)
    id_prop_path = dataset_root / "id_prop.csv"

    rows: list[str] = []
    for split_name in ("train", "val", "test"):
        for record in splits[split_name]:
            jid = record["id"]
            target = record["target"]
            source_json = RAW_DATA_DIR / f"{jid}.json"
            dest_poscar = dataset_root / f"{jid}.vasp"
            if not dest_poscar.exists():
                write_poscar_from_json(source_json, dest_poscar)
            rows.append(f"{dest_poscar.name},{target}")

    id_prop_path.write_text("\n".join(rows) + "\n")
    counts = {name: len(items) for name, items in splits.items()}
    return dataset_root, counts


def build_config_file(
    dataset_root: Path,
    counts: dict[str, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> Path:
    config = {
        "dataset": "user_data",
        "target": "target",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "keep_data_order": True,
        "random_seed": RANDOM_SEED,
        "num_workers": 0,
        "progress": True,
        "output_dir": str(CLI_OUTPUT_DIR),
        "model": {
            "name": "alignn",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            "link": "identity",
            "zero_inflated": False,
            "classification": False,
            "num_classes": 2,
            "extra_features": 0,
        },
    }
    config_path = dataset_root / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def run_training_cli(command: list[str], cwd: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    output = "\n".join(filter(None, [completed.stdout, completed.stderr]))
    return completed.returncode == 0, output


def resolve_executable(name: str) -> str | None:
    direct = shutil.which(name)
    if direct:
        return direct
    sibling = Path(sys.executable).resolve().parent / name
    if sibling.exists():
        return str(sibling)
    return None


def parse_training_output(output: str) -> dict[str, list[float]]:
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    pattern = re.compile(
        r"epoch\s*[:=]?\s*(\d+).*?train.*?loss\s*[:=]\s*([0-9.]+).*?val.*?loss\s*[:=]\s*([0-9.]+).*?(?:mae\s*[:=]\s*([0-9.]+))?",
        re.IGNORECASE,
    )
    for line in output.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        history["train_loss"].append(float(match.group(2)))
        history["val_loss"].append(float(match.group(3)))
        if match.group(4) is not None:
            history["val_mae"].append(float(match.group(4)))
    return history


def capture_cli_artifacts(output_dir: Path) -> tuple[Path | None, dict[str, list[float]]]:
    best_model = None
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for candidate in output_dir.rglob("best_model.pt"):
        best_model = candidate
        break

    for candidate in output_dir.rglob("*.json"):
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and {"train_loss", "val_loss"} <= set(payload.keys()):
            history = payload
            break
    return best_model, history


def manual_training_fallback(reason: str) -> dict[str, object]:
    return {
        "status": "failed",
        "runner": "manual_fallback_unavailable",
        "message": reason,
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }


def run(
    quick: bool = True,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> dict[str, object]:
    ensure_directories()
    set_random_seeds()
    CLI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "status": "failed",
        "runner": None,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "quick": quick,
        "command_attempts": [],
    }

    try:
        dataset_root, counts = prepare_alignn_dataset(quick=quick)
        config_path = build_config_file(
            dataset_root=dataset_root,
            counts=counts,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
        summary["split_sizes"] = counts
    except Exception as exc:
        failure = manual_training_fallback(str(exc))
        summary.update(failure)
        write_json(TRAINING_HISTORY_PATH, summary)
        print(f"⚠️ Training setup failed: {exc}")
        return summary

    commands = [
        [
            "alignn_train_finetune",
            "--root_dir",
            str(dataset_root),
            "--config_name",
            str(config_path),
            "--output_dir",
            str(CLI_OUTPUT_DIR),
            "--device",
            device,
        ],
        [
            "train_alignn.py",
            "--root_dir",
            str(dataset_root),
            "--config_name",
            str(config_path),
            "--file_format",
            "poscar",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--target_key",
            "target",
            "--output_dir",
            str(CLI_OUTPUT_DIR),
            "--device",
            device,
        ],
    ]

    combined_output = ""
    for command in commands:
        executable = resolve_executable(command[0])
        summary["command_attempts"].append(" ".join(command))
        if executable is None:
            combined_output += f"\nCommand not found: {command[0]}"
            continue
        command[0] = executable
        ok, output = run_training_cli(command, cwd=Path.cwd())
        combined_output += f"\n$ {' '.join(command)}\n{output}\n"
        if ok:
            summary["status"] = "completed"
            summary["runner"] = Path(command[0]).name
            break

    parsed_history = parse_training_output(combined_output)
    best_model, history_from_artifacts = capture_cli_artifacts(CLI_OUTPUT_DIR)
    history = history_from_artifacts if any(history_from_artifacts.values()) else parsed_history

    if best_model is not None:
        shutil.copy2(best_model, BEST_MODEL_PATH)
        summary["best_model_path"] = str(BEST_MODEL_PATH)

    summary["train_loss"] = history.get("train_loss", [])
    summary["val_loss"] = history.get("val_loss", [])
    summary["val_mae"] = history.get("val_mae", [])
    summary["raw_log_path"] = str(CLI_OUTPUT_DIR / "train_stdout.log")
    (CLI_OUTPUT_DIR / "train_stdout.log").write_text(combined_output)

    if summary["status"] != "completed":
        fallback = manual_training_fallback(
            "ALIGNN CLI did not complete successfully. Use pretrained ALIGNN plus RF/Ridge baselines."
        )
        summary.update(
            {
                "runner": fallback["runner"],
                "message": fallback["message"],
            }
        )

    write_json(TRAINING_HISTORY_PATH, summary)
    print(f"💾 Saved training history to {TRAINING_HISTORY_PATH}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ALIGNN model")
    parser.add_argument("--quick", action="store_true", help="Use the quick 1000-sample split")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    try:
        run(
            quick=args.quick,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    except Exception as exc:
        print(f"❌ Training step failed: {exc}")
        raise


if __name__ == "__main__":
    main()
