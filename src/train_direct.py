"""
Direct ALIGNN training path that bypasses the ALIGNN 2026 CLI wrappers.

The installed ALIGNN 2026.4.2 package has two practical issues on the
AutoDL image used for this project:
1. DGL tries to load a GraphBolt library for PyTorch 2.8 that is not shipped.
2. The public training wrapper only enters its training loop for atomwise model
   names, which is not the graphwise band-gap regression model needed here.

This script still uses ALIGNN's official graph construction, dataloaders, and
graphwise ALIGNN model, but owns the small regression training loop directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np

from utils import (
    DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    ensure_directories,
    set_random_seeds,
    write_json,
)

DIRECT_OUTPUT_DIR = RESULTS_DIR / "alignn_direct_run"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
SELF_TRAINED_PREDICTIONS_PATH = RESULTS_DIR / "self_trained_predictions.npz"


def install_dgl_graphbolt_stub() -> None:
    """Avoid DGL 2.1 import failure with PyTorch 2.8 on the AutoDL image."""
    sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))


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


def load_atoms_dict(jid: str) -> dict[str, object]:
    path = RAW_DATA_DIR / f"{jid}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing structure JSON: {path}")
    return json.loads(path.read_text())


def build_dataset_array(
    splits: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    dataset: list[dict[str, object]] = []
    counts = {name: len(records) for name, records in splits.items()}
    for split_name in ("train", "val", "test"):
        for record in splits[split_name]:
            jid = record["id"]
            dataset.append(
                {
                    "jid": jid,
                    "target": float(record["target"]),
                    "atoms": load_atoms_dict(jid),
                }
            )
    return dataset, counts


def prepare_loaders(
    dataset: list[dict[str, object]],
    counts: dict[str, int],
    output_dir: Path,
    batch_size: int,
    use_lmdb: bool,
):
    install_dgl_graphbolt_stub()
    from alignn.data import get_train_val_loaders

    return get_train_val_loaders(
        dataset="user_data",
        dataset_array=dataset,
        target="target",
        n_train=counts["train"],
        n_val=counts["val"],
        n_test=counts["test"],
        keep_data_order=True,
        batch_size=batch_size,
        workers=0,
        pin_memory=False,
        line_graph=True,
        split_seed=RANDOM_SEED,
        atom_features="cgcnn",
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        cutoff_extra=3.0,
        max_neighbors=12,
        id_tag="jid",
        output_features=1,
        output_dir=str(output_dir),
        filename=str(output_dir / "cache_"),
        use_lmdb=use_lmdb,
        dtype="float32",
    )


def make_model(hidden_features: int, alignn_layers: int, gcn_layers: int):
    install_dgl_graphbolt_stub()
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=hidden_features,
        output_features=1,
        link="identity",
        zero_inflated=False,
        classification=False,
        num_classes=2,
        extra_features=0,
    )
    return ALIGNN(config), config


def move_batch_to_device(batch, device):
    graph, line_graph, lattice, target = batch
    return (
        graph.to(device),
        line_graph.to(device),
        lattice.to(device),
        target.to(device).view(-1).float(),
    )


def predict_batch(model, batch, device):
    graph, line_graph, lattice, target = move_batch_to_device(batch, device)
    prediction = model([graph, line_graph, lattice]).view(-1)
    return prediction, target


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction, target = predict_batch(model, batch, device)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_size = int(target.numel())
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def evaluate_model(model, loader, criterion, device) -> tuple[float, float, list[float], list[float], list[str]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    targets: list[float] = []
    predictions: list[float] = []
    ids: list[str] = []

    dataset_ids = list(getattr(loader.dataset, "ids", []))
    cursor = 0
    with torch_no_grad():
        for batch in loader:
            prediction, target = predict_batch(model, batch, device)
            loss = criterion(prediction, target)
            pred_np = prediction.detach().cpu().numpy().reshape(-1)
            target_np = target.detach().cpu().numpy().reshape(-1)
            batch_size = int(target_np.size)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            predictions.extend(float(x) for x in pred_np)
            targets.extend(float(x) for x in target_np)
            ids.extend(dataset_ids[cursor : cursor + batch_size])
            cursor += batch_size

    mae = float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets)))) if targets else float("nan")
    return total_loss / max(total_items, 1), mae, targets, predictions, ids


class torch_no_grad:
    """Small local context manager to delay importing torch until needed."""

    def __enter__(self):
        import torch

        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        return self._context.__exit__(exc_type, exc, traceback)


def write_prediction_csv(path: Path, ids: Iterable[str], targets: Iterable[float], predictions: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target", "prediction"])
        writer.writerows(zip(ids, targets, predictions))


def run(
    quick: bool = False,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_features: int = 256,
    alignn_layers: int = 4,
    gcn_layers: int = 4,
    use_lmdb: bool = True,
    rebuild_cache: bool = False,
) -> dict[str, object]:
    ensure_directories()
    set_random_seeds()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    install_dgl_graphbolt_stub()
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but unavailable; using CPU.")
        device = "cpu"
    torch_device = torch.device(device)

    output_dir = DIRECT_OUTPUT_DIR / ("quick" if quick else "full")
    if rebuild_cache and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_training_splits(quick=quick)
    dataset, counts = build_dataset_array(splits)
    print(f"📦 Loaded dataset array: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    train_loader, val_loader, test_loader, _ = prepare_loaders(
        dataset=dataset,
        counts=counts,
        output_dir=output_dir,
        batch_size=batch_size,
        use_lmdb=use_lmdb,
    )

    model, model_config = make_model(
        hidden_features=hidden_features,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
    )
    model = model.to(torch_device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.3,
    )

    history = {
        "status": "running",
        "runner": "train_direct.py",
        "device": str(torch_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "quick": quick,
        "split_sizes": counts,
        "output_dir": str(output_dir),
        "model_config": model_config.model_dump(),
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }
    write_json(output_dir / "config.json", history)

    best_val_mae = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, torch_device)
        val_loss, val_mae, _, _, _ = evaluate_model(model, val_loader, criterion, torch_device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        saving_msg = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            shutil.copy2(output_dir / "best_model.pt", BEST_MODEL_PATH)
            saving_msg = " | saved best"

        print(
            f"🚂 Epoch {epoch:03d}/{epochs} "
            f"train_L1={train_loss:.5f} val_L1={val_loss:.5f} val_MAE={val_mae:.5f}{saving_msg}"
        )
        history["status"] = "running"
        history["best_val_mae"] = best_val_mae
        history["best_model_path"] = str(BEST_MODEL_PATH)
        write_json(TRAINING_HISTORY_PATH, history)

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    if (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=torch_device))

    test_loss, test_mae, targets, predictions, ids = evaluate_model(model, test_loader, criterion, torch_device)
    np.savez(
        SELF_TRAINED_PREDICTIONS_PATH,
        ids=np.asarray(ids, dtype=str),
        targets=np.asarray(targets, dtype=float),
        predictions=np.asarray(predictions, dtype=float),
    )
    write_prediction_csv(output_dir / "prediction_results_test_set.csv", ids, targets, predictions)

    history.update(
        {
            "status": "completed",
            "test_loss": test_loss,
            "test_mae": test_mae,
            "self_trained_predictions": str(SELF_TRAINED_PREDICTIONS_PATH),
        }
    )
    write_json(TRAINING_HISTORY_PATH, history)
    write_json(output_dir / "training_history.json", history)
    print(f"💾 Saved self-trained predictions to {SELF_TRAINED_PREDICTIONS_PATH}")
    print(f"✅ Direct ALIGNN test MAE: {test_mae:.5f} eV")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALIGNN directly without the ALIGNN CLI")
    parser.add_argument("--quick", action="store_true", help="Use the 1000-sample quick split")
    parser.add_argument("--device", default="cuda", help="Training device: cuda or cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--alignn_layers", type=int, default=4)
    parser.add_argument("--gcn_layers", type=int, default=4)
    parser.add_argument("--no_lmdb", action="store_true", help="Use in-memory graph dataset instead of LMDB")
    parser.add_argument("--rebuild_cache", action="store_true", help="Delete this run's cached ALIGNN graph data first")
    args = parser.parse_args()

    try:
        run(
            quick=args.quick,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_features=args.hidden_features,
            alignn_layers=args.alignn_layers,
            gcn_layers=args.gcn_layers,
            use_lmdb=not args.no_lmdb,
            rebuild_cache=args.rebuild_cache,
        )
    except Exception as exc:
        print(f"❌ Direct ALIGNN training failed: {exc}")
        raise


if __name__ == "__main__":
    main()
