#!/usr/bin/env python3
"""
Project pipeline entry point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import FIGURES_DIR, PROJECT_ROOT, REPORT_DIR, RESULTS_DIR, set_random_seeds, setup_matplotlib


def run_step(step_num: int, quick: bool, device: str) -> bool:
    step_map = {
        1: ("Download JARVIS dataset", _step_download),
        2: ("Explore and split data", _step_explore),
        3: ("Pretrained ALIGNN + baselines", _step_predict),
        4: ("Train ALIGNN", _step_train),
        5: ("Evaluate models", _step_evaluate),
        6: ("Generate figures", _step_visualize),
        7: ("Write report", _step_report),
    }
    if step_num not in step_map:
        raise ValueError(f"Unknown step number: {step_num}")

    label, handler = step_map[step_num]
    print(f"\n{'=' * 72}")
    print(f"STEP {step_num}: {label}")
    print(f"{'=' * 72}")

    start = time.time()
    try:
        handler(quick=quick, device=device)
        print(f"✅ Step {step_num} finished in {time.time() - start:.1f}s")
        return True
    except Exception as exc:
        print(f"❌ Step {step_num} failed after {time.time() - start:.1f}s: {exc}")
        return False


def _step_download(quick: bool, device: str) -> None:
    del quick, device
    from data_download import download_jarvis_dataset

    download_jarvis_dataset()


def _step_explore(quick: bool, device: str) -> None:
    del device
    from data_explore import run

    run(quick=quick)


def _step_predict(quick: bool, device: str) -> None:
    del device
    from predict import run

    run(quick=quick)


def _step_train(quick: bool, device: str) -> None:
    from train import run

    run(quick=quick, device=device, epochs=10 if quick else 50, batch_size=32, learning_rate=0.001)


def _step_evaluate(quick: bool, device: str) -> None:
    from evaluate import run

    run(quick=quick, device=device)


def _step_visualize(quick: bool, device: str) -> None:
    del quick, device
    from visualize import run

    run()


def metric_text(value: object, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _collect_report_context() -> dict[str, object]:
    dataset = json.loads((RESULTS_DIR / "dataset_stats.json").read_text()) if (RESULTS_DIR / "dataset_stats.json").exists() else {}
    benchmark = json.loads((RESULTS_DIR / "pretrain_benchmark.json").read_text()) if (RESULTS_DIR / "pretrain_benchmark.json").exists() else {}
    evaluation = json.loads((RESULTS_DIR / "evaluation_report.json").read_text()) if (RESULTS_DIR / "evaluation_report.json").exists() else {}
    training = json.loads((RESULTS_DIR / "training_history.json").read_text()) if (RESULTS_DIR / "training_history.json").exists() else {}
    return {
        "dataset": dataset,
        "benchmark": benchmark,
        "evaluation": evaluation,
        "training": training,
    }


def _figure_line(name: str, description: str) -> str:
    path = FIGURES_DIR / name
    if path.exists():
        return f"- `{name}`: {description}"
    return f"- `{name}`: not generated in this run"


def _write_report_files(context: dict[str, object]) -> None:
    dataset = context["dataset"]
    benchmark = context["benchmark"]
    evaluation = context["evaluation"]
    training = context["training"]

    pretrained = benchmark.get("alignn_pretrained", {})
    rf = benchmark.get("random_forest", {})
    ridge = benchmark.get("ridge", {})

    train_status = training.get("status", "not run")
    train_message = training.get("message", "No self-trained ALIGNN checkpoint was produced in this run.")
    train_final_mae = None
    if training.get("val_mae"):
        train_final_mae = training["val_mae"][-1]

    dataset_size = dataset.get("total_materials", "not available")
    train_size = dataset.get("splits", {}).get("train", "not available")
    val_size = dataset.get("splits", {}).get("val", "not available")
    test_size = dataset.get("splits", {}).get("test", "not available")
    family_rows = evaluation.get("per_family", []) or []
    top_family = family_rows[0]["family"] if family_rows else "not available"

    report = f"""# Predicting Band Gaps of Two-Dimensional Materials Using ALIGNN

**Author**: Xiaoyu Wang  
**Course**: INFO5000 — Introduction to Data Science  
**Date**: April 27, 2026

## Abstract

This project studies band-gap prediction from crystal structures using the Atomistic Line Graph Neural Network (ALIGNN), with classical composition-based regressors as fallbacks and baselines. The implemented pipeline downloads the JARVIS dataset, builds deterministic train/validation/test splits, generates exploratory figures, evaluates Random Forest and Ridge baselines on Magpie composition features, attempts pretrained ALIGNN inference, and prepares an optional self-training path for GPU execution. In the current run, the processed dataset contains {dataset_size} valid materials with split sizes of train={train_size}, validation={val_size}, and test={test_size}. The available baseline results are Random Forest MAE {metric_text(rf.get('MAE_eV'), suffix=' eV')}, Ridge MAE {metric_text(ridge.get('MAE_eV'), suffix=' eV')}, and pretrained ALIGNN MAE {metric_text(pretrained.get('MAE_eV'), suffix=' eV')}. The self-trained ALIGNN stage finished with status `{train_status}`, so the report emphasizes the robust baseline results and the prepared GPU training workflow. Overall, the codebase is now organized around reproducible phases that can be re-run locally in quick mode or on a remote CUDA machine for full ALIGNN training.

## 1. Introduction

Two-dimensional materials are attractive because their electronic properties depend strongly on atomic arrangement, composition, and local bonding geometry. The band gap is the central target in this project because it controls whether a material behaves as a metal, semiconductor, or insulator, and therefore determines suitability for electronic and optoelectronic devices.

Direct density functional theory calculations are accurate but expensive. This motivates machine learning models that map crystal structures to band gaps much faster. ALIGNN is a strong candidate because it combines the crystal graph with a line graph, allowing the model to encode both bond connectivity and bond-angle information.

## 2. Methodology

The pipeline follows the proposal structure. First, the JARVIS dataset is downloaded and filtered to valid materials with non-negative band-gap values and available structure files. A deterministic 80/10/10 split is then written to `data/train_id_prop.csv`, `data/val_id_prop.csv`, and `data/test_id_prop.csv`, with a 1000-sample quick subset generated for debugging and CPU-only testing.

For baselines, the project uses Magpie composition descriptors with two regressors: Random Forest and Ridge Regression. For deep learning, the code first attempts pretrained ALIGNN inference and then supports self-training through the ALIGNN command-line tooling on a prepared structure directory. The training stage is configured for `n_workers=0` and can be switched between `cpu` and `cuda`.

## 3. Dataset

The processed dataset statistics from this run are:

- Total valid materials: {dataset_size}
- Train / validation / test: {train_size} / {val_size} / {test_size}
- Mean band gap: {metric_text(dataset.get('bandgap_mean'), suffix=' eV')}
- Standard deviation: {metric_text(dataset.get('bandgap_std'), suffix=' eV')}
- Metals: {dataset.get('n_metal', 'not available')}
- Semiconductors: {dataset.get('n_semiconductor', 'not available')}
- Insulators: {dataset.get('n_insulator', 'not available')}

The top identified material family in the processed metadata is {top_family}. Family labels are heuristic and were derived from chemical formulas and material identifiers for downstream grouped evaluation.

## 4. Results

### 4.1 Data exploration

The exploratory analysis summarizes the overall band-gap distribution, the metal/semiconductor/insulator split, the relation between band gap and unit-cell size, and the most common material families. These results are saved in `figures/data_exploration.png`.

### 4.2 Baseline comparison

- Pretrained ALIGNN: MAE {metric_text(pretrained.get('MAE_eV'), suffix=' eV')}, RMSE {metric_text(pretrained.get('RMSE_eV'), suffix=' eV')}, R² {metric_text(pretrained.get('R2'))}
- Random Forest: MAE {metric_text(rf.get('MAE_eV'), suffix=' eV')}, RMSE {metric_text(rf.get('RMSE_eV'), suffix=' eV')}, R² {metric_text(rf.get('R2'))}
- Ridge Regression: MAE {metric_text(ridge.get('MAE_eV'), suffix=' eV')}, RMSE {metric_text(ridge.get('RMSE_eV'), suffix=' eV')}, R² {metric_text(ridge.get('R2'))}

The comparison figure is saved as `figures/method_comparison.png`.

### 4.3 ALIGNN training curves

The self-training stage recorded status `{train_status}`. The latest available validation MAE is {metric_text(train_final_mae, suffix=' eV')}. {train_message}

### 4.4 Prediction scatter plots

Prediction-vs-actual and residual plots are generated for each available model. The pretrained ALIGNN plot is stored as `figures/eval_pretrained.png`, while the classical baselines are stored as `figures/eval_random_forest.png` and `figures/eval_ridge.png`.

### 4.5 Error analysis

Per-family error analysis is written to `results/evaluation_report.json` and visualized in `figures/per_family_performance.png` when enough grouped data are available. Learning-curve results for the baseline models are saved to `results/learning_curve.json` and visualized in `figures/learning_curve.png`.

## 5. Discussion

The current pipeline prioritizes reliability. When heavy deep-learning dependencies are unavailable or ALIGNN training does not complete, the workflow still produces usable dataset statistics, exploratory analysis, baseline regressors, and report figures. This matches the project requirement that pre-trained inference and classical baselines must remain available as fallbacks.

The current limitation is that the strongest result, a fully self-trained ALIGNN model, still depends on a successful CUDA-enabled environment. The code now includes explicit dataset preparation, CLI training attempts, and GPU-targeted execution hooks, so the remaining work is mostly operational rather than structural.

## 6. Conclusion

This project now has a functional, reproducible pipeline for band-gap prediction experiments based on JARVIS materials data. The local CPU path covers downloading, splitting, exploratory analysis, pretrained inference attempts, Random Forest and Ridge baselines, evaluation plots, concept figures, and report generation. The remote GPU path is prepared for full ALIGNN training. The next step is to execute that training successfully on the AutoDL instance and compare the resulting MAE directly against the baseline models.

## Figures Produced

{_figure_line('data_exploration.png', 'Band-gap distribution and dataset overview')}
{_figure_line('alignn_overview.png', 'ALIGNN workflow diagram')}
{_figure_line('crystal_graph_demo.png', 'Crystal graph and line graph concept')}
{_figure_line('physics_context.png', 'Physics motivation and ML pipeline context')}
{_figure_line('eval_pretrained.png', 'Pretrained ALIGNN scatter and residual plots')}
{_figure_line('eval_self_trained.png', 'Self-trained ALIGNN scatter and residual plots')}
{_figure_line('training_history.png', 'Training and validation curves')}
{_figure_line('method_comparison.png', 'Model benchmark comparison')}
{_figure_line('learning_curve.png', 'Baseline learning curve')}
{_figure_line('per_family_performance.png', 'Grouped family-wise MAE comparison')}

## References

1. Xie, T., & Grossman, J. C. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.
2. Choudhary, K., et al. Atomistic Line Graph Neural Network for improved materials property predictions.
3. Ward, L., et al. A general-purpose machine learning framework for predicting properties of inorganic materials.
4. Butler, K. T., et al. Machine learning for molecular and materials science.
5. Meng, S., et al. Deep learning in two-dimensional materials research.

## Appendix

### Reproducibility

1. Install the dependencies listed in `requirements.txt`.
2. Run `python run_pipeline.py --quick` for the fast CPU-only debug path.
3. Run `python run_pipeline.py` for the full local pipeline.
4. For remote CUDA training, use the credentials in `results/gpu_connection.json`, then run `python src/train.py --device cuda --epochs 50 --batch_size 64`.
5. After downloading the GPU results, re-run `python src/evaluate.py` and `python src/visualize.py` to refresh the report assets.
"""

    slides = """# Slides Outline

1. Title slide
   Project title, author, course, date.
2. Motivation
   Why 2D materials matter and why band-gap prediction is useful.
3. Problem setting
   DFT accuracy versus high-throughput cost.
4. Dataset
   JARVIS source, filtering, and final split sizes.
5. ALIGNN architecture
   Crystal graph plus line graph.
6. Baseline models
   Random Forest and Ridge with Magpie descriptors.
7. Data exploration
   Distribution of band gaps and family composition.
8. Benchmark results
   MAE / RMSE / R² comparison across available models.
9. Error analysis
   Scatter plots, residuals, and per-family behavior.
10. Training status
   Quick-mode training path and GPU execution plan.
11. Discussion
   Reliability-first pipeline and current limitations.
12. Conclusion
   Main findings and next steps toward full CUDA ALIGNN training.
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text(report)
    (REPORT_DIR / "slides_outline.md").write_text(slides)


def _step_report(quick: bool, device: str) -> None:
    del quick, device
    context = _collect_report_context()
    _write_report_files(context)
    print(f"📝 Wrote report files to {REPORT_DIR}")


def run_remote_gpu_training_from_file(quick: bool) -> bool:
    credentials_path = RESULTS_DIR / "gpu_connection.json"
    if not credentials_path.exists():
        raise FileNotFoundError(f"Missing GPU credential file: {credentials_path}")

    creds = json.loads(credentials_path.read_text())
    required = {"host", "port", "user", "password"}
    missing = required - set(creds)
    if missing:
        raise KeyError(f"gpu_connection.json is missing required fields: {sorted(missing)}")

    epochs = "10" if quick else "50"
    host = creds["host"]
    port = str(creds["port"])
    user = creds.get("user", "root")
    password = creds["password"]

    remote_dir = "/root/dl_2d_bandgap"
    rsync_cmd = [
        "sshpass",
        "-p",
        password,
        "rsync",
        "-avz",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=data/raw",
        "-e",
        f"ssh -p {port} -o StrictHostKeyChecking=no",
        f"{PROJECT_ROOT}/",
        f"{user}@{host}:{remote_dir}/",
    ]
    subprocess.run(rsync_cmd, check=True)

    remote_script = (
        f"cd {remote_dir} && "
        "pip install torch --index-url https://download.pytorch.org/whl/cu121 && "
        "pip install -r requirements.txt && "
        f"python src/train.py --device cuda --epochs {epochs} --batch_size 64 && "
        "python src/evaluate.py --device cuda"
    )
    ssh_cmd = [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-p",
        port,
        "-o",
        "StrictHostKeyChecking=no",
        f"{user}@{host}",
        remote_script,
    ]
    subprocess.run(ssh_cmd, check=True)

    for directory in ("results", "figures"):
        scp_cmd = [
            "sshpass",
            "-p",
            password,
            "scp",
            "-P",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_dir}/{directory}/*",
            str(PROJECT_ROOT / directory),
        ]
        subprocess.run(scp_cmd, check=True)

    print("⚠️ Shut down the AutoDL GPU instance after verifying the downloaded results.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2D band-gap project pipeline")
    parser.add_argument("--quick", action="store_true", help="Use the 1000-sample quick path and 10 training epochs")
    parser.add_argument("--skip-to", type=int, default=1, help="Start execution from this step number")
    parser.add_argument("--stop-after", type=int, default=7, help="Stop execution after this step number")
    parser.add_argument("--skip-steps", nargs="*", type=int, default=[], help="Specific steps to skip")
    parser.add_argument("--device", default="cpu", help="Training device: cpu or cuda")
    parser.add_argument("--gpu-from-file", action="store_true", help="Run the remote GPU training workflow using results/gpu_connection.json")
    args = parser.parse_args()

    set_random_seeds()
    setup_matplotlib()

    print("=" * 72)
    print("DL for 2D Materials Band Gap Prediction")
    print(f"Quick mode: {args.quick}")
    print(f"Device: {args.device}")
    print(f"Start step: {args.skip_to}")
    print(f"Stop step: {args.stop_after}")
    print("=" * 72)

    if args.gpu_from_file:
        run_remote_gpu_training_from_file(quick=args.quick)
        return

    results: dict[int, bool] = {}
    for step in range(args.skip_to, args.stop_after + 1):
        if step in args.skip_steps:
            print(f"⏭️ Skipping step {step} by user request.")
            results[step] = True
            continue
        success = run_step(step, quick=args.quick, device=args.device)
        results[step] = success
        if not success and step in (1, 2, 3):
            break

    print(f"\n{'=' * 72}")
    print("Pipeline Summary")
    print(f"{'=' * 72}")
    for step, success in results.items():
        print(f"{'✅' if success else '❌'} Step {step}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Report:  {REPORT_DIR}")


if __name__ == "__main__":
    main()
