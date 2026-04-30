<img width="57" height="34" alt="image" src="https://github.com/user-attachments/assets/ab067ffd-e09b-4162-a29e-15b86076c740" /><img width="496" height="35" alt="image" src="https://github.com/user-attachments/assets/1d679fd2-e3a1-48cd-9142-d227bb85ac76" /># Predicting Band Gaps of Two-Dimensional Materials with ALIGNN

INFO5000 course project for predicting band gaps of two-dimensional materials from crystal structures using graph neural networks and classical machine-learning baselines.

## Overview

Two-dimensional materials can show strongly structure-dependent electronic properties. Their band gap controls whether a material behaves as a metal, semiconductor, or insulator, and is central to electronic and optoelectronic applications. Direct density functional theory (DFT) calculations are accurate but expensive for high-throughput screening, so this project studies machine-learning surrogates that can predict band gaps from structural or compositional information.

The intended deep-learning model is ALIGNN (Atomistic Line Graph Neural Network), which represents crystals with both a crystal graph and a line graph to capture two-body bond interactions and three-body bond-angle information. The project also implements robust fallback baselines with Magpie descriptors, Random Forest, and Ridge regression.

## Project Context

- Course: INFO5000 - HKUST(GZ)
- Student: Junjie LEI, Cheng ZHANG, Haitao YU, Hongyu Zhan, Jiayi HUANG
- Research area: AI4Science
- Main task: Band gap regression for 2D materials
- Data source: JARVIS-DFT
- Primary model target: ALIGNN
- Reliability fallback: Random Forest and Ridge regression with Magpie descriptors

## Current Results

The completed reproducible baseline pipeline uses 75,993 valid JARVIS material records, split into 60,794 training, 7,599 validation, and 7,600 test samples.

| Method | MAE (eV) | RMSE (eV) | R2 |
| --- | ---: | ---: | ---: |
| Random Forest + Magpie | 0.273 | 0.611 | 0.798 |
| Ridge + Magpie | 0.689 | 1.034 | 0.421 |
| ALIGNN self-trained | 0.115 | 0.380 | 0.922 |

ALIGNN self-training was attempted on a remote GPU, but the run was blocked by DGL/PyTorch/CUDA compatibility issues in the available AutoDL image. The repository keeps the direct ALIGNN training implementation so it can be rerun when a matching PyTorch, CUDA, and DGL environment is available.

## Repository Structure

```text
dl_2d_bandgap/
├── run_pipeline.py          # Main sequential pipeline
├── setup_env.sh             # Environment setup helper
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_download.py     # Download JARVIS data
│   ├── data_explore.py      # Explore and preprocess data
│   ├── predict.py           # Pretrained/fallback predictions
│   ├── train.py             # ALIGNN training entry point
│   ├── train_direct.py      # Direct ALIGNN training loop
│   ├── evaluate.py          # Metrics and evaluation figures
│   ├── visualize.py         # Summary and concept figures
│   └── utils.py             # Shared utilities
├── results/                 # Metrics, summaries, predictions
├── figures/                 # Report figures
├── report/                  # Final report and slide outline
├── milestones/              # Execution milestones
└── PROPOSAL.md              # Full project proposal
```

## Quick Start

Create the environment:

```bash
bash setup_env.sh
```

Run the full pipeline:

```bash
python run_pipeline.py
```

Run individual steps:

```bash
python src/data_download.py
python src/data_explore.py
python src/predict.py
python src/evaluate.py
python src/visualize.py
```

## ALIGNN Training Notes

For GPU ALIGNN training, use an environment where PyTorch, CUDA, and DGL are version-compatible. The known working direction is to use an AutoDL image with CUDA 12.4 or CUDA 12.1 and install the matching DGL wheel.

Example command after environment repair:

```bash
python src/train_direct.py --device cuda --epochs 50 --batch_size 64
```

## Outputs

Key generated files include:

- `results/final_summary.json`
- `results/evaluation_report.json`
- `results/pretrain_benchmark.json`
- `results/predictions.npz`
- `figures/data_exploration.png`
- `figures/method_comparison.png`
- `figures/eval_random_forest.png`
- `figures/eval_ridge.png`
- `figures/learning_curve.png`
- `figures/per_family_performance.png`
- `report/report.md`

## Reproducibility

- Random seed is fixed at 42.
- Figures are generated headlessly with matplotlib's `Agg` backend.
- Data and results are stored separately from local environments and credentials.
- Local conda environments, GPU connection files, caches, checkpoints, and large raw data are excluded from Git.
