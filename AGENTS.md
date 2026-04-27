# AGENTS.md — Project Instructions for AI Coding Agents

## Project Overview

**Project**: Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks (ALIGNN)
**Type**: Course project for INFO5000 — Introduction to Data Science, HKUST(GZ)
**Author**: Xiaoyu Wang (PhD student, Advanced Materials)
**Goal**: Train and evaluate an ALIGNN GNN model to predict band gaps from crystal structures, producing a complete project report with figures and results.

## Architecture

```
dl_2d_bandgap/
├── AGENTS.md              ← You are here
├── PROPOSAL.md            ← Full project proposal (13 sections, READ THIS FIRST)
├── READING_LIST.md        ← Recommended papers
├── CODEX_PROMPT.md        ← Codex execution prompt
├── requirements.txt       ← Python dependencies
├── setup_env.sh           ← Conda environment setup
├── run_pipeline.py        ← Main entry point (runs all steps sequentially)
├── src/
│   ├── utils.py           ← Shared utilities (paths, config, reproducibility)
│   ├── data_download.py   ← Step 1: Download JARVIS dataset
│   ├── data_explore.py    ← Step 2: Data exploration & preprocessing
│   ├── predict.py         ← Step 3: Pre-trained model predictions + RF baseline
│   ├── train.py           ← Step 4: Train ALIGNN model
│   ├── evaluate.py        ← Step 5: Compute metrics & evaluation figures
│   └── visualize.py       ← Step 6: Summary visualization & report figures
├── data/                  ← Raw & processed data (downloaded at runtime)
├── results/               ← Model outputs, metrics (JSON)
│   └── gpu_connection.json ← GPU SSH credentials (read by pipeline)
├── figures/               ← All publication-quality figures (PNG, 150+ DPI)
└── report/                ← Final project report (LaTeX/Markdown → PDF)
```

## Key Constraints

1. **No PyTorch experience** — The user is a physics researcher who knows Python but not deep learning. Code should be well-commented and educational.
2. **macOS Apple Silicon (local)** — No GPU available locally. CPU-only for development and testing.
3. **Cloud GPU (AutoDL)** — RTX PRO 6000 / 96 GB instance available for training. SSH credentials in `results/gpu_connection.json`.
4. **Time-critical** — 3-day deadline. Reliability > perfection.
5. **Reliability first** — The project must produce usable results. If ALIGNN training fails, fall back to pre-trained model + Random Forest baseline.
6. **English report** — All code comments can be Chinese, but the report, figures, and variable names should be in English.

## GPU Strategy

### Remote (GPU) — Primary Workflow
- **Do NOT install PyTorch locally** — the remote GPU instance already has PyTorch via its AutoDL image
- All computation (data processing, training, evaluation) goes to the remote GPU
- Read SSH credentials from `results/gpu_connection.json`
- Upload project, run training, download results
- **Always shut down** the AutoDL instance after training to avoid unnecessary charges

### SSH Connection (read from results/gpu_connection.json)
```json
{
  "host": "connect.cqa1.seetacloud.com",
  "port": 48992,
  "user": "root",
  "gpu": "RTX PRO 6000 / 96 GB"
}
```

```bash
# Connect
ssh -p 48992 root@connect.cqa1.seetacloud.com

# Upload project
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.playwright-cli' \
  -e "ssh -p 48992" dl_2d_bandgap/ root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/

# Download results
rsync -avz -e "ssh -p 48992" \
  root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/results/ results/
rsync -avz -e "ssh -p 48992" \
  root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/figures/ figures/
```

## Technology Stack

- **Python 3.11**, **PyTorch 2.x** (CUDA on GPU, CPU locally), **ALIGNN** (pip package, includes DGL)
- **Materials tools**: pymatgen, ASE, matminer, jarvis-tools
- **ML baselines**: scikit-learn (Random Forest, Ridge Regression)
- **Visualization**: matplotlib, seaborn (Agg backend, no GUI)
- **Data source**: JARVIS-DFT (~53,000 materials from NIST, public, no API key)

## Known Issues (DO NOT FIX — already handled in prompt)

- `run_quick.py` has broken imports (old file naming). This file is DEPRECATED — use `run_pipeline.py` instead.
- Old numbered scripts (`01_download_data.py`, etc.) still exist for reference. The new properly-named modules in `src/` are the canonical versions.

## Coding Standards

- Use `matplotlib.use("Agg")` before any pyplot import (headless environment)
- Set `RANDOM_SEED = 42` everywhere for reproducibility
- Save all figures at `dpi=150, bbox_inches="tight"`
- Use `Path` from `pathlib` (not `os.path`) for file paths
- Print progress with emoji prefixes for readability
- All scripts must be runnable standalone: `python src/train.py`

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| ALIGNN MAE | < 0.25 eV (on test set) |
| ALIGNN R² | > 0.85 |
| ALIGNN must outperform RF baseline | MAE_ALIGNN < MAE_RF |
