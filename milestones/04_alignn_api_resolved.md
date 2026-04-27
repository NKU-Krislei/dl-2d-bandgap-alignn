# Milestone: ALIGNN API Resolved, Fallback Chosen
**Created**: 2026-04-27 23:17
**Status**: COMPLETE

## What Was Just Done
Implemented `src/train_direct.py`, a direct graphwise ALIGNN training script that bypasses the broken ALIGNN 2026 CLI path.
The installed ALIGNN API was resolved: `alignn.train.train_dgl()` exists, but it only enters its training loop for model names containing `alignn_`, so a graphwise `"alignn"` model needs a custom loop.
Training still could not proceed reliably because DGL/PyTorch CUDA compatibility failed at runtime.

## Current Project State
- Phase: Phase 2G training debug completed with fallback
- Key files modified: `src/train_direct.py`, `src/predict.py`, `src/evaluate.py`, `results/training_history.json`
- Remote state: DGL was tested as CPU-only 2.1.0, CUDA 2.4.0+cu121, and CUDA 2.5.0+cu124
- Results: Self-training did not complete; RF/Ridge baselines are the completed quantitative deliverable, with pretrained ALIGNN unavailable in this run

## Next Immediate Steps
1. Use the documented RF/Ridge fallback results for evaluation and report.
2. Regenerate figures and report from baseline/pretrained artefacts.
3. Download remote results, figures, and report locally.

## Key Commands That Work
```bash
python3 src/evaluate.py
python3 src/visualize.py
python3 run_pipeline.py --skip-to 7
```

## Lessons Learned / Gotchas
- DGL 2.1.0 imported only after stubbing GraphBolt, but it was CPU-only and could not move graphs to CUDA.
- DGL 2.4.0+cu121 and 2.5.0+cu124 could move a toy graph to CUDA, but both hung during the first ALIGNN forward pass with PyTorch 2.8.0+cu128.
- Per the project fallback rule, continuing to force CUDA training is lower reliability than using the completed RF/Ridge baseline results.
