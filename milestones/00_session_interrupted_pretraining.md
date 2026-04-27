# Milestone: Session Interrupted — Pre-Training Debug
**Created**: 2026-04-27 22:34
**Status**: IN PROGRESS (Phase 2G — GPU Training)

## What Was Just Done
Previous Codex session spent extensive time debugging ALIGNN CLI compatibility issues on remote GPU. Four problems were identified:
1. **TrainingConfig schema drift** — 2026 ALIGNN package rejects fields the wrapper generated (`output_features`, `n_workers`, `device`). Fixed by querying valid fields dynamically.
2. **Atomwise vs Graphwise model** — CLI defaults to atomwise (force) model, not the graphwise (scalar) model needed for band gap prediction. Setting `model.name="alignn"` resolves config parsing but NOT runtime behavior.
3. **train_grad logic inconsistency** — Even with correct graphwise config, the CLI entrypoint still enters gradient/force training paths. **UNRESOLVED.**
4. **CLI command format** — `alignn_train_finetune` has different 2026 signature; `train_alignn.py` works but has the gradient issue.

The session was about to bypass the CLI entirely and call ALIGNN library API directly via a custom Python script.

## Current Project State
- **Phase**: 2G (GPU Training) — STUCK at config/training invocation
- **Key files modified**: `src/train.py` (wrapper patched for schema, needs further fix for graphwise)
- **Remote state**: GPU instance was shut down. New instance ready (port 48992).
- **Results**: No successful training run. `results/training_history.json` is stale (pre-patch failure).
- **Data**: `data/alignn_training_data/full/` exists on remote with POSCAR files + `id_prop.csv`

## Next Immediate Steps
1. SSH into new GPU instance, verify connectivity and GPU availability
2. Re-upload project and reinstall dependencies
3. **Bypass CLI** — write `src/train_direct.py` that calls ALIGNN library API directly:
   - Inspect `alignn.train_dgl.train_dgl()` signature and `alignn.data.get_train_val_loaders()`
   - Build config using `TrainingConfig` with graphwise `ALIGNNConfig`
   - Call the training function directly, skipping `train_alignn.py` CLI
4. Run one-epoch validation, then scale to full training
5. Download results and run evaluation pipeline

## Key Commands That Work
```bash
# SSH connection
ssh -p 48992 -o StrictHostKeyChecking=no root@connect.cqa1.seetacloud.com 'echo OK'

# Check GPU
ssh -p 48992 -o StrictHostKeyChecking=no root@connect.cqa1.seetacloud.com 'nvidia-smi'

# Upload project
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='data/raw' --exclude='.conda-env' \
  -e "ssh -p 48992 -o StrictHostKeyChecking=no" \
  dl_2d_bandgap/ root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/

# Required env var for matplotlib on remote
MPLCONFIGDIR=/root/dl_2d_bandgap/results/.mplconfig
```

## Lessons Learned / Gotchas
- **DO NOT use CLI** (`alignn_train_finetune` or `train_alignn.py`) — the train_grad logic is broken/inconsistent for graphwise configs
- **Call ALIGNN library API directly** — `alignn.train_dgl.train_dgl()` or equivalent
- `config.json` must use `model.name="alignn"` (graphwise), NOT `alignn_atomwise`
- `calculate_gradient` field behavior differs between config parsing and runtime CLI
- Always set `n_workers=0` in dataloaders
- Remote Python is at `/root/dl_2d_bandgap/.remote-env/bin/python`
- 2026 ALIGNN package: `pip install alignn` installs the latest wheel with breaking API changes
