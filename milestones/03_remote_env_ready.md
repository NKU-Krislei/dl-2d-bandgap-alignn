# Milestone: Remote Environment Ready
**Created**: 2026-04-27 22:54
**Status**: COMPLETE

## What Was Just Done
Verified the remote Python environment. Python 3.12.3, PyTorch 2.8.0+cu128, CUDA, ALIGNN 2026.4.2, DGL, JARVIS, pymatgen, and scikit-learn are installed.
The remote already contains processed JARVIS data and split files from the prior session.

## Current Project State
- Phase: Phase 2G direct ALIGNN training implementation
- Key files modified: `src/train_direct.py`, `src/predict.py`, `milestones/03_remote_env_ready.md`
- Remote state: Project uploaded; source patches synced; data splits available on the GPU host
- Results: Prior baseline artefacts exist remotely, but no completed self-trained ALIGNN result yet

## Next Immediate Steps
1. Test `src/train_direct.py` on the quick split for one epoch.
2. If the direct training path works, run full CUDA training.
3. Download `results/` and `figures/`, then run evaluation/report generation.

## Key Commands That Work
```bash
python3 -m py_compile src/train_direct.py src/predict.py
python3 src/train_direct.py --quick --device cuda --epochs 1 --batch_size 64 --rebuild_cache
```

## Lessons Learned / Gotchas
- DGL 2.1 lacks `libgraphbolt_pytorch_2.8.0.so`; stubbing `dgl.graphbolt` before importing DGL/ALIGNN avoids the unused GraphBolt import failure.
- ALIGNN 2026.4.2 has `alignn.train.train_dgl`, not `alignn.train_dgl.train_dgl`.
- The installed `train_dgl()` only enters its training loop for model names containing `alignn_`, so graphwise `"alignn"` needs a direct training loop.
