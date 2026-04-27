# Milestone: Project Uploaded
**Created**: 2026-04-27 22:48
**Status**: COMPLETE

## What Was Just Done
Uploaded the local project directory to `/root/dl_2d_bandgap/` on the GPU instance with `rsync`.
The upload excluded `.git`, Python caches, `data/raw`, `.conda-env`, and `.playwright-cli`.

## Current Project State
- Phase: Phase 2G remote environment setup
- Key files modified: `milestones/02_project_uploaded.md`
- Remote state: Project files are present at `/root/dl_2d_bandgap/`
- Results: No completed training results locally

## Next Immediate Steps
1. Verify remote Python, PyTorch, CUDA, and package state.
2. Install missing dependencies from `requirements.txt` without reinstalling PyTorch.
3. Inspect the installed ALIGNN API and implement `src/train_direct.py`.

## Key Commands That Work
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='data/raw' --exclude='.conda-env' --exclude='.playwright-cli' -e 'ssh -p 48992 -o StrictHostKeyChecking=no' ./ root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/
```

## Lessons Learned / Gotchas
- Use interactive SSH/rsync because the local non-interactive askpass path is blocked.
- Keep syncing `milestones/` so future sessions can resume from the latest state.
