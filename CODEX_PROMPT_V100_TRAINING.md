# Codex Prompt: ALIGNN Training on V100 Instance (CUDA 12.4)

> **Purpose**: Complete ALIGNN GPU training on a fresh AutoDL V100-32GB instance with CUDA 12.4.
> **How to use**: Open a NEW Codex session, `cd dl_2d_bandgap/`, paste the full content of this file as your task.
> **OVERRIDE**: Ignore any SSH credentials in `AGENTS.md` or `results/gpu_connection.json` — the NEW instance details are in this file.

---

## Project Background

**What this project is**: An INFO5000 (Introduction to Data Science, HKUST-GZ) course project that uses the ALIGNN graph neural network to predict band gaps of 2D materials from their crystal structures.

**Goal**: Train ALIGNN on the JARVIS-DFT 2D materials dataset (75,993 samples), evaluate against Random Forest baseline, and update the project report with ALIGNN metrics.

**Why we're here now**: The previous AutoDL instance had PyTorch 2.8.0 + CUDA 12.8 (cu128). DGL's latest wheel only supports up to CUDA 12.4 (cu124), causing all ALIGNN training to hang silently during CUDA kernel loading. The user has created a new instance with a CUDA 12.4-compatible image to resolve this.

---

## Current Project State

### Already Done (do not redo)
- ✅ All 75,993 JARVIS-DFT 2D materials downloaded to `data/jarvis_dft_3d/` (locally)
- ✅ Data split CSV files exist: `data/train_id_prop.csv` (60,794), `data/val_id_prop.csv` (7,599), `data/test_id_prop.csv` (7,600)
- ✅ Small split also exists: `data/train_small_id_prop.csv` (800), `data/val_small_id_prop.csv` (100), `data/test_small_id_prop.csv` (100)
- ✅ Baseline models done: Random Forest MAE=0.273 eV, R²=0.798; Ridge MAE=0.689 eV
- ✅ All `src/` modules written and tested
- ✅ `src/train_direct.py` — custom training loop that bypasses broken ALIGNN CLI
- ✅ Report skeleton exists at `report/report.md` (needs ALIGNN metrics filled in)
- ✅ Figures exist at `figures/` (need to add training curve figures)
- ❌ ALIGNN self-training: NEVER completed successfully due to CUDA mismatch

### Local Workspace
All project files are on your local machine at the current working directory (`dl_2d_bandgap/`). The full dataset is in `data/`.

---

## New GPU Instance

```
GPU:        V100-32GB × 1
CPU:        6 vCPU Intel Xeon Gold 6130 @ 2.10GHz
RAM:        25 GB
System disk: 30 GB  (will be wiped on image change)
Data disk:  50 GB   ← PROJECT FILES GO HERE
SSH:        ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com
Password:   2t8a/Oki4JIH
Image:      PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4
Target path on remote: /autodl-tmp/dl_2d_bandgap/   ← DATA DISK (preserved across reboots)
```

**CRITICAL PATH NOTE**: Always use `/autodl-tmp/` as the project root on remote, NEVER `/root/`. The `/root/` system disk is only 30 GB and is wiped on image changes.

**Disk budget**: Data disk is 50 GB free. Project data (~12 GB raw JARVIS files) + LMDB graph cache (~5-8 GB) + model checkpoints (~1 GB) = ~20 GB total. Should fit comfortably.

---

## Your Task (Execute in Order)

### Phase 0: Verify GPU Environment

SSH and verify the environment matches expectations:

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && echo "=== PyTorch ===" && python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")" && echo "=== System ===" && python3 --version && free -h | grep Mem'
```

Expected output (confirmed by user):
```
Tesla V100-SXM2-32GB, 32510 MiB, ...
PyTorch: 2.5.1
CUDA available: True
CUDA version: 12.4
Python 3.12.x
Mem: ~25G total
```

**STOP if CUDA is NOT 12.4 or PyTorch is NOT 2.5.x. Report the exact version.**

### Phase 1: Upload Project to Remote Data Disk

Upload the entire local project to `/autodl-tmp/` on remote:

```bash
# Run from the parent directory of dl_2d_bandgap/
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.conda-env' \
  --exclude='results/alignn_direct_run/' \
  -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  dl_2d_bandgap/ root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/
```

Verify the upload:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'echo "=== Project structure ===" && ls /autodl-tmp/dl_2d_bandgap/ && echo "=== Data CSVs ===" && ls /autodl-tmp/dl_2d_bandgap/data/*.csv | head -10 && echo "=== Raw data count ===" && ls /autodl-tmp/dl_2d_bandgap/data/jarvis_dft_3d/ | wc -l && echo "=== Source files ===" && ls /autodl-tmp/dl_2d_bandgap/src/'
```

Expected: ~75,993 files in `data/jarvis_dft_3d/`, all `src/*.py` present.

Create milestone:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cat > /autodl-tmp/dl_2d_bandgap/milestones/01_uploaded.md << '"'"'EOF'"'"'
# Milestone: Project Uploaded to V100
**Status**: COMPLETE
## What Was Done
Project files uploaded to /autodl-tmp/dl_2d_bandgap/ on new V100-32GB instance with CUDA 12.4.
## Next Steps
1. Install DGL cu12x
2. Verify ALIGNN imports
3. Run training
EOF'
```

### Phase 2: Install DGL (CUDA-Matched)

**Known environment**: PyTorch 2.5.1, CUDA 12.4, Python 3.12, Ubuntu 22.04.

Install DGL with the correct wheel for PyTorch 2.5 + CUDA 12.4:

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu124/repo.html 2>&1 | tail -20'
```

**If the above URL returns 404 or "No matching distribution"**, try these fallbacks in order:

```bash
# Fallback 1: generic repo (auto-selects version)
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Fallback 2: torch-2.4 wheel (compatible with PyTorch 2.x series)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Fallback 3: PyPI latest (may pull CPU version — check after install)
pip install dgl
```

**After install, verify DGL version and CUDA graph works (this was the exact failure point on the previous instance):**

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com 'python3 -c "
import sys, types
# Inject GraphBolt stub (needed for DGL 2.x + PyTorch 2.x compatibility)
sys.modules.setdefault(\"dgl.graphbolt\", types.ModuleType(\"dgl.graphbolt\"))
import dgl, torch
print(f\"DGL: {dgl.__version__}\")
print(f\"PyTorch: {torch.__version__}\")
print(f\"PyTorch CUDA: {torch.version.cuda}\")
# Move a graph to CUDA — this is the operation that hung silently on the previous instance
g = dgl.graph(([0,1,2],[1,2,3])).to(\"cuda\")
print(f\"Graph device: {g.device}\")
print(\"DGL CUDA test: PASSED\")
"'
```

**STOP if this test hangs or errors. Do NOT proceed to training. Report the exact error.**

Expected output:
```
DGL: 2.x.x
PyTorch: 2.5.1
PyTorch CUDA: 12.4
Graph device: cuda:0
DGL CUDA test: PASSED
```

### Phase 3: Install Other Dependencies

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com 'cd /autodl-tmp/dl_2d_bandgap && pip install -r requirements.txt 2>&1 | tail -30'
```

**Important notes for this environment:**

1. **DO NOT install PyTorch** — it's already in the AutoDL image (PyTorch 2.5.1). If `requirements.txt` triggers a torch reinstall, remove that line first:
   ```bash
   ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
     "sed -i '/^torch/d' /autodl-tmp/dl_2d_bandgap/requirements.txt && pip install -r /autodl-tmp/dl_2d_bandgap/requirements.txt 2>&1 | tail -30"
   ```

2. **Python 3.12 compatibility**: Most packages support 3.12 well, but if any package fails (e.g., `matminer` needing older numpy), install without it and note the failure. The critical packages are: `alignn`, `jarvis-tools`, `pymatgen`, `scikit-learn`, `dgl`.

3. **Check available disk space** before installing (data disk 50 GB):
   ```bash
   ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com 'df -h /autodl-tmp/'
   ```

Verify ALIGNN imports correctly:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com 'python3 -c "
import sys, types
sys.modules.setdefault(\"dgl.graphbolt\", types.ModuleType(\"dgl.graphbolt\"))
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.data import get_train_val_loaders
print(\"ALIGNN imports: OK\")
"'
```

Create milestone after environment is ready:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cat > /autodl-tmp/dl_2d_bandgap/milestones/02_env_ready.md << '"'"'EOF'"'"'
# Milestone: Environment Ready
**Status**: COMPLETE
## What Was Done
- DGL installed with CUDA-matched wheel
- ALIGNN imports verified
- DGL CUDA graph test passed
## Next Steps
Run: python3 src/train_direct.py --device cuda --epochs 50 --batch_size 64
EOF'
```

### Phase 4: Run ALIGNN Training

**Use `src/train_direct.py` — this script bypasses the broken ALIGNN CLI and owns the training loop directly.**

DO NOT use `src/train.py` (it wraps the broken CLI). DO NOT use `alignn_train` command.

**Hardware budget**: V100-32GB VRAM, 25GB RAM, 6 vCPU.
- `batch_size=64` is safe and recommended (uses ~6-8 GB VRAM)
- `batch_size=128` may work but risks OOM with large graphs
- `workers=0` is already set in the script (avoids multiprocessing issues with 6 vCPU)

**Quick test first** (strongly recommended — takes ~5 min, confirms everything works before the 3-4 hour full run):
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cd /autodl-tmp/dl_2d_bandgap && python3 src/train_direct.py --device cuda --epochs 5 --batch_size 64 --quick 2>&1'
```

If quick test prints `🚂 Epoch 001/005 train_L1=...` successfully, proceed to full training:

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cd /autodl-tmp/dl_2d_bandgap && python3 src/train_direct.py --device cuda --epochs 50 --batch_size 64 2>&1 | tee /autodl-tmp/training_log.txt'
```

**Watch for these checkpoints in the output:**
1. `📦 Loaded dataset array: train=60794, val=7599, test=7600` — data loaded OK
2. Graph construction / LMDB cache building (may take 10-20 min for 75k materials — this is normal)
3. `🚂 Epoch 001/050 train_L1=... val_L1=... val_MAE=...` — training started ✅
4. `✅ Direct ALIGNN test MAE: X.XXXXX eV` — training complete ✅

**If it hangs after data loading with no output for >5 minutes**, Ctrl+C and run with `--no_lmdb` flag:
```bash
python3 src/train_direct.py --device cuda --epochs 50 --batch_size 64 --no_lmdb
```

**Additional flags** (use if needed):
- `--rebuild_cache` — delete previous LMDB cache before building (if cache seems corrupted)
- `--no_lmdb` — use in-memory graphs (no disk cache, slower start but avoids LMDB issues)

### Phase 5: Post-Training Evaluation and Report

After training completes (look for `✅ Direct ALIGNN test MAE`):

```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cd /autodl-tmp/dl_2d_bandgap && python3 src/evaluate.py && python3 src/visualize.py'
```

Then update the report with actual ALIGNN metrics:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cd /autodl-tmp/dl_2d_bandgap && cat results/training_history.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"ALIGNN Test MAE: {d.get(\"test_mae\", \"N/A\")} eV\")"'
```

Read `results/training_history.json` to get the actual test MAE and R², then update the corresponding sections in `report/report.md` with the real numbers.

Create final milestone:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'cat > /autodl-tmp/dl_2d_bandgap/milestones/04_training_complete.md << '"'"'EOF'"'"'
# Milestone: Training Complete
**Status**: COMPLETE
## Achieved Metrics
See results/training_history.json for test_mae and val_mae.
EOF'
```

### Phase 6: Download Results to Local

```bash
# Run from PARENT directory of dl_2d_bandgap/
rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/results/ \
  dl_2d_bandgap/results/

rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/figures/ \
  dl_2d_bandgap/figures/

rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/report/ \
  dl_2d_bandgap/report/

rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/milestones/ \
  dl_2d_bandgap/milestones/
```

Also download the training log:
```bash
scp -P 37139 -o StrictHostKeyChecking=no \
  root@region-46.seetacloud.com:/autodl-tmp/training_log.txt \
  dl_2d_bandgap/results/training_log.txt
```

---

## Critical Rules

| Rule | Details |
|------|---------|
| **Always `/autodl-tmp/`** | Never use `/root/` for project files on remote |
| **Use `train_direct.py`** | DO NOT use `train.py` or `alignn_train` CLI |
| **Verify DGL CUDA first** | Test graph.to("cuda") BEFORE starting 4-hour training |
| **Quick test first** | Run `--quick --epochs 5` before full training |
| **No local PyTorch install** | All training is on the remote GPU |
| **2-attempt limit** | If training fails twice, stop and report (use RF baseline) |
| **DO NOT modify** | `PROPOSAL.md` or `READING_LIST.md` |
| **Remind user** | Tell user to shut down AutoDL instance after results are downloaded |

---

## Fallback Plan (If ALIGNN Still Fails)

If ALIGNN training fails on CUDA 12.4 after 2 attempts:

1. Document the failure in `results/training_history.json` with `"status": "failed"` and the exact error
2. Use existing RF/Ridge baseline results (already computed, in `results/evaluation_report.json`)
3. Update `report/report.md` to explain the CUDA environment issue and present RF results as primary
4. The project is still complete and scientifically valid

**Current baseline results (already available):**
- Random Forest: MAE=0.273 eV, R²=0.798
- Ridge Regression: MAE=0.689 eV

---

## Success Criteria

| Checkpoint | What to look for |
|------------|-----------------|
| Phase 0 ✅ | `CUDA available: True` and CUDA version 12.1-12.4 |
| Phase 2 ✅ | `DGL CUDA test: PASSED` (graph moves to GPU without hanging) |
| Phase 3 ✅ | `ALIGNN imports: OK` |
| Phase 4 ✅ | `🚂 Epoch 001/050 train_L1=...` prints within 30 min |
| Phase 4 ✅ | `✅ Direct ALIGNN test MAE: X.XXXXX eV` after all 50 epochs |
| Phase 5 ✅ | `report/report.md` updated with actual ALIGNN numbers |
| Phase 6 ✅ | Results downloaded, user reminded to shut down instance |

---

## Reference Files

| File | Notes |
|------|-------|
| `src/train_direct.py` | **Primary training script** — use this |
| `src/evaluate.py` | Evaluation metrics and plots |
| `src/visualize.py` | Summary figures |
| `src/utils.py` | Shared paths (auto-resolves from `__file__`, no path changes needed) |
| `report/report.md` | Existing report — fill in ALIGNN sections |
| `milestones/` | Progress snapshots — check here if resuming |
| `results/training_history.json` | Training progress (written after each epoch) |
