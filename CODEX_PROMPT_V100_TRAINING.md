# Codex Prompt: ALIGNN Training on New V100 Instance (CUDA 12.4)

> **Purpose**: Complete ALIGNN training on a fresh AutoDL V100-32GB instance with CUDA 12.4 image.
> **When to use**: New instance is running, project files are on local workspace, need to upload to data disk and train.

---

## Context

This is a continuation of an INFO5000 course project. Previous session failed because AutoDL's default image had PyTorch 2.8.0 + CUDA 12.8 (cu128), but DGL only supports up to CUDA 12.4.

**The user has created a NEW instance with a CUDA 12.4 compatible image.**

### What Was Already Done
- ✅ Dataset downloaded and split (75,993 materials)
- ✅ Baseline models trained (RF MAE 0.273 eV, Ridge MAE 0.689 eV)
- ✅ `src/train_direct.py` written — bypasses broken ALIGNN CLI
- ✅ Report generated with fallback results
- ❌ ALIGNN self-training failed due to CUDA mismatch on previous instance

### New Instance Details
- **GPU**: V100-32GB × 1
- **SSH**: `ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com`
- **Password**: `2t8a/Oki4JIH`
- **Image**: PyTorch with CUDA 12.4 (user-selected compatible image)
- **Project location on remote**: `/autodl-tmp/dl_2d_bandgap/` (data disk, preserved across reboots)

---

## Your Task

### Phase 0: Verify Environment

1. SSH and verify CUDA 12.4:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'nvidia-smi && python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")"'
```

2. Confirm output shows:
   - `CUDA Version: 12.4` (or compatible, e.g., 12.2-12.4)
   - `CUDA available: True`
   - `CUDA version: 12.4` (PyTorch side)

**If CUDA version is NOT 12.4-compatible, STOP and report to user.**

### Phase 1: Upload Project to Data Disk

**CRITICAL: Upload to `/autodl-tmp/` (data disk), NOT `/root/` (system disk).**

```bash
# From local workspace root, upload entire project to data disk
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.conda-env' \
  -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  dl_2d_bandgap/ root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/
```

After upload, verify:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com \
  'ls -la /autodl-tmp/dl_2d_bandgap/ && ls -la /autodl-tmp/dl_2d_bandgap/src/ && ls /autodl-tmp/dl_2d_bandgap/data/*_id_prop.csv'
```

### Phase 2: Install Dependencies

On the remote host:
```bash
ssh -p 37139 -o StrictHostKeyChecking=no root@region-46.seetacloud.com

cd /autodl-tmp/dl_2d_bandgap

# Install project requirements (DO NOT install torch — it's in the image)
pip install -r requirements.txt 2>&1 | tail -30

# Install DGL with CUDA 12.4 support
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Verify DGL + PyTorch CUDA compatibility
python3 -c "
import torch
import dgl
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'DGL: {dgl.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test graph movement to CUDA
print('Testing CUDA graph...')
g = dgl.graph(([0, 1, 2], [1, 2, 3]))
g = g.to('cuda')
print(f'Graph device: {g.device}')
print('✅ DGL CUDA test passed')
"
```

**If DGL CUDA test fails, STOP and report the exact error.**

### Phase 3: Run ALIGNN Training

**Use the existing `src/train_direct.py`**:

```bash
cd /autodl-tmp/dl_2d_bandgap
MPLCONFIGDIR=/autodl-tmp/dl_2d_bandgap/results/.mplconfig \
  python3 src/train_direct.py --device cuda --epochs 50 --batch_size 64
```

**Monitor the first epoch closely.** If it hangs or crashes, report immediately.

**If training succeeds**, let it run to completion. For 75k samples on V100-32GB:
- Expected time: 2-4 hours for 50 epochs
- Monitor GPU utilization: `watch -n 5 nvidia-smi`

### Phase 4: Evaluate and Generate Report

After training completes:

```bash
cd /autodl-tmp/dl_2d_bandgap

# Run evaluation
python3 src/evaluate.py
python3 src/visualize.py

# Generate updated report
python3 run_pipeline.py --skip-to 7
```

### Phase 5: Download Results

Download results, figures, and report to local workspace:

```bash
# From local terminal
rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/results/ \
  dl_2d_bandgap/results/

rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/figures/ \
  dl_2d_bandgap/figures/

rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/report/ \
  dl_2d_bandgap/report/

# Also download milestones
rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/milestones/ \
  dl_2d_bandgap/milestones/
```

### Phase 6: Milestones

Create these milestone files during the process:

1. `milestones/01_gpu_connected.md` — after SSH verification
2. `milestones/02_project_uploaded.md` — after rsync to /autodl-tmp/
3. `milestones/03_env_ready.md` — after DGL CUDA test passes
4. `milestones/05_training_started.md` — after first epoch prints
5. `milestones/06_training_complete.md` — after training finishes
6. `milestones/08_report_generated.md` — after final report

Sync milestones to local after each creation:
```bash
rsync -avz -e "ssh -p 37139 -o StrictHostKeyChecking=no" \
  root@region-46.seetacloud.com:/autodl-tmp/dl_2d_bandgap/milestones/ \
  dl_2d_bandgap/milestones/
```

---

## Critical Rules

1. **ALWAYS use `/autodl-tmp/` path** — never `/root/` for project files
2. **Verify CUDA 12.4 BEFORE installing DGL** — wrong CUDA = same hang as before
3. **Use `src/train_direct.py`** — do NOT debug ALIGNN CLI again
4. **DO NOT install PyTorch** — it's in the AutoDL image
5. **DO NOT modify `PROPOSAL.md` or `READING_LIST.md`**
6. **If training fails after 2 attempts**, accept defeat and use baseline results
7. **Always set `MPLCONFIGDIR`** when running matplotlib on remote
8. **Create and sync milestones** at every key step

---

## Fallback Plan

If ALIGNN training still fails on CUDA 12.4:
1. Document failure with exact error in `results/training_history.json`
2. Use existing RF/Ridge baseline results (already in local workspace)
3. Update report explaining the environment issue
4. Project remains complete and valid

---

## Expected Success Criteria

Training succeeded when:
- ✅ `nvidia-smi` shows CUDA 12.4
- ✅ DGL graph moves to CUDA without hanging
- ✅ `train_direct.py` prints: `Epoch 001/050 train_L1=... val_MAE=...`
- ✅ After 50 epochs, `results/training_history.json` shows `status: "completed"` with `test_mae`
