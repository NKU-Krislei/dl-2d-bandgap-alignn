# ⚠️ DEPRECATED — DO NOT USE THIS FILE

> **This is the original project setup prompt. It has been superseded.**
> **Use `CODEX_PROMPT_V100_TRAINING.md` for the current task.**
> The project is already set up. Do not re-run Phase 0/1 setup from this file.

---

# Codex Task Prompt: INFO5000 Project v1 Execution (ARCHIVED)

> **How to use this prompt:**
> 1. `cd` into the `dl_2d_bandgap/` directory
> 2. Run `codex` and paste the entire content below as your task
> 3. Alternatively: `codex --full-auto` with this as the system prompt in AGENTS.md

---

## Task

You are tasked with building and executing a complete machine learning project for a course called INFO5000 (Introduction to Data Science) at HKUST(GZ). The project uses graph neural networks (GNNs) to predict the band gaps of materials from their crystal structures. Your job is to make the existing codebase fully functional, run all experiments, generate all figures, and produce a complete project report.

**READ `PROPOSAL.md` FIRST.** It contains the full project specification, methodology, experimental design, and evaluation criteria.

## Project Structure

```
├── PROPOSAL.md            ← Full project specification (READ THIS FIRST)
├── AGENTS.md              ← Project conventions, GPU SSH info, constraints
├── CODEX_PROMPT.md        ← You are here
├── requirements.txt       ← Python dependencies
├── setup_env.sh           ← Conda environment setup
├── run_pipeline.py        ← Main entry point (runs all steps sequentially)
├── src/
│   ├── utils.py           ← Shared config (paths, seeds)
│   ├── data_download.py   ← Download JARVIS dataset
│   ├── data_explore.py    ← Explore & preprocess
│   ├── predict.py         ← Pre-trained model + RF baseline
│   ├── train.py           ← Train ALIGNN
│   ├── evaluate.py        ← Metrics & evaluation plots
│   └── visualize.py       ← Summary figures
├── data/                  ← Downloaded data goes here
├── results/               ← Model outputs, metrics JSON
│   └── gpu_connection.json ← GPU SSH credentials
├── figures/               ← All output figures
└── report/                ← Final report
```

## What You Need To Do

### Phase 0: Environment & Infrastructure (MUST DO FIRST)

1. **Verify `requirements.txt`** exists with all dependencies:
   ```
   torch>=2.0
   alignn
   pymatgen
   ase
   matminer
   jarvis-tools
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   tqdm
   networkx
   ```

2. **Verify `src/utils.py`** exists with shared configuration:
   - Project root path (auto-detected from `__file__`)
   - `DATA_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `REPORT_DIR` paths
   - `RANDOM_SEED = 42`
   - A `setup_matplotlib()` function that sets Agg backend, font family, and `axes.unicode_minus = False`
   - A `set_random_seeds()` function for reproducibility

3. **Verify `run_pipeline.py`** exists at project root with:
   - Runs steps 1-6 sequentially with progress reporting
   - `--skip-to N` flag to resume from a specific step
   - `--quick` flag that uses small dataset (1000 samples) and 10 epochs
   - Each step wrapped in try/except with clear error messages

### Phase 1: Fix & Refactor Existing Scripts

The existing scripts in `src/` are numbered (`01_download_data.py`, etc.) and may have several issues. Create clean, properly-named versions:

4. **Refactor `01_download_data.py` → `src/data_download.py`**:
   - Keep the core download logic (JARVIS dataset from Figshare)
   - Add proper error handling for network failures
   - Add a checksum verification step

5. **Refactor `02_explore_data.py` → `src/data_explore.py`**:
   - Import `utils.py` for paths and config
   - Fix: `load_jarvis_data()` returns `(rows, header, struct_dir)` — header is `reader.fieldnames`, which is a list, not the first argument
   - Add a `prepare_small_dataset()` function that creates `train_small_id_prop.csv` with first 1000 samples
   - Ensure 80/10/10 split produces `train_id_prop.csv`, `val_id_prop.csv`, `test_id_prop.csv` in `data/` directory

6. **Refactor `03_predict_premodel.py` → `src/predict.py`**:
   - Fix the Random Forest baseline: it should train on `train_id_prop.csv` and evaluate on `test_id_prop.csv`
   - Add a Ridge Regression baseline (using same Magpie features)
   - For the pre-trained ALIGNN model: try `alignn.pretrained.get_alignn_ffdb_model()` first, fall back gracefully
   - Save all predictions to `results/predictions.npz`

7. **Refactor `04_train_model.py` → `src/train.py`**:
   - Import `utils.py`
   - Add robust fallback: primary via `alignn_train_finetune` CLI, secondary via manual loop
   - Default: 50 epochs, batch_size=32, lr=0.001, small dataset
   - Add `--device` argument (default: "cpu", for GPU: "cuda")
   - Save training history JSON and best model checkpoint

8. **Refactor `05_evaluate.py` → `src/evaluate.py`**:
   - Import `utils.py`
   - Add `per_family_evaluation()` — group materials by chemical family
   - Add `learning_curve_analysis()` — plot MAE vs training set size (100, 500, 1000, 2500, 5000)
   - Generate all evaluation figures to `figures/`

9. **Refactor `06_visualize.py` → `src/visualize.py`**:
   - Import `utils.py`
   - Keep the concept figures (overview, crystal graph demo, physics context)
   - Add `generate_all_figures()` that creates the complete figure set

### Phase 2L: Local Experiments (CPU) — SKIP LOCAL ENV SETUP

**⚠️ IMPORTANT**: Do NOT install PyTorch or other packages locally. The remote GPU instance already has PyTorch installed via its AutoDL image. Skip Phase 2L entirely and proceed directly to Phase 2G.

- Data exploration (CSV processing, statistics) can be done locally without PyTorch
- All heavy computation (ALIGNN training, baseline models) goes to the remote GPU
- If any local step is needed, use the system's existing Python and basic packages only

### Phase 2G: GPU Training via SSH

**GPU credentials are saved in `results/gpu_connection.json`.** Read that file to get host, port, and password.

**⚠️ The remote AutoDL instance already has PyTorch/CUDA installed via its image.** DO NOT run `pip install torch`. Only install Python packages that are not already present.

13. **Upload project to remote GPU**:
    ```bash
    # Read credentials
    HOST=$(python -c "import json; d=json.load(open('results/gpu_connection.json')); print(d['host'])")
    PORT=$(python -c "import json; d=json.load(open('results/gpu_connection.json')); print(d['port'])")
    PASS=$(python -c "import json; d=json.load(open('results/gpu_connection.json')); print(d['password'])")

    # Upload (exclude heavy/unnecessary files)
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='data/raw' \
      -e "sshpass -p $PASS ssh -p $PORT -o StrictHostKeyChecking=no" \
      dl_2d_bandgap/ root@$HOST:/root/dl_2d_bandgap/
    ```

14. **Install additional dependencies and train on GPU**:
    ```bash
    sshpass -p "$PASS" ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST << 'EOF'
      cd /root/dl_2d_bandgap
      # Only install packages NOT already in the AutoDL PyTorch image
      pip install -r requirements.txt 2>&1 | tail -20
      python src/train.py --device cuda --epochs 50 --batch_size 64
      python src/evaluate.py --device cuda
    EOF
    ```
    - The remote instance has **RTX PRO 6000 / 96 GB** — use `batch_size=64` or larger
    - If ALIGNN fails, try: `pip install dgl -f https://data.dgl.ai/wheels/repo.html && pip install alignn`

15. **Download results back to local**:
    ```bash
    sshpass -p "$PASS" scp -P $PORT -o StrictHostKeyChecking=no \
      root@$HOST:/root/dl_2d_bandgap/results/* results/
    sshpass -p "$PASS" scp -P $PORT -o StrictHostKeyChecking=no \
      root@$HOST:/root/dl_2d_bandgap/figures/* figures/
    ```

16. **REMIND THE USER to shut down the GPU instance on AutoDL** after results are downloaded. Charges accrue while running.

### Phase 3: Generate the Report

17. **Create `report/report.md`** — a complete project report with:
    - **Title page**: Project title, author, course, date
    - **Abstract** (~200 words): Summarize the problem, method, and key results
    - **1. Introduction**: 2D materials importance, DFT bottleneck, ML opportunity
    - **2. Methodology**: ALIGNN architecture, crystal graph + line graph representation, training procedure
    - **3. Dataset**: JARVIS-DFT description, filtering, train/val/test split
    - **4. Results**:
      - 4.1 Data exploration (band gap distribution figure)
      - 4.2 Baseline comparison (ALIGNN vs CGCNN vs RF vs Ridge — bar chart)
      - 4.3 ALIGNN training curves (loss + MAE vs epoch)
      - 4.4 Prediction scatter plots (predicted vs actual)
      - 4.5 Error analysis (residual distribution, error vs band gap)
      - 4.6 Data efficiency (learning curve if available)
      - 4.7 Per-family performance (grouped bar chart by material family)
    - **5. Discussion**: Why ALIGNN works, limitations, physical interpretation
    - **6. Conclusion**: Summary of findings, future work (DeepH, transfer learning to twisted bilayers)
    - **References**: All cited papers
    - **Appendix**: Code repository link, reproducibility instructions

    **IMPORTANT**: Fill in actual numbers from experiment results. Read JSON files in `results/` to get actual MAE, RMSE, R² values. Do NOT use placeholder values.

18. **Create presentation slides outline** in `report/slides_outline.md`:
    - 10-12 slides covering: motivation, ALIGNN architecture, dataset, results, conclusions

## Critical Rules

1. **NEVER install PyTorch locally** — the remote GPU already has it. All training runs on the remote instance.
2. **NEVER run destructive commands** on files outside the `dl_2d_bandgap/` directory
3. **NEVER modify `PROPOSAL.md` or `READING_LIST.md`** — they are reference documents
4. **NEVER install packages system-wide** — always use the conda environment
5. **NEVER use `pip install` with `sudo`**
6. **ALWAYS print progress** so the user can follow along
7. **ALWAYS wrap each step in try/except** — if one step fails, log the error and continue
8. **ALWAYS save intermediate results** — don't wait until the end
9. If ALIGNN installation fails, try: `pip install dgl -f https://data.dgl.ai/wheels/repo.html && pip install alignn`
10. Use `n_workers=0` for all dataloaders (macOS multiprocessing issues)
11. **NEVER log SSH passwords in console output** — read from `results/gpu_connection.json` programmatically
12. **REMIND the user to shut down AutoDL GPU instance** after training completes

## Expected Output Files

```
figures/
├── data_exploration.png          ← Band gap distribution, material type pie chart
├── alignn_overview.png           ← ALIGNN workflow diagram
├── crystal_graph_demo.png        ← h-BN → graph → line graph
├── physics_context.png           ← Band structure + ML pipeline
├── eval_pretrained.png           ← Pre-trained ALIGNN scatter + residuals
├── eval_self_trained.png         ← Self-trained ALIGNN scatter + residuals
├── training_history.png          ← Loss and MAE curves
├── method_comparison.png         ← Bar chart: ALIGNN vs RF vs Ridge
├── learning_curve.png            ← MAE vs training set size
└── per_family_performance.png    ← Grouped bar chart by material family

results/
├── dataset_stats.json            ← Data statistics
├── predictions.npz               ← All model predictions
├── pretrain_benchmark.json       ← Pre-trained model metrics
├── training_history.json         ← Training curves data
├── evaluation_report.json        ← Final evaluation metrics
└── gpu_connection.json           ← GPU SSH credentials (gitignored)

report/
├── report.md                     ← Complete project report
└── slides_outline.md             ← Presentation slides outline
```

## Quick Start Command

```bash
cd dl_2d_bandgap

# Main command — skip Phase 2L (local setup), go straight to GPU:
codex --full-auto "Read PROPOSAL.md and AGENTS.md first. Execute Phase 0, Phase 1, then Phase 2G (upload to GPU, train, download results), then Phase 3 (report). Do NOT install packages locally. All computation on remote GPU. Print progress at each step."
```
