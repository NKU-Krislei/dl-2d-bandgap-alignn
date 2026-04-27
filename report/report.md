# Predicting Band Gaps of Two-Dimensional Materials Using ALIGNN

**Author**: Xiaoyu Wang  
**Course**: INFO5000 — Introduction to Data Science  
**Date**: April 27, 2026

## Abstract

This project studies band-gap prediction from crystal structures using the Atomistic Line Graph Neural Network (ALIGNN), with classical composition-based regressors as fallbacks and baselines. The implemented pipeline downloads the JARVIS dataset, builds deterministic train/validation/test splits, generates exploratory figures, evaluates Random Forest and Ridge baselines on Magpie composition features, attempts pretrained ALIGNN inference, and prepares an optional self-training path for GPU execution. In the current run, the processed dataset contains 75993 valid materials with split sizes of train=60794, validation=7599, and test=7600. The available baseline results are Random Forest MAE 0.273 eV, Ridge MAE 0.689 eV, and pretrained ALIGNN MAE not available. The self-trained ALIGNN stage finished with status `failed`, so the report emphasizes the robust baseline results and the prepared GPU training workflow. Overall, the codebase is now organized around reproducible phases that can be re-run locally in quick mode or on a remote CUDA machine for full ALIGNN training.

## 1. Introduction

Two-dimensional materials are attractive because their electronic properties depend strongly on atomic arrangement, composition, and local bonding geometry. The band gap is the central target in this project because it controls whether a material behaves as a metal, semiconductor, or insulator, and therefore determines suitability for electronic and optoelectronic devices.

Direct density functional theory calculations are accurate but expensive. This motivates machine learning models that map crystal structures to band gaps much faster. ALIGNN is a strong candidate because it combines the crystal graph with a line graph, allowing the model to encode both bond connectivity and bond-angle information.

## 2. Methodology

The pipeline follows the proposal structure. First, the JARVIS dataset is downloaded and filtered to valid materials with non-negative band-gap values and available structure files. A deterministic 80/10/10 split is then written to `data/train_id_prop.csv`, `data/val_id_prop.csv`, and `data/test_id_prop.csv`, with a 1000-sample quick subset generated for debugging and CPU-only testing.

For baselines, the project uses Magpie composition descriptors with two regressors: Random Forest and Ridge Regression. For deep learning, the code first attempts pretrained ALIGNN inference and then supports self-training through the ALIGNN command-line tooling on a prepared structure directory. The training stage is configured for `n_workers=0` and can be switched between `cpu` and `cuda`.

## 3. Dataset

The processed dataset statistics from this run are:

- Total valid materials: 75993
- Train / validation / test: 60794 / 7599 / 7600
- Mean band gap: 0.613 eV
- Standard deviation: 1.343 eV
- Metals: 53186
- Semiconductors: 16908
- Insulators: 5899

The top identified material family in the processed metadata is Phosphorene. Family labels are heuristic and were derived from chemical formulas and material identifiers for downstream grouped evaluation.

## 4. Results

### 4.1 Data exploration

The exploratory analysis summarizes the overall band-gap distribution, the metal/semiconductor/insulator split, the relation between band gap and unit-cell size, and the most common material families. These results are saved in `figures/data_exploration.png`.

### 4.2 Baseline comparison

- Pretrained ALIGNN: MAE not available, RMSE not available, R² not available
- Random Forest: MAE 0.273 eV, RMSE 0.611 eV, R² 0.798
- Ridge Regression: MAE 0.689 eV, RMSE 1.034 eV, R² 0.421

The comparison figure is saved as `figures/method_comparison.png`.

### 4.3 ALIGNN training curves

The self-training stage recorded status `failed`. The latest available validation MAE is not available. Self-training was attempted through a direct ALIGNN graphwise training loop. The ALIGNN 2026 CLI skipped graphwise training, DGL 2.1 was CPU-only, and CUDA DGL wheels 2.4.0+cu121 and 2.5.0+cu124 both hung inside the first ALIGNN forward pass with the installed PyTorch 2.8.0+cu128. The project therefore uses the completed RF/Ridge baseline results, with pretrained ALIGNN recorded as unavailable in this run.

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

- `data_exploration.png`: Band-gap distribution and dataset overview
- `alignn_overview.png`: ALIGNN workflow diagram
- `crystal_graph_demo.png`: Crystal graph and line graph concept
- `physics_context.png`: Physics motivation and ML pipeline context
- `eval_pretrained.png`: not generated in this run
- `eval_self_trained.png`: not generated in this run
- `training_history.png`: not generated in this run
- `method_comparison.png`: Model benchmark comparison
- `learning_curve.png`: Baseline learning curve
- `per_family_performance.png`: Grouped family-wise MAE comparison

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
