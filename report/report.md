# Predicting Band Gaps of Two-Dimensional Materials Using ALIGNN

**Author**: Xiaoyu Wang
**Course**: INFO5000 - Introduction to Data Science
**Date**: April 28, 2026

## Abstract

This project predicts material band gaps from crystal structures using the Atomistic Line Graph Neural Network (ALIGNN), with classical composition-based regressors as baseline models. The implemented workflow downloads the JARVIS dataset, builds deterministic train/validation/test splits, generates exploratory figures, evaluates Random Forest and Ridge Regression models on Magpie composition features, and trains a graphwise ALIGNN model on a CUDA-enabled V100 instance. In the final run, the processed dataset contains 75993 valid materials with split sizes of train=60794, validation=7599, and test=7600. The self-trained ALIGNN model reaches test MAE 0.115 eV, RMSE 0.380 eV, and R2 0.922, outperforming the Random Forest baseline with MAE 0.273 eV and the Ridge baseline with MAE 0.689 eV. These results support the project hypothesis that a graph neural network with bond-angle information can predict band gaps more accurately than composition-only baseline models.

## 1. Introduction

Two-dimensional materials are attractive because their electronic properties depend strongly on atomic arrangement, composition, and local bonding geometry. The band gap is the central target in this project because it controls whether a material behaves as a metal, semiconductor, or insulator, and therefore determines its suitability for electronic and optoelectronic devices.

Direct density functional theory calculations are useful but computationally expensive. This motivates machine learning models that map crystal structures to band gaps much faster. ALIGNN is a strong candidate because it combines a crystal graph with a line graph, allowing the model to encode both bond connectivity and bond-angle information.

## 2. Methodology

The pipeline follows the proposal structure. First, the JARVIS dataset is downloaded and filtered to valid materials with non-negative band-gap values and available structure files. A deterministic 80/10/10 split is then written to `data/train_id_prop.csv`, `data/val_id_prop.csv`, and `data/test_id_prop.csv`, with a 1000-sample quick subset generated for debugging and CPU-only testing.

For baselines, the project uses Magpie composition descriptors with two regressors: Random Forest and Ridge Regression. For deep learning, the final run uses a direct ALIGNN graphwise training loop to avoid incompatible command-line behavior while preserving the official ALIGNN graph construction, dataloader, and model implementation. The V100 training run uses CUDA, batch size 64, 50 epochs, LMDB graph caching, and deterministic split ordering.

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

### 4.2 Model comparison

- Pretrained ALIGNN: MAE not available, RMSE not available, R2 not available
- Self-trained ALIGNN: MAE 0.115 eV, RMSE 0.380 eV, R2 0.922
- Random Forest: MAE 0.273 eV, RMSE 0.611 eV, R2 0.798
- Ridge Regression: MAE 0.689 eV, RMSE 1.034 eV, R2 0.421

The comparison figure is saved as `figures/method_comparison.png`.

### 4.3 ALIGNN training curves

The self-trained ALIGNN stage completed all 50 epochs on the V100 GPU. The best validation MAE was 0.117 eV at epoch 48, and the final held-out test MAE was 0.115 eV. The final training loss was 0.040 eV, indicating that the model converged substantially beyond the classical baselines. The training history figure is saved as `figures/training_history.png`.

### 4.4 Prediction scatter plots

Prediction-vs-actual and residual plots are generated for each available model. The self-trained ALIGNN plot is saved as `figures/eval_self_trained.png`, while the classical baselines are saved as `figures/eval_random_forest.png` and `figures/eval_ridge.png`.

### 4.5 Error analysis

Per-family error analysis is written to `results/evaluation_report.json` and visualized in `figures/per_family_performance.png` when enough grouped data are available. Learning-curve results for the baseline models are saved to `results/learning_curve.json` and visualized in `figures/learning_curve.png`.

## 5. Discussion

The final results show a clear performance hierarchy. The self-trained ALIGNN model reduces MAE by about 58% relative to the Random Forest baseline, from 0.273 eV to 0.115 eV, and reaches R2 0.922 compared with Random Forest R2 0.798. This improvement is consistent with ALIGNN's physical motivation: bond angles and local graph structure contain structural signals that composition descriptors alone cannot fully represent.

The project still keeps classical baseline models as reliable fallbacks. This is useful because the deep learning stack depends on compatibility between CUDA, PyTorch, DGL, and ALIGNN. After installing a CUDA-compatible DGL wheel on the V100 instance, the direct training loop completed successfully and produced the strongest result.

## 6. Conclusion

This project now has a functional, reproducible pipeline for JARVIS-based band-gap prediction experiments. The local CPU path covers downloading, splitting, exploratory analysis, pretrained inference attempts, Random Forest and Ridge baselines, evaluation plots, concept figures, and report generation. The remote GPU path successfully trained ALIGNN and reached test MAE 0.115 eV, below the 0.25 eV target threshold and clearly better than the Random Forest baseline.

## Figures Produced

- `data_exploration.png`: Band-gap distribution and dataset overview
- `alignn_overview.png`: ALIGNN workflow diagram
- `crystal_graph_demo.png`: Crystal graph and line graph concept
- `physics_context.png`: Physics motivation and ML pipeline context
- `eval_pretrained.png`: not generated in this run
- `eval_self_trained.png`: Self-trained ALIGNN prediction and residual plots
- `training_history.png`: ALIGNN training and validation curves
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
4. For remote CUDA training, use the credentials in `results/gpu_connection.json`, then run `python src/train_direct.py --device cuda --epochs 50 --batch_size 64`.
5. After downloading the GPU results, re-run `python src/evaluate.py` and `python src/visualize.py` to refresh the report assets.
