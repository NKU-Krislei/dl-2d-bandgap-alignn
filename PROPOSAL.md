# INFO5000 Course Project Proposal

## Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks: An ALIGNN-Based Study

---

**Course**: INFO5000 — Introduction to Data Science  
**Date**: April 26, 2026  
**Version**: 1.0  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Background & Motivation](#2-background--motivation)
3. [Literature Review](#3-literature-review)
4. [Research Objectives](#4-research-objectives)
5. [Methodology](#5-methodology)
6. [Dataset Design](#6-dataset-design)
7. [Experimental Design](#7-experimental-design)
8. [Technical Implementation Plan](#8-technical-implementation-plan)
9. [Expected Results & Success Criteria](#9-expected-results--success-criteria)
10. [Risk Assessment & Mitigation](#10-risk-assessment--mitigation)
11. [Timeline](#11-timeline)
12. [Deliverables](#12-deliverables)
13. [References](#13-references)

---

## 1. Abstract

Two-dimensional (2D) materials exhibit extraordinary electronic properties that depend sensitively on their atomic structure, making band gap prediction a central challenge in materials physics. Traditional density functional theory (DFT) calculations, while accurate, are computationally prohibitive for high-throughput screening of large material spaces. This project proposes to employ graph neural networks (GNNs), specifically the Atomistic Line Graph Neural Network (ALIGNN), to predict the band gaps of 2D materials directly from their crystal structures. ALIGNN represents materials as a crystal graph (encoding two-body atomic interactions) coupled with a line graph (encoding three-body bond-angle interactions), enabling it to learn physically meaningful structural features automatically. Using the JARVIS-DFT database (~53,000 materials), we will train and evaluate ALIGNN models with a focus on the 2D materials subset. The project will conduct systematic experiments including: (i) baseline comparison with classical GNN (CGCNN) and traditional machine learning (Random Forest with Magpie descriptors), (ii) data efficiency analysis, and (iii) interpretability assessment through attention visualization. This study aims to demonstrate that GNN-based approaches can achieve DFT-level accuracy for band gap prediction at a fraction of the computational cost, thereby accelerating the discovery and design of novel 2D materials for electronic applications.

---

## 2. Background & Motivation

### 2.1 Two-Dimensional Materials and Their Electronic Properties

Since the isolation of graphene in 2004 [1], two-dimensional (2D) materials — including transition metal dichalcogenides (TMDs, e.g., MoS₂, WS₂), hexagonal boron nitride (h-BN), black phosphorus, and MXenes — have emerged as a cornerstone of modern condensed matter physics and materials science. Their atomically thin geometry endows them with exceptional properties: high carrier mobility, tunable band gaps, mechanical flexibility, and suitability for next-generation electronic and optoelectronic devices [2].

The **band gap** (E_g) is arguably the single most important electronic property of a semiconductor material. It determines:
- Whether a material is a metal (E_g ≈ 0), semiconductor (0 < E_g ≲ 3 eV), or insulator (E_g > 3 eV)
- The wavelength range of light a material can absorb or emit
- The on/off current ratio in transistor applications
- The thermoelectric efficiency through the Seebeck coefficient

For 2D materials specifically, the band gap is highly sensitive to:
- **Number of layers**: e.g., MoS₂ transitions from indirect gap (1.29 eV, bulk) to direct gap (1.89 eV, monolayer)
- **Twist angle** in bilayer systems: moiré superlattice formation modifies the Brillouin zone folding and band reconstruction
- **Strain engineering**: external strain can modulate E_g by hundreds of meV

### 2.2 The Computational Bottleneck: DFT

Density functional theory (DFT) is the standard *ab initio* method for computing electronic structures. However, it faces fundamental scalability limitations:

| Factor | Impact |
|--------|--------|
| System size | Computational cost scales as O(N³) with the number of atoms |
| Single calculation | Minutes to hours per structure on modern clusters |
| High-throughput needs | Screening thousands of candidates is impractical |
| Twisted structures | Large supercells (100–1000+ atoms) become prohibitively expensive |

**The key insight**: If a machine learning model can learn the mapping **crystal structure → band gap** from existing DFT data, it could predict the band gap of a new structure in milliseconds rather than hours.

### 2.3 Why Graph Neural Networks?

Materials are naturally represented as **graphs**: atoms are nodes, chemical bonds are edges, and atomic properties (element type, electronegativity, valence) are node features. This is not an approximation — it is a faithful structural representation. GNNs are designed precisely for learning from graph-structured data through **message passing**, where each node aggregates information from its neighbors iteratively.

Compared to traditional ML approaches (e.g., CNN on images, ANN on hand-crafted descriptors), GNNs offer:
1. **Automatic feature extraction**: No need for manual descriptor engineering
2. **Rotation/translation invariance**: Physical symmetries are built into the graph representation
3. **Scalability**: Computationally efficient for variable-size crystal structures
4. **Interpretability**: Attention mechanisms can reveal which atoms/bonds contribute to predictions

### 2.4 Personal Research Connection

As a PhD student studying the electronic structure of **bilayer hexagonal boron nitride (h-BN)**, I am personally motivated by the challenge of predicting how twist angles and stacking configurations modify the electronic properties of 2D materials. While this course project focuses on the broader 2D materials band gap prediction task using public databases, the methods and insights gained will directly inform my ongoing doctoral research.

---

## 3. Literature Review

### 3.1 Early ML Approaches for Band Gap Prediction

| Work | Method | Data | Key Result |
|------|--------|------|------------|
| Ward et al. (2016) [3] | Random Forest + 145 crystal descriptors | ~4,000 materials | MAE ≈ 0.39 eV on MP formation energy |
| Xie & Grossman (2018) [4] | CGCNN (Crystal Graph CNN) | Materials Project | First GNN for crystals;开创性工作 |
| Nemnes et al. (2019) [5] | ANN | DFT on h-BN/graphene | R² = 0.998 on hybrid h-BN/graphene band gap |
| Dong et al. (2020) [6] | CNN/VGG16/ResNet | DFT on h-BN/graphene | < 10% relative error for 90%+ cases |

These early works demonstrated the feasibility of ML for band gap prediction but relied heavily on hand-crafted descriptors (Ward et al.) or image-based representations (Dong et al.) that lose structural information.

### 3.2 Graph Neural Networks for Materials

**CGCNN (Crystal Graph Convolutional Neural Network)** [4] — the foundational work:
- Input: Crystal structure → Crystal graph (atoms = nodes, bonds = edges)
- Architecture: Message passing neural network with convolutional update
- Benchmark: Outperformed random forest on multiple Materials Project tasks
- Limitation: Only captures two-body interactions (no explicit bond-angle information)

**ALIGNN (Atomistic Line Graph Neural Network)** [7] — the current state-of-the-art:
- **Key innovation**: Dual graph representation
  - **Crystal graph** G: atoms → nodes, bonds → edges (two-body interactions)
  - **Line graph** L(G): bonds → nodes, bond-bond connections → edges (three-body / bond-angle interactions)
- The line graph is constructed such that two edges sharing a common atom in G become connected nodes in L(G). This naturally encodes angular information without manual feature engineering.
- **Architecture**: Alternating message passing between G and L(G)
- **Performance**: Up to **85% improvement** over CGCNN across 52 material properties on the JARVIS-DFT dataset (~75,000 materials)
- **Open source**: Fully available via pip install, with pre-trained models and Colab tutorials

### 3.3 Reviews and Surveys

| Review | Year | Key Takeaway |
|--------|------|-------------|
| Meng et al., Front. Phys. [8] | 2024 | Comprehensive survey of DL in 2D materials (characterization, prediction, design); introduces CNN/GAN/U-net methods |
| Butler et al., Nature [9] | 2018 | Machine learning for molecular and materials science; early overview of property prediction |
| Choudhary et al., NPJ Comp Mat [7] | 2021 | ALIGNN original paper with benchmarks |
| Schmidt et al., Nature [10] | 2019 | ML in materials science: "Past, present, and future" |
| Merchant et al., Nature [11] | 2023 | Deep learning for materials design; foundation model perspective |

**Gap identified**: While Meng et al. [8] surveyed DL applications in 2D materials, the review focused predominantly on CNN/ANN-based approaches and did not cover modern GNN methods (CGCNN, ALIGNN) or foundation model approaches. Our project directly addresses this gap by applying the state-of-the-art GNN to 2D materials band gap prediction.

### 3.4 Positioning of This Work

This project is positioned as a **reproducible applied study** that:
1. Applies ALIGNN (SOTA GNN) to the task of 2D materials band gap prediction
2. Provides systematic baseline comparisons (CGCNN, Random Forest)
3. Analyzes data efficiency and model interpretability
4. Demonstrates the practical viability of GNN-based materials property prediction

It is **not** claiming to propose a new model architecture. Instead, it focuses on rigorous experimentation and clear analysis of an existing SOTA method on a physically motivated task.

---

## 4. Research Objectives

### Primary Objective

> Train and evaluate an ALIGNN model for predicting the band gaps of 2D materials from crystal structures, and demonstrate that it achieves near-DFT accuracy at orders-of-magnitude lower computational cost.

### Specific Research Questions

| # | Research Question | Corresponding Experiment |
|---|-------------------|--------------------------|
| RQ1 | How does ALIGNN compare with classical GNN (CGCNN) and traditional ML (RF) for 2D materials band gap prediction? | Baseline comparison (Section 7.1) |
| RQ2 | How much training data is needed to achieve reliable predictions? | Data efficiency analysis (Section 7.3) |
| RQ3 | Does the model perform consistently across different 2D material families (TMDs, h-BN, phosphorene, etc.)? | Family-wise evaluation (Section 7.4) |
| RQ4 | What structural features does the model learn to associate with band gap values? | Interpretability analysis (Section 7.5) |
| RQ5 | Can a pre-trained ALIGNN model (trained on all materials) generalize well to unseen 2D materials? | Transfer learning evaluation (Section 7.6) |

### Hypotheses

1. **H1**: ALIGNN will outperform CGCNN and Random Forest for 2D materials band gap prediction, because the line graph representation captures bond-angle information critical for electronic structure.
2. **H2**: Performance will saturate with a moderate amount of training data (~5,000–10,000 structures), consistent with the findings of Choudhary et al. [7].
3. **H3**: The model will show larger prediction errors for narrow-gap semiconductors (E_g < 0.5 eV) and metals (E_g ≈ 0), due to the fundamental difficulty of DFT in treating these systems.

---

## 5. Methodology

### 5.1 Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Crystal    │     │   Graph      │     │   ALIGNN    │     │  Band Gap    │
│  Structure  │────▶│  Encoding    │────▶│   Model     │────▶│  Prediction  │
│  (POSCAR)   │     │  (G + L(G))  │     │  (GNN)      │     │  (eV)        │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                    │                    │                    │
  Atomic positions   Atoms = Nodes       Message Passing     Regression Output
  + Element types   Bonds = Edges       (G ↔ L(G))         (scalar value)
                    Bond angles via
                    Line Graph
```

### 5.2 Graph Representation

#### 5.2.1 Crystal Graph G

The crystal graph G = (V, E) is constructed as follows:

- **Nodes V**: Each atom i is represented by a feature vector **h_i** ∈ ℝ^d, composed of:
  - **Elemental features** (92-dimensional Magpie descriptor set [3]): atomic number, group, row, period, electronegativity, covalent radius, valence electrons, first ionization energy, electron affinity, polarizability, etc.
  
- **Edges E**: An edge (i, j) exists between atoms i and j if their distance d_ij is within a cutoff radius r_c (typically the sum of covalent radii × 1.5 or similar criterion). Edge features include:
  - Bond distance d_ij
  - Relative coordinates

#### 5.2.2 Line Graph L(G)

The line graph L(G) = (V', E') is constructed from G:

- **Nodes V'**: Each edge in G becomes a node in L(G). If edge e_k connects atoms i and j in G, the corresponding node in L(G) has features:
  - Concatenation of atom features: [**h_i** ∥ **h_j**]
  - Distance: d_ij
  
- **Edges E'**: Two nodes e_k = (i,j) and e_l = (i,m) in L(G) are connected if they share a common atom i in G. This means:
  - The "bond angle" θ_jim is implicitly captured by the connectivity pattern
  - The message passing on L(G) propagates information between bonds that share an atom, encoding **three-body correlations**

**Key advantage**: Unlike CGCNN, which only performs message passing on G (two-body), ALIGNN alternates between G and L(G), enabling the model to learn from angular relationships without explicitly computing them as input features.

### 5.3 ALIGNN Architecture

The ALIGNN model consists of TALIGNN ALIGNN layers, each comprising:

1. **GNN layer on G**: Message passing on the crystal graph
   - Message: **m_ij** = φ_m(**h_i**, **h_j**, **e_ij**)
   - Update: **h_i** ← φ_u(**h_i**, Σ_j **m_ij**)

2. **GNN layer on L(G)**: Message passing on the line graph
   - Uses updated bond features from step 1
   - Propagates angular information across the structure

3. **Readout**: Global pooling (sum/mean) over all atom representations → Fully connected layers → **Band gap prediction** (scalar)

**Default hyperparameters** [7]:
- ALIGNN layers: 3
- GCN layers: 3
- Hidden dimension: 256
- Activation: ReLU / SiLU
- Output: Linear regression head

### 5.4 Training Procedure

```
Input: Training set D = {(G_i, y_i)} where G_i = crystal structure, y_i = DFT band gap

1. Initialize model parameters θ
2. For epoch = 1 to E:
     For each batch B = {(G_i, y_i)}:
       a. Encode each G_i into graph representation
       b. Forward pass: ŷ_i = ALIGNN(G_i; θ)
       c. Compute loss: L = (1/|B|) Σ (ŷ_i - y_i)²
       d. Backward pass: ∂L/∂θ via automatic differentiation
       e. Update: θ ← θ - α · AdamW(∂L/∂θ)
   Evaluate on validation set
   Save best model checkpoint
3. Evaluate saved model on test set
```

**Optimizer**: AdamW (weight decay = 0.01)  
**Learning rate**: 0.001 with cosine annealing schedule  
**Loss function**: Mean Squared Error (MSE)  
**Batch size**: 32–64 (memory permitting)  
**Epochs**: 100–300 (early stopping if validation loss plateaus)

---

## 6. Dataset Design

### 6.1 Primary Dataset: JARVIS-DFT

| Attribute | Value |
|-----------|-------|
| **Name** | JARVIS-DFT (Joint Automated Repository for Various Integrated Simulations) |
| **Maintainer** | NIST (National Institute of Standards and Technology, USA) |
| **Total materials** | ~53,000 (after quality filtering) |
| **Properties** | Band gap, formation energy, bulk modulus, shear modulus, etc. |
| **DFT functional** | OPT (optB88vdW) — van der Waals corrected |
| **Format** | POSCAR + JSON (ALIGNN-compatible) |
| **Access** | Public, no API key required |
| **URL** | https://jarvis.nist.gov |

### 6.2 2D Materials Subsets

JARVIS-DFT contains a mixture of 3D bulk materials and some 2D materials. We will use the following strategy:

**Strategy 1: Chemical formula filtering**  
Identify 2D material families by chemical formula patterns:
- TMDs: MX₂ (M = Mo, W, Ti, Ta, Nb; X = S, Se, Te)
- h-BN family: BxNy
- Phosphorene, Arsenene: P, As
- MXenes: Ti₃C₂Tx, etc.
- Graphene family: C

**Strategy 2: Dimensionality indicator**  
Use the `dimensionality` field in JARVIS metadata if available, or compute from crystal structure.

**Strategy 3: Thickness criterion**  
Materials with interlayer vacuum spacing > 10 Å (commonly used criterion for exfoliable 2D materials).

### 6.3 Data Splitting

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 80% | Model fitting |
| Validation | 10% | Hyperparameter tuning, early stopping |
| Test | 10% | Final evaluation (never seen during training) |

**Important**: Splitting is performed **randomly at the material level**, not by structure similarity, to avoid data leakage while maintaining a realistic evaluation setting.

### 6.4 Data Preprocessing

1. **Quality filtering**: Remove materials with negative or anomalous band gaps
2. **Outlier detection**: Remove materials with |E_g| > 10 eV (likely calculation artifacts)
3. **Standardization**: Normalize band gap targets (subtract mean, divide by std) for training; denormalize for evaluation
4. **Graph construction**: Use ALIGNN's built-in graph constructor with default cutoff radius

---

## 7. Experimental Design

### 7.1 Experiment 1: Baseline Comparison

**Objective**: Establish that ALIGNN outperforms simpler methods.

| Method | Type | Input | Expected MAE (eV) |
|--------|------|-------|-------------------|
| ALIGNN | GNN (this work) | Crystal graph + line graph | ~0.15–0.20 |
| CGCNN | GNN (baseline) | Crystal graph only | ~0.25–0.35 |
| Random Forest | Classical ML | Magpie descriptors (145 features) | ~0.35–0.45 |
| Ridge Regression | Linear model | Magpie descriptors | ~0.50–0.60 |

**Metrics**:
- Mean Absolute Error (MAE): MAE = (1/N) Σ |ŷ_i - y_i|
- Root Mean Square Error (RMSE): RMSE = √[(1/N) Σ (ŷ_i - y_i)²]
- Coefficient of Determination (R²): R² = 1 - Σ(ŷ_i - y_i)² / Σ(y_i - ȳ)²
- Pearson correlation coefficient (r) between predictions and targets

### 7.2 Experiment 2: Cross-Validation Study

**Objective**: Ensure results are not sensitive to the particular train/test split.

- Perform **5-fold cross-validation** on the full dataset
- Report mean ± standard deviation of MAE across folds
- Use identical hyperparameters for all folds

### 7.3 Experiment 3: Data Efficiency Analysis

**Objective**: Determine how much training data is needed.

- Train ALIGNN on **progressively larger subsets**: 100, 500, 1,000, 2,500, 5,000, 10,000, 25,000 structures
- Evaluate on the same held-out test set
- Plot: MAE vs. training set size (learning curve)
- Identify the "knee point" where adding more data yields diminishing returns

### 7.4 Experiment 4: Performance Across Material Families

**Objective**: Assess whether the model performs uniformly across different types of 2D materials.

- Group test set materials by family (TMDs, BN, C-based, etc.)
- Compute per-family MAE and R²
- Identify families where the model underperforms and discuss possible physical reasons

### 7.5 Experiment 5: Physical Interpretability

**Objective**: Understand what the model has learned.

1. **Attention weight visualization**: For selected 2D materials (e.g., monolayer MoS₂, h-BN, phosphorene), extract and visualize the attention weights from ALIGNN's message passing layers. This reveals which atoms/bonds the model considers most important for band gap prediction.

2. **Error analysis**: Systematically analyze the largest prediction errors:
   - Are errors correlated with specific structural features? (e.g., number of atoms, symmetry)
   - Are narrow-gap materials systematically harder to predict?
   - Are there specific element types that cause difficulties?

3. **Feature importance (baseline methods)**: For Random Forest and Ridge Regression, extract feature importance rankings to identify which Magpie descriptors are most predictive of band gap.

### 7.6 Experiment 6: Pre-trained Model Transfer

**Objective**: Evaluate the generalization capability of ALIGNN's pre-trained model.

- Use ALIGNN's official pre-trained model (trained on full JARVIS-DFT)
- Evaluate directly on the 2D materials test set without fine-tuning
- Compare with our self-trained model
- If time permits: Fine-tune the pre-trained model on the 2D subset and compare

---

## 8. Technical Implementation Plan

### 8.1 Software Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Programming Language | Python | 3.11 | Core implementation |
| Deep Learning Framework | PyTorch | 2.x | Model training & inference |
| GNN Framework | DGL (Deep Graph Library) | Latest | Graph operations |
| ALIGNN | alignn (pip package) | 2024.x | Model implementation |
| Materials Tools | pymatgen, ASE | Latest | Crystal structure handling |
| ML Baselines | scikit-learn | 1.x | RF, Ridge regression |
| Data Analysis | pandas, numpy | Latest | Data processing |
| Visualization | matplotlib, seaborn | Latest | Figures for report |
| Environment Management | conda | Latest | Dependency isolation |

### 8.2 Hardware Requirements

| Resource | Requirement | Justification |
|----------|-------------|---------------|
| CPU | Any modern multi-core | GNN training is CPU-friendly for this scale |
| RAM | ≥ 16 GB | Graph construction for large datasets |
| GPU | Optional (not required) | Would accelerate training but not essential |
| Storage | ≥ 5 GB | Dataset + model checkpoints |

**Note**: All experiments can be completed on a standard laptop (macOS Apple Silicon) without GPU. ALIGNN is optimized for efficient inference even on CPU.

### 8.3 Code Architecture

```
dl_2d_bandgap/
├── README.md                     # Project documentation
├── setup_env.sh                  # Environment setup script
├── src/
│   ├── 01_download_data.py       # Download JARVIS dataset
│   ├── 02_explore_data.py        # Data exploration & preprocessing
│   ├── 03_predict_premodel.py    # Pre-trained model predictions + RF baseline
│   ├── 04_train_model.py         # Train ALIGNN (and CGCNN)
│   ├── 05_evaluate.py            # Compute metrics, generate evaluation figures
│   ├── 06_visualize.py           # Concept figures, summary visualizations
│   └── utils.py                  # Shared utilities
├── data/                         # Raw and processed data
│   ├── jarvis_dft_3d/            # JARVIS structures (JSON + POSCAR)
│   ├── train_id_prop.csv         # Training set
│   ├── val_id_prop.csv           # Validation set
│   └── test_id_prop.csv          # Test set
├── results/                      # Model outputs and metrics (JSON)
├── figures/                      # All figures for the report (PNG, PDF)
└── report/                       # Final report and presentation
```

### 8.4 Reproducibility Measures

1. **Random seed fixed**: All random operations use `seed=42`
2. **Version pinning**: `requirements.txt` with exact package versions
3. **Data provenance**: All data downloaded from official sources with documented URLs
4. **One-click execution**: Scripts are designed to be run sequentially from a single entry point

---

## 9. Expected Results & Success Criteria

### 9.1 Expected Quantitative Results

Based on the ALIGNN literature [7] and analogous benchmarks:

| Metric | Target (All Materials) | Target (2D Subset) |
|--------|----------------------|-------------------|
| ALIGNN MAE | 0.15–0.20 eV | 0.18–0.25 eV |
| CGCNN MAE | 0.25–0.35 eV | 0.30–0.40 eV |
| RF MAE | 0.35–0.50 eV | 0.40–0.55 eV |
| ALIGNN R² | 0.90–0.95 | 0.85–0.92 |

**Rationale for higher 2D errors**: 2D materials may be underrepresented in the training set, and their electronic properties (especially for narrow-gap TMDs) are more challenging for DFT itself.

### 9.2 Success Criteria

| Criterion | Threshold | Justification |
|-----------|-----------|---------------|
| ALIGNN trains successfully | ✅ Loss converges | Minimum viable result |
| ALIGNN outperforms RF | MAE_ALIGNN < MAE_RF | Validates GNN approach |
| ALIGNN achieves MAE < 0.3 eV | On test set | DFT-typical accuracy range |
| Learning curve shows saturation | MAE stabilizes | Data sufficiency |
| Report is complete | All sections present | Course requirement |
| Code is reproducible | One-click run | Scientific standard |

**Minimum viable product (MVP)**: If ALIGNN training encounters technical difficulties, we can fall back to using ALIGNN's official pre-trained model for direct predictions, combined with the Random Forest baseline — this still yields a complete and meaningful study.

### 9.3 Figures Expected in the Report

1. **Data distribution**: Band gap histogram, material type pie chart, band gap vs. unit cell size
2. **ALIGNN architecture diagram**: Crystal graph + line graph + message passing flow
3. **Crystal graph example**: h-BN structure → crystal graph → line graph visualization
4. **Training curves**: Training/validation loss and MAE vs. epoch
5. **Prediction scatter plots**: Predicted vs. actual band gap (for each method)
6. **Residual analysis**: Residual distribution, error vs. band gap
7. **Method comparison bar chart**: MAE/RMSE/R² comparison across methods
8. **Learning curve**: MAE vs. training set size
9. **Per-family performance**: Grouped bar chart by material family
10. **Physics context**: Band structure schematic, 2D material band gap comparison

---

## 10. Risk Assessment & Mitigation

### 10.1 Risk Matrix

| # | Risk | Probability | Impact | Mitigation Strategy |
|---|------|-------------|--------|---------------------|
| R1 | ALIGNN installation fails (dependency conflicts) | Medium | High | Use conda isolation; fall back to Colab notebook |
| R2 | JARVIS dataset download fails (large file) | Low | High | Provide alternative: Materials Project API (matminer) |
| R3 | Training takes too long on CPU | Medium | Medium | Reduce dataset size; use pre-trained model as primary result |
| R4 | Model performance is poor (high MAE) | Low | Medium | Analyze failure modes; this is itself a valid finding |
| R5 | 2D materials subset is too small for meaningful analysis | Medium | Medium | Use all materials for training; evaluate 2D subset as analysis |
| R6 | CGCNN training code unavailable/broken | Low | Low | Replace with DenseGNN or skip GNN-GNN comparison |
| R7 | Insufficient time for all experiments | Medium | Medium | Prioritize: pre-trained model > self-trained > interpretability |

### 10.2 Fallback Plan

If the primary plan (training ALIGNN from scratch) encounters unresolvable technical issues:

**Fallback Level 1**: Use ALIGNN's pre-trained model for all predictions
- Still produces: scatter plots, error analysis, per-family evaluation
- Loses: training curves, data efficiency analysis, custom training

**Fallback Level 2**: Use Random Forest + Magpie descriptors only
- Still produces: baseline results, feature importance analysis
- Loses: GNN comparison, graph-based interpretability

Both fallback levels yield a complete, presentable project report.

---

## 11. Timeline

### Detailed 3-Day Execution Plan

| Day | Time Block | Task | Deliverable |
|-----|-----------|------|-------------|
| **Day 1** | Morning (2h) | Environment setup + data download | ALIGNN installed, JARVIS data downloaded |
| | Afternoon (3h) | Data exploration + preprocessing | Data distribution figures, train/val/test split |
| | Evening (2h) | Pre-trained model predictions | Scatter plot of pre-trained ALIGNN results |
| **Day 2** | Morning (3h) | ALIGNN self-training + RF baseline | Training curves, RF results |
| | Afternoon (3h) | Evaluation + all comparison figures | Method comparison bar chart, residual plots |
| | Evening (2h) | Additional analyses (learning curve, per-family) | Learning curve plot, per-family table |
| **Day 3** | Morning (4h) | Report writing (Sections 1–6) | Background, methodology, results |
| | Afternoon (3h) | Report writing (Sections 7–9) + figures | Discussion, conclusions, references |
| | Evening (2h) | Final review + presentation prep | Polished report, slides outline |

### Contingency Time

Each day has 1–2 hours of buffer time for debugging and unexpected issues.

---

## 12. Deliverables

### 12.1 Code Repository
- Complete, documented Python codebase (`dl_2d_bandgap/`)
- One-click setup and execution scripts
- README with usage instructions

### 12.2 Data
- Downloaded and preprocessed JARVIS dataset
- Train/val/test splits with documented statistics
- All evaluation results in JSON format

### 12.3 Figures (10+ publication-quality figures)
- Listed in Section 9.3

### 12.4 Project Report
- **Format**: PDF (LaTeX or Markdown → PDF)
- **Length**: 10–15 pages (body text)
- **Structure**: Abstract → Introduction → Methods → Results → Discussion → Conclusion → References
- **Language**: English

### 12.5 Presentation Slides
- **Format**: PDF slides (10–15 slides)
- **Structure**: Motivation → Methods → Results → Conclusions → Future Work
- **Language**: English

---

## 13. References

[1] K. S. Novoselov et al., "Electric field effect in atomically thin carbon films," *Science*, 306, 666–669 (2004).

[2] A. C. Ferrari et al., "Science and technology roadmap for graphene, related two-dimensional crystals, and hybrid systems," *Nanoscale*, 7, 4598–4810 (2015).

[3] L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, "A general-purpose machine learning force field for bulk and nanostructured phosphorus," *NPJ Computational Materials*, 2, 16028 (2016).

[4] T. Xie and J. C. Grossman, "Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties," *Physical Review Letters*, 120, 145301 (2018).

[5] G. B. Nemnes et al., "Band gap prediction of large hybrid graphene–hexagonal boron nitride systems using machine learning," *Computational Materials Science*, 168, 120–127 (2019).

[6] Y. Dong et al., "Predicting band gap of two-dimensional h-BN/graphene van der Waals heterostructures using convolutional neural network," *Journal of Physics: Condensed Matter*, 32, 115702 (2020).

[7] K. Choudhary, B. DeCost, et al., "Atomistic Line Graph Neural Network for improved materials property predictions," *NPJ Computational Materials*, 7, 185 (2021). DOI: 10.1038/s41524-021-00650-1

[8] X. Meng, C. Qin, et al., "Deep learning in two-dimensional materials: Characterization, prediction, and design," *Frontiers of Physics*, 19, 53601 (2024).

[9] K. T. Butler, D. W. Davies, H. Cartwright, O. Isayev, and A. Walsh, "Machine learning for molecular and materials science," *Nature*, 559, 547–555 (2018).

[10] J. Schmidt, M. R. G. Marques, S. Botti, and M. A. L. Marques, "Machine learning the structural and energetic properties of molecules and materials," *NPJ Computational Materials*, 7, 45 (2019).

[11] A. Merchant et al., "Scaling deep learning for materials discovery," *Nature*, 624, 80–85 (2023).

---

*End of Proposal*
