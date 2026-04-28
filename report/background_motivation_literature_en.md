# Slides Background & Motivation Literature Guide

**Project:** Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks
**Purpose:** This note selects the most important papers for the background and motivation part of the slides, explains why each paper matters, and gives a coherent story line for the presentation.
**Recommended priority:** If time is limited, read papers 2, 3, 5, and 7 first. If you need stronger Q&A preparation, also read papers 1, 4, 6, and 8.

---

## 1. One-Sentence Story

Two-dimensional materials have technologically important but highly structure-sensitive electronic properties; DFT can compute band gaps but is too expensive for large-scale screening; machine learning can learn fast surrogate models from existing DFT databases; graph neural networks are especially suitable because crystal structures are naturally graphs; ALIGNN further improves crystal GNNs by explicitly encoding bond-angle information through line graphs.

---

## 2. Core Reading List

### 1. Machine Learning for Molecular and Materials Science

**Citation:** Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. *Nature* 559, 547-555 (2018).
**Link:** https://www.nature.com/articles/s41586-018-0337-2
**DOI:** https://doi.org/10.1038/s41586-018-0337-2

**Brief summary:**
This is a broad review of machine learning in chemistry and materials science. It explains how ML can accelerate property prediction, materials discovery, characterization, and design.

**Why it matters for this project:**
This paper supports the high-level motivation: ML is not meant to replace physics-based methods, but to learn fast surrogate models from expensive simulations and experiments. It gives a strong opening context for why data-driven materials science is useful.

**How to use it in slides:**
Use it in the first background slide to say that materials discovery is moving from one-by-one first-principles calculations toward high-throughput data plus machine learning.

---

### 2. High-Throughput Identification and Characterization of Two-Dimensional Materials Using Density Functional Theory

**Citation:** Choudhary, K., & Garrity, K. F. *Scientific Reports* 7, 5179 (2017).
**Link:** https://www.nature.com/articles/s41598-017-05402-0
**DOI:** https://doi.org/10.1038/s41598-017-05402-0

**Brief summary:**
This paper uses high-throughput DFT to identify and characterize two-dimensional materials. It is a concrete example of how large computational datasets for 2D materials are built.

**Why it matters for this project:**
It directly supports the materials background of the project. It shows that 2D materials form a large and meaningful search space, but systematic DFT screening is computationally demanding.

**How to use it in slides:**
Use it to justify the focus on 2D materials: they are scientifically important, technologically relevant, and suitable for high-throughput screening assisted by ML.

---

### 3. The Joint Automated Repository for Various Integrated Simulations (JARVIS) for Data-Driven Materials Design

**Citation:** Choudhary, K. et al. *npj Computational Materials* 6, 173 (2020).
**Link:** https://www.nature.com/articles/s41524-020-00440-1
**DOI:** https://doi.org/10.1038/s41524-020-00440-1

**Brief summary:**
This paper introduces the JARVIS infrastructure, including JARVIS-DFT, JARVIS-ML, and JARVIS-tools. JARVIS-DFT provides large-scale DFT-computed structures and properties for materials informatics.

**Why it matters for this project:**
This is the primary citation for the dataset used in the project. It also explains why public DFT databases make supervised learning for materials property prediction possible.

**How to use it in slides:**
Use it in the dataset slide: "We use JARVIS-DFT, a public NIST database containing DFT-computed materials structures and properties."

---

### 4. Bandgap Prediction of Two-Dimensional Materials Using Machine Learning

**Citation:** Zhang, Y. et al. *PLOS ONE* 16, e0255637 (2021).
**Link:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255637
**DOI:** https://doi.org/10.1371/journal.pone.0255637

**Brief summary:**
This paper predicts band gaps of 2D materials using classical machine learning methods, including SVR, MLP, GBDT, and Random Forest.

**Why it matters for this project:**
It is one of the most task-aligned papers: the material class is 2D materials and the target property is band gap. It also supports the use of Random Forest as a reasonable baseline.

**How to use it in slides:**
Use it to introduce the gap between classical ML and graph-based ML: previous work shows that ML can predict 2D band gaps, but many methods rely on hand-crafted descriptors.

---

### 5. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties

**Citation:** Xie, T., & Grossman, J. C. *Physical Review Letters* 120, 145301 (2018).
**Link:** https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
**DOI:** https://doi.org/10.1103/PhysRevLett.120.145301

**Brief summary:**
This is the foundational crystal graph neural network paper. It represents a crystal as a graph, where atoms are nodes and neighboring atom pairs are edges, then applies graph convolution to predict material properties.

**Why it matters for this project:**
CGCNN is the key bridge from descriptor-based ML to structure-based graph learning for crystals. It is also the conceptual predecessor of ALIGNN.

**How to use it in slides:**
Use it to explain why crystals are naturally suited for GNNs: atoms form nodes, bonds or neighbor relations form edges, and message passing learns local chemical environments.

---

### 6. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals

**Citation:** Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. *Chemistry of Materials* 31, 3564-3572 (2019).
**Link:** https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
**DOI:** https://doi.org/10.1021/acs.chemmater.9b01294

**Brief summary:**
This paper introduces MEGNet, a graph network framework for molecules and crystals. It shows that graph networks can learn useful atomistic representations and can transfer learned chemical information across related prediction tasks.

**Why it matters for this project:**
It helps position ALIGNN within the broader development of graph neural networks for materials. It also shows that GNNs are not a single special-purpose model, but a general framework for atomistic property prediction.

**How to use it in slides:**
Mention it briefly after CGCNN: CGCNN introduced crystal graph learning, while MEGNet demonstrated a more general graph-network framework for molecules and crystals.

---

### 7. Atomistic Line Graph Neural Network for Improved Materials Property Predictions

**Citation:** Choudhary, K., DeCost, B. et al. *npj Computational Materials* 7, 185 (2021).
**Link:** https://www.nature.com/articles/s41524-021-00650-1
**DOI:** https://doi.org/10.1038/s41524-021-00650-1

**Brief summary:**
This is the core ALIGNN paper. ALIGNN performs message passing on both the atomistic bond graph and its line graph. In the original graph, atoms are nodes and bonds are edges. In the line graph, bonds become nodes, and pairs of bonds sharing an atom are connected, which allows the model to encode bond-angle relationships.

**Why it matters for this project:**
This is the central method paper. The physical motivation is that many material properties depend not only on pairwise atomic distances but also on local geometry and bond angles. Since band gap is a structure-sensitive electronic property, ALIGNN is a physically motivated model choice.

**How to use it in slides:**
This should be the main method-motivation slide. The story can be:

1. CGCNN learns atom-neighbor interactions.
2. However, band gaps can be sensitive to local geometry and bond angles.
3. ALIGNN adds a line graph so the model can learn bond-angle relationships.
4. Therefore, ALIGNN is better suited for structure-sensitive materials property prediction.

---

### 8. Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm

**Citation:** Dunn, A. et al. *npj Computational Materials* 6, 138 (2020).
**Link:** https://www.nature.com/articles/s41524-020-00406-3
**DOI:** https://doi.org/10.1038/s41524-020-00406-3

**Brief summary:**
This paper introduces Matbench, a benchmark suite for materials property prediction, and Automatminer, an automated reference ML pipeline.

**Why it matters for this project:**
It supports the evaluation philosophy of the project: use standard regression metrics, compare against classical ML baselines, and evaluate models under consistent data splits instead of reporting a single isolated result.

**How to use it in slides:**
Use it in the experimental design or evaluation slide: the project follows the benchmarking culture in materials informatics by comparing graph-based learning with descriptor-based baselines.

---

## 3. Optional Backup Papers for Q&A

### Deep Learning in Two-Dimensional Materials: Characterization, Prediction, and Design

**Link:** https://link.springer.com/article/10.1007/s11467-024-1394-7
**DOI:** https://doi.org/10.1007/s11467-024-1394-7

Use this as a recent review if someone asks whether deep learning is broadly used in 2D materials beyond band gap prediction. It is useful for broader impact and future work.

### SchNet: A Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions

**Link:** https://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions

SchNet is an important early atomistic neural network. It is useful if you want to explain how neural networks can handle continuous interatomic distances and atomic coordinates, but it is less central than CGCNN and ALIGNN for this project.

### Neural Message Passing for Quantum Chemistry

**Link:** https://arxiv.org/abs/1704.01212

This paper gives the general message passing neural network framework. It is useful for understanding GNN mathematics, but for a short course presentation, CGCNN and ALIGNN are enough.

### Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks

**Link:** https://arxiv.org/abs/1909.01315

DGL is the graph-learning library used under the hood by ALIGNN. Cite this only for technical Q&A; it is not central to the scientific motivation.

---

## 4. How to Connect the Papers into One Story

### Slide 1: Why 2D Materials?

Start with the materials problem, not the model.

**Key message:**
2D materials are important for electronics and optoelectronics because their electronic properties can be tuned by composition, thickness, strain, stacking, and local structure. The band gap is a key property because it determines whether a material behaves as a metal, semiconductor, or insulator.

**Supporting papers:**
Choudhary & Garrity 2017; Zhang et al. 2021.

---

### Slide 2: Why Is Band Gap Prediction Difficult?

Explain the computational bottleneck.

**Key message:**
DFT is physically grounded and reliable, but it is computationally expensive. For large-scale screening of candidate 2D materials, especially structures with defects, strain, heterostructures, or large supercells, doing DFT for every candidate is too slow.

**Supporting papers:**
Butler et al. 2018; JARVIS 2020.

---

### Slide 3: Why Machine Learning?

Move from the bottleneck to the data-driven solution.

**Key message:**
High-throughput DFT databases make supervised learning possible. We can use DFT-computed band gaps as labels and train an ML model to approximate the mapping from crystal structure to band gap. The ML model can then be used for fast screening.

**Supporting papers:**
Butler et al. 2018; JARVIS 2020.

---

### Slide 4: Why Not Only Classical ML?

Introduce the limitation of descriptors.

**Key message:**
Classical ML methods such as Random Forest can be effective baselines, but they usually depend on hand-crafted descriptors. For crystal structures, descriptors may lose local geometric information such as bond angles, local distortions, and connectivity patterns.

**Supporting papers:**
Zhang et al. 2021; Matbench 2020.

---

### Slide 5: Why Graph Neural Networks?

Introduce the structure representation.

**Key message:**
A crystal is naturally a graph: atoms are nodes and neighboring atom pairs are edges. Through message passing, each atom aggregates information from its local environment, and the model learns a structure-aware representation for material-level property prediction.

**Supporting papers:**
CGCNN 2018; MEGNet 2019.

---

### Slide 6: Why ALIGNN?

Explain the model choice.

**Key message:**
Standard crystal GNNs mainly capture pairwise atom-neighbor interactions. However, band gaps can be sensitive to local geometry and bond angles. ALIGNN introduces a line graph so that the model can explicitly learn bond-angle relationships, making it more physically motivated for structure-sensitive electronic properties.

**Supporting paper:**
ALIGNN 2021.

---

### Slide 7: What This Project Does

Bring the literature back to your own work.

**Key message:**
This project uses JARVIS-DFT data for band gap prediction and compares descriptor-based baselines with the ALIGNN/GNN approach. Even if ALIGNN self-training is limited by DGL/PyTorch/CUDA compatibility, the project still has a complete data-processing pipeline, baseline evaluation, and a literature-supported model motivation.

**Supporting papers:**
JARVIS 2020; ALIGNN 2021; Matbench 2020.

---

## 5. Presentation Script

You can use the following paragraph almost directly in your slides narration:

> The motivation of this project starts from two-dimensional materials. Their band gaps are central for electronic and optoelectronic applications, but the band gap is highly sensitive to atomic structure, local bonding, thickness, strain, and stacking. DFT can calculate these properties, but large-scale screening by DFT alone is computationally expensive.
>
> With high-throughput databases such as JARVIS-DFT, we can use existing DFT results as training data and build machine learning models that learn the mapping from crystal structure to band gap. Classical models such as Random Forest can already provide useful baselines, but they often rely on manually designed descriptors.
>
> A more natural representation is to treat a crystal as a graph: atoms are nodes and neighbor relations are edges. CGCNN introduced this idea for crystalline materials, and MEGNet further showed that graph networks are broadly useful for molecules and crystals. ALIGNN extends this idea by adding a line graph, which allows the model to learn not only pairwise atom-neighbor interactions but also bond-angle relationships.
>
> Therefore, ALIGNN is a physically motivated choice for this project: band gap is a structure-sensitive electronic property, and ALIGNN can encode richer local geometric information than a standard crystal graph model.

---

## 6. Minimal Reading Plan

**If you only have one hour:**

1. Choudhary & Garrity 2017: read the abstract and figures to understand the 2D materials high-throughput DFT background.
2. JARVIS 2020: read the abstract and dataset description to understand the data source.
3. CGCNN 2018: focus on Figure 1 and the method section to understand crystal graphs.
4. ALIGNN 2021: focus on the abstract, Figure 1, and benchmark table.

**If you have three hours:**

1. Add Zhang et al. 2021 for task-specific 2D band gap ML context.
2. Add Butler et al. 2018 for broader ML-for-materials motivation.
3. Add Matbench 2020 for evaluation and baseline framing.
4. Skim MEGNet 2019 to understand where ALIGNN fits in the crystal GNN literature.

---

## 7. Citation Mapping for Slides

| Slide Topic | Recommended Citation |
|---|---|
| Why 2D materials matter | Choudhary & Garrity 2017 |
| DFT is powerful but expensive; ML accelerates materials discovery | Butler et al. 2018 |
| Dataset source | JARVIS 2020 |
| Previous ML work on 2D band gap prediction | Zhang et al. 2021 |
| Crystals as graphs | CGCNN 2018 |
| GNNs as a general framework for molecules and crystals | MEGNet 2019 |
| Why ALIGNN | ALIGNN 2021 |
| Why compare baselines and report MAE/RMSE/R2 | Matbench 2020 |
