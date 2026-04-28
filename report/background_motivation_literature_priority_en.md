# Time-Critical Literature Guide for Slides

**Project:** Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks
**Purpose:** Select only the papers needed to tell a clear background and motivation story under a tight deadline.
**Ranking rule:** Papers are ordered by importance for this project. The importance index is from 1 to 5, where **5 = essential** and **1 = optional**.

---

## 1. Recommended Reading Under a Tight Deadline

If you only have **1-2 days**, read these four papers first:

| Rank | Paper | Importance | Why it is essential |
|---:|---|---:|---|
| 1 | ALIGNN | 5/5 | Core model of the project |
| 2 | JARVIS | 5/5 | Dataset source and data-driven materials context |
| 3 | CGCNN | 5/5 | Explains why crystals can be represented as graphs |
| 4 | 2D Bandgap ML | 4/5 | Closest prior work to the project task |

These four papers are enough to tell the full story:

> 2D materials need fast band gap prediction. JARVIS provides DFT data. CGCNN shows that crystal structures can be learned as graphs. ALIGNN improves crystal GNNs by adding bond-angle information, which is physically relevant for structure-sensitive electronic properties.

---

## 2. Ranked Core Papers

### 1. Atomistic Line Graph Neural Network for Improved Materials Property Predictions

**Importance index:** 5/5
**Citation:** Choudhary, K., DeCost, B. et al. *npj Computational Materials* 7, 185 (2021).
**Link:** https://www.nature.com/articles/s41524-021-00650-1
**DOI:** https://doi.org/10.1038/s41524-021-00650-1

**Why it is ranked #1:**
This is the core method paper for the project. ALIGNN is the model named in the project title, so the slides must clearly explain what it is and why it is chosen.

**Key idea to remember:**
Standard crystal GNNs pass messages on an atom-bond graph. ALIGNN also builds a **line graph**, where bonds become nodes and bond-angle relationships are encoded. This allows the model to learn richer local geometry.

**How to use it in slides:**
Use this paper for the main method motivation:

> Band gap is sensitive to local structure. ALIGNN is suitable because it learns not only atom-neighbor interactions but also bond-angle information.

---

### 2. The Joint Automated Repository for Various Integrated Simulations (JARVIS) for Data-Driven Materials Design

**Importance index:** 5/5
**Citation:** Choudhary, K. et al. *npj Computational Materials* 6, 173 (2020).
**Link:** https://www.nature.com/articles/s41524-020-00440-1
**DOI:** https://doi.org/10.1038/s41524-020-00440-1

**Why it is ranked #2:**
This is the dataset paper. Since the project uses JARVIS-DFT, this paper is necessary for explaining where the data comes from and why it is reliable.

**Key idea to remember:**
JARVIS is a public data infrastructure for materials design. JARVIS-DFT contains DFT-computed structures and properties that can be used to train supervised ML models.

**How to use it in slides:**
Use it in the dataset/background slide:

> We use JARVIS-DFT because it provides large-scale DFT-computed material structures and band gap labels.

---

### 3. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties

**Importance index:** 5/5
**Citation:** Xie, T., & Grossman, J. C. *Physical Review Letters* 120, 145301 (2018).
**Link:** https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
**DOI:** https://doi.org/10.1103/PhysRevLett.120.145301

**Why it is ranked #3:**
This paper explains the basic idea behind crystal GNNs. Without CGCNN, it is hard to explain ALIGNN clearly.

**Key idea to remember:**
A crystal can be represented as a graph: atoms are nodes and neighboring atom pairs are edges. GNN message passing learns local chemical environments from structure.

**How to use it in slides:**
Use it before ALIGNN:

> CGCNN shows that crystal structures can be directly learned as graphs, avoiding heavy manual feature engineering.

---

### 4. Bandgap Prediction of Two-Dimensional Materials Using Machine Learning

**Importance index:** 4/5
**Citation:** Zhang, Y. et al. *PLOS ONE* 16, e0255637 (2021).
**Link:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255637
**DOI:** https://doi.org/10.1371/journal.pone.0255637

**Why it is ranked #4:**
This is the closest prior work to the project task: it focuses on band gap prediction for 2D materials using ML.

**Key idea to remember:**
Classical ML methods can predict 2D material band gaps, but they usually rely on hand-crafted descriptors.

**How to use it in slides:**
Use it to justify both the task and the baseline:

> Previous work shows that ML is useful for 2D band gap prediction. Our project extends this motivation using graph-based structural learning.

---

### 5. Machine Learning for Molecular and Materials Science

**Importance index:** 3/5
**Citation:** Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. *Nature* 559, 547-555 (2018).
**Link:** https://www.nature.com/articles/s41586-018-0337-2
**DOI:** https://doi.org/10.1038/s41586-018-0337-2

**Why it is ranked #5:**
This is a broad review. It is useful for the opening motivation, but it is less project-specific than ALIGNN, JARVIS, CGCNN, and the 2D band gap paper.

**Key idea to remember:**
ML accelerates materials discovery by learning from expensive simulations and experiments.

**How to use it in slides:**
Use one sentence in the introduction:

> Machine learning is increasingly used as a fast surrogate model in materials science.

---

### 6. Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm

**Importance index:** 3/5
**Citation:** Dunn, A. et al. *npj Computational Materials* 6, 138 (2020).
**Link:** https://www.nature.com/articles/s41524-020-00406-3
**DOI:** https://doi.org/10.1038/s41524-020-00406-3

**Why it is ranked #6:**
This paper is useful for evaluation framing, especially if you want to justify metrics and baseline comparison.

**Key idea to remember:**
Materials ML models should be evaluated using clear splits, standard metrics, and baseline comparisons.

**How to use it in slides:**
Use it in the evaluation slide:

> We report MAE, RMSE, and R2 and compare against descriptor-based baselines.

---

### 7. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals

**Importance index:** 2/5
**Citation:** Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. *Chemistry of Materials* 31, 3564-3572 (2019).
**Link:** https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
**DOI:** https://doi.org/10.1021/acs.chemmater.9b01294

**Why it is ranked #7:**
MEGNet helps position ALIGNN within the broader GNN-for-materials literature, but it is not necessary for a short presentation.

**Key idea to remember:**
Graph networks are a general framework for molecules and crystals, not just one specific architecture.

**How to use it in slides:**
Mention only if you have time:

> CGCNN and MEGNet established graph neural networks as effective models for crystal property prediction.

---

### 8. High-Throughput Identification and Characterization of Two-Dimensional Materials Using Density Functional Theory

**Importance index:** 2/5
**Citation:** Choudhary, K., & Garrity, K. F. *Scientific Reports* 7, 5179 (2017).
**Link:** https://www.nature.com/articles/s41598-017-05402-0
**DOI:** https://doi.org/10.1038/s41598-017-05402-0

**Why it is ranked #8:**
This paper is useful for extra 2D materials background, but the project story can still be clear without it if time is short.

**Key idea to remember:**
High-throughput DFT has already been used to identify and characterize large numbers of 2D materials.

**How to use it in slides:**
Use it only if you want a stronger 2D materials background slide.

---

## 3. Minimal Story for Slides

Use this as the narrative spine:

1. **Problem:** Band gap is a key electronic property for 2D materials, but DFT screening is expensive.
2. **Data:** JARVIS-DFT provides DFT-computed structures and band gap labels.
3. **Baseline:** Classical ML can predict 2D band gaps, but often depends on hand-crafted descriptors.
4. **Graph representation:** CGCNN shows that crystals can be represented as graphs and learned directly from atomic structure.
5. **Model choice:** ALIGNN improves this by adding line graph message passing, which captures bond-angle information relevant to structure-sensitive properties.
6. **Project:** We use this motivation to study band gap prediction from crystal structures and compare against descriptor-based baselines.

---

## 4. Short Presentation Script

> The motivation of this project starts from two-dimensional materials. Their band gaps are important for electronic and optoelectronic applications, but band gaps are sensitive to atomic structure and local bonding. DFT can compute these properties, but large-scale DFT screening is expensive.
>
> JARVIS-DFT provides a public database of DFT-computed material structures and properties, which makes supervised learning possible. Previous work has shown that classical machine learning can predict 2D material band gaps, but these methods often rely on hand-crafted descriptors.
>
> A more natural representation is to treat a crystal as a graph, where atoms are nodes and neighboring atom pairs are edges. CGCNN introduced this idea for crystal property prediction. ALIGNN extends it by adding a line graph, so the model can learn bond-angle relationships as well as atom-neighbor interactions.
>
> Therefore, ALIGNN is a physically motivated choice for band gap prediction: the target property is structure-sensitive, and the model can encode richer local geometric information than a standard crystal graph model.
