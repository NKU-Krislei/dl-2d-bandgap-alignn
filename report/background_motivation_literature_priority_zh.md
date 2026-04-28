# 时间紧迫版核心文献阅读清单

**项目题目：** Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks
**用途：** 这是一份新的精简版文档，用于在只有 1-2 天准备 slides 的情况下快速抓住 background 和 motivation。原始完整文档已保留。
**排序规则：** 按本项目的重要性从 1 到 n 排序。重要性指数为 1-5，**5 = 必读**，**1 = 可选**。

---

## 1. 时间很紧时最推荐读什么？

如果只有 **1-2 天**，优先读下面四篇：

| 排名 | 论文 | 重要性指数 | 为什么必须读 |
|---:|---|---:|---|
| 1 | ALIGNN | 5/5 | 项目的核心模型 |
| 2 | JARVIS | 5/5 | 项目的数据来源和数据驱动材料背景 |
| 3 | CGCNN | 5/5 | 解释为什么晶体可以表示成 graph |
| 4 | 2D Bandgap ML | 4/5 | 和本项目任务最接近的已有工作 |

这四篇已经足够把故事讲清楚：

> 二维材料需要快速 band gap prediction。JARVIS 提供 DFT 数据。CGCNN 说明晶体结构可以表示成 graph 并用 GNN 学习。ALIGNN 在此基础上加入 bond-angle 信息，因此更适合结构敏感的电子性质预测。

---

## 2. 按重要性排序的核心论文

### 1. Atomistic Line Graph Neural Network for Improved Materials Property Predictions

**重要性指数：** 5/5
**文献信息：** Choudhary, K., DeCost, B. et al. *npj Computational Materials* 7, 185 (2021).
**链接：** https://www.nature.com/articles/s41524-021-00650-1
**DOI：** https://doi.org/10.1038/s41524-021-00650-1

**为什么排第 1：**
这是项目的核心方法论文。项目标题中写的是 ALIGNN，所以 slides 必须讲清楚 ALIGNN 是什么、为什么选择它。

**要记住的核心思想：**
普通 crystal GNN 在 atom-bond graph 上做 message passing。ALIGNN 额外构建 **line graph**：把 bond 当成节点，把 bond-angle relationship 编码进去，因此可以学习更丰富的局部几何信息。

**Slides 中怎么讲：**
用于核心 method motivation：

> Band gap is sensitive to local structure. ALIGNN is suitable because it learns not only atom-neighbor interactions but also bond-angle information.

---

### 2. The Joint Automated Repository for Various Integrated Simulations (JARVIS) for Data-Driven Materials Design

**重要性指数：** 5/5
**文献信息：** Choudhary, K. et al. *npj Computational Materials* 6, 173 (2020).
**链接：** https://www.nature.com/articles/s41524-020-00440-1
**DOI：** https://doi.org/10.1038/s41524-020-00440-1

**为什么排第 2：**
这是项目数据来源的核心引用。既然本项目使用 JARVIS-DFT，就必须说明数据从哪里来，以及为什么它适合 supervised learning。

**要记住的核心思想：**
JARVIS 是面向材料设计的公开数据和工具平台。JARVIS-DFT 提供 DFT 计算得到的结构和性质，可用于训练材料性质预测模型。

**Slides 中怎么讲：**
用于 dataset/background slide：

> We use JARVIS-DFT because it provides large-scale DFT-computed material structures and band gap labels.

---

### 3. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties

**重要性指数：** 5/5
**文献信息：** Xie, T., & Grossman, J. C. *Physical Review Letters* 120, 145301 (2018).
**链接：** https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
**DOI：** https://doi.org/10.1103/PhysRevLett.120.145301

**为什么排第 3：**
这篇论文解释了 crystal GNN 的基本思想。没有 CGCNN，ALIGNN 的动机不容易讲清楚。

**要记住的核心思想：**
晶体可以表示成 graph：原子是 nodes，近邻原子对是 edges。GNN 通过 message passing 从结构中学习局部化学环境。

**Slides 中怎么讲：**
放在 ALIGNN 前面：

> CGCNN shows that crystal structures can be directly learned as graphs, avoiding heavy manual feature engineering.

---

### 4. Bandgap Prediction of Two-Dimensional Materials Using Machine Learning

**重要性指数：** 4/5
**文献信息：** Zhang, Y. et al. *PLOS ONE* 16, e0255637 (2021).
**链接：** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255637
**DOI：** https://doi.org/10.1371/journal.pone.0255637

**为什么排第 4：**
这是和本项目任务最接近的已有工作：材料对象是 2D materials，目标性质是 band gap，方法是 ML。

**要记住的核心思想：**
传统 ML 可以用于 2D material band gap prediction，但通常依赖人工设计 descriptors。

**Slides 中怎么讲：**
用于说明任务背景和 baseline：

> Previous work shows that ML is useful for 2D band gap prediction. Our project extends this motivation using graph-based structural learning.

---

### 5. Machine Learning for Molecular and Materials Science

**重要性指数：** 3/5
**文献信息：** Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. *Nature* 559, 547-555 (2018).
**链接：** https://www.nature.com/articles/s41586-018-0337-2
**DOI：** https://doi.org/10.1038/s41586-018-0337-2

**为什么排第 5：**
这是宏观综述。它适合做开场 motivation，但不如前四篇贴近项目本身。

**要记住的核心思想：**
ML 可以从昂贵的模拟和实验数据中学习快速 surrogate model，从而加速材料发现。

**Slides 中怎么讲：**
一句话带过即可：

> Machine learning is increasingly used as a fast surrogate model in materials science.

---

### 6. Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm

**重要性指数：** 3/5
**文献信息：** Dunn, A. et al. *npj Computational Materials* 6, 138 (2020).
**链接：** https://www.nature.com/articles/s41524-020-00406-3
**DOI：** https://doi.org/10.1038/s41524-020-00406-3

**为什么排第 6：**
这篇用于 evaluation framing。如果你想说明为什么要报告 MAE、RMSE、R2，为什么要和 baseline 比较，就引用它。

**要记住的核心思想：**
材料 ML 模型应该使用清晰的数据划分、标准指标和 baseline comparison。

**Slides 中怎么讲：**
用于 evaluation slide：

> We report MAE, RMSE, and R2 and compare against descriptor-based baselines.

---

### 7. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals

**重要性指数：** 2/5
**文献信息：** Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. *Chemistry of Materials* 31, 3564-3572 (2019).
**链接：** https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
**DOI：** https://doi.org/10.1021/acs.chemmater.9b01294

**为什么排第 7：**
MEGNet 可以帮助你把 ALIGNN 放进更大的 GNN-for-materials 文献脉络里，但短 presentation 不一定需要。

**要记住的核心思想：**
Graph networks 是适用于 molecules 和 crystals 的通用框架。

**Slides 中怎么讲：**
如果时间够，可以一句话提到：

> CGCNN and MEGNet established graph neural networks as effective models for crystal property prediction.

---

### 8. High-Throughput Identification and Characterization of Two-Dimensional Materials Using Density Functional Theory

**重要性指数：** 2/5
**文献信息：** Choudhary, K., & Garrity, K. F. *Scientific Reports* 7, 5179 (2017).
**链接：** https://www.nature.com/articles/s41598-017-05402-0
**DOI：** https://doi.org/10.1038/s41598-017-05402-0

**为什么排第 8：**
这篇可以加强二维材料背景，但如果时间很紧，不读它也能把本项目 story 讲清楚。

**要记住的核心思想：**
高通量 DFT 已经被用于系统性识别和表征大量二维材料。

**Slides 中怎么讲：**
只有在想加强 2D materials background 时使用。

---

## 3. Slides 最小 Story

建议按这个逻辑讲：

1. **Problem:** Band gap 是二维材料的重要电子性质，但 DFT 大规模筛选成本高。
2. **Data:** JARVIS-DFT 提供 DFT 计算得到的结构和 band gap labels。
3. **Baseline:** 传统 ML 可以预测 2D band gap，但通常依赖 hand-crafted descriptors。
4. **Graph representation:** CGCNN 说明晶体可以表示为 graph，并直接从原子结构学习。
5. **Model choice:** ALIGNN 加入 line graph message passing，能捕捉 bond-angle information。
6. **Project:** 本项目基于这个动机研究 crystal-structure-based band gap prediction，并与 descriptor-based baselines 比较。

---

## 4. 简短英文讲稿

> The motivation of this project starts from two-dimensional materials. Their band gaps are important for electronic and optoelectronic applications, but band gaps are sensitive to atomic structure and local bonding. DFT can compute these properties, but large-scale DFT screening is expensive.
>
> JARVIS-DFT provides a public database of DFT-computed material structures and properties, which makes supervised learning possible. Previous work has shown that classical machine learning can predict 2D material band gaps, but these methods often rely on hand-crafted descriptors.
>
> A more natural representation is to treat a crystal as a graph, where atoms are nodes and neighboring atom pairs are edges. CGCNN introduced this idea for crystal property prediction. ALIGNN extends it by adding a line graph, so the model can learn bond-angle relationships as well as atom-neighbor interactions.
>
> Therefore, ALIGNN is a physically motivated choice for band gap prediction: the target property is structure-sensitive, and the model can encode richer local geometric information than a standard crystal graph model.
