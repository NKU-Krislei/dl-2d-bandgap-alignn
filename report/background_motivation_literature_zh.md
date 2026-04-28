# Slides Background & Motivation 核心文献阅读报告

**项目题目：** Predicting Band Gaps of Two-Dimensional Materials Using Graph Neural Networks
**用途：** 这份中文文档用于你自己理解和备课；英文 slides 可以参考对应的英文版 `background_motivation_literature_en.md`。
**阅读优先级：** 如果时间紧，优先读第 2、3、5、7 篇；如果要准备答辩 Q&A，再读第 1、4、6、8 篇。

---

## 1. 一句话 Story

二维材料具有重要但高度结构敏感的电子性质；DFT 可以计算 band gap，但难以承担大规模筛选；机器学习可以从已有 DFT 数据中学习快速 surrogate model；晶体结构天然适合表示为 graph，因此 GNN 是合理选择；ALIGNN 进一步通过 line graph 显式编码 bond-angle 信息，使模型更适合预测结构敏感的材料性质。

---

## 2. 核心阅读清单

### 1. Machine Learning for Molecular and Materials Science

**文献信息：** Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. *Nature* 559, 547-555 (2018).
**链接：** https://www.nature.com/articles/s41586-018-0337-2
**DOI：** https://doi.org/10.1038/s41586-018-0337-2

**简要介绍：**
这是一篇关于机器学习在化学和材料科学中应用的综述论文。它解释了 ML 如何加速性质预测、材料发现、表征和设计。

**为什么和本项目相关：**
这篇论文支撑项目的最高层 motivation：ML 不是为了取代物理模型，而是从昂贵的 DFT 或实验数据中学习快速近似模型，用于更大规模的材料筛选。

**如何用于 slides：**
放在第一张 background slide，用来说明材料发现正在从“逐个做第一性原理计算”转向“高通量数据 + 机器学习”的 workflow。

---

### 2. High-Throughput Identification and Characterization of Two-Dimensional Materials Using Density Functional Theory

**文献信息：** Choudhary, K., & Garrity, K. F. *Scientific Reports* 7, 5179 (2017).
**链接：** https://www.nature.com/articles/s41598-017-05402-0
**DOI：** https://doi.org/10.1038/s41598-017-05402-0

**简要介绍：**
这篇论文用高通量 DFT 识别和表征二维材料，是二维材料数据库和高通量计算方向的重要工作。

**为什么和本项目相关：**
它直接支撑项目的材料背景：二维材料是一个很大的候选空间，具有重要应用价值，但如果系统性筛选全部依赖 DFT，计算成本会很高。

**如何用于 slides：**
用于解释为什么选择 2D materials：二维材料科学意义强、应用潜力大，也适合结合高通量计算和 ML 进行筛选。

---

### 3. The Joint Automated Repository for Various Integrated Simulations (JARVIS) for Data-Driven Materials Design

**文献信息：** Choudhary, K. et al. *npj Computational Materials* 6, 173 (2020).
**链接：** https://www.nature.com/articles/s41524-020-00440-1
**DOI：** https://doi.org/10.1038/s41524-020-00440-1

**简要介绍：**
这篇论文介绍 JARVIS 数据和工具体系，包括 JARVIS-DFT、JARVIS-ML 和 JARVIS-tools。JARVIS-DFT 提供大量 DFT 计算得到的材料结构和性质。

**为什么和本项目相关：**
这是本项目数据来源的核心引用。它说明公开 DFT 数据库如何让 supervised learning 的材料性质预测成为可能。

**如何用于 slides：**
在 dataset slide 中引用：本项目使用 JARVIS-DFT，这是 NIST 提供的公开数据库，包含 DFT 计算得到的材料结构和性质。

---

### 4. Bandgap Prediction of Two-Dimensional Materials Using Machine Learning

**文献信息：** Zhang, Y. et al. *PLOS ONE* 16, e0255637 (2021).
**链接：** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255637
**DOI：** https://doi.org/10.1371/journal.pone.0255637

**简要介绍：**
这篇论文使用传统机器学习方法预测二维材料 band gap，包括 SVR、MLP、GBDT 和 Random Forest。

**为什么和本项目相关：**
它和本项目任务最接近：材料对象是二维材料，目标性质是 band gap。它也说明 Random Forest 是一个合理 baseline，而不是随意选择的模型。

**如何用于 slides：**
用来引出 classical ML 和 graph-based ML 的区别：已有工作证明 ML 可以预测 2D band gap，但很多方法依赖人工设计 descriptors。

---

### 5. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties

**文献信息：** Xie, T., & Grossman, J. C. *Physical Review Letters* 120, 145301 (2018).
**链接：** https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
**DOI：** https://doi.org/10.1103/PhysRevLett.120.145301

**简要介绍：**
这是 crystal graph neural network 的开创性论文。它将晶体表示成 graph：原子是 nodes，近邻原子对是 edges，然后通过 graph convolution 预测材料性质。

**为什么和本项目相关：**
CGCNN 是从 descriptor-based ML 走向 structure-based graph learning 的关键桥梁，也是理解 ALIGNN 的前置基础。

**如何用于 slides：**
用于解释为什么晶体适合 GNN：晶体天然由原子和近邻关系组成，GNN 的 message passing 可以学习局域化学环境。

---

### 6. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals

**文献信息：** Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. *Chemistry of Materials* 31, 3564-3572 (2019).
**链接：** https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
**DOI：** https://doi.org/10.1021/acs.chemmater.9b01294

**简要介绍：**
这篇论文提出 MEGNet，将 graph network 用于分子和晶体性质预测，并展示了图模型在不同材料性质任务中的泛化能力。

**为什么和本项目相关：**
它帮助你说明 GNN 不是某一个单独模型，而是一类适合 atomistic property prediction 的通用框架。它也能帮助把 ALIGNN 放在 crystal GNN 的发展脉络中。

**如何用于 slides：**
在 CGCNN 之后简短提到：CGCNN 开启了 crystal graph learning，MEGNet 进一步说明 graph networks 可以作为分子和晶体的通用框架。

---

### 7. Atomistic Line Graph Neural Network for Improved Materials Property Predictions

**文献信息：** Choudhary, K., DeCost, B. et al. *npj Computational Materials* 7, 185 (2021).
**链接：** https://www.nature.com/articles/s41524-021-00650-1
**DOI：** https://doi.org/10.1038/s41524-021-00650-1

**简要介绍：**
这是 ALIGNN 的核心论文。ALIGNN 同时在 atomistic bond graph 和 line graph 上做 message passing。普通 graph 中，原子是 nodes、键是 edges；line graph 中，键变成 nodes，共享一个原子的两条键之间形成连接，从而编码 bond-angle relationships。

**为什么和本项目相关：**
这是项目的核心方法来源。它的物理动机是：许多材料性质不仅依赖原子间距离，也依赖局域几何和键角。Band gap 是结构敏感的电子性质，因此 ALIGNN 是一个有物理合理性的模型选择。

**如何用于 slides：**
这是最重要的 method motivation slide。建议按下面逻辑讲：

1. CGCNN 学习 atom-neighbor interactions。
2. 但 band gap 可能对 local geometry 和 bond angles 敏感。
3. ALIGNN 加入 line graph，让模型学习 bond-angle relationships。
4. 因此 ALIGNN 更适合结构敏感的材料性质预测。

---

### 8. Benchmarking Materials Property Prediction Methods: The Matbench Test Set and Automatminer Reference Algorithm

**文献信息：** Dunn, A. et al. *npj Computational Materials* 6, 138 (2020).
**链接：** https://www.nature.com/articles/s41524-020-00406-3
**DOI：** https://doi.org/10.1038/s41524-020-00406-3

**简要介绍：**
这篇论文提出 Matbench，一个材料性质预测 benchmark，并介绍 Automatminer 作为自动化 ML 参考方法。

**为什么和本项目相关：**
它支撑项目的 evaluation 思路：应该使用标准 regression metrics，和 classical ML baselines 比较，并在一致的数据划分下评估模型，而不是只展示单个模型结果。

**如何用于 slides：**
在 experimental design 或 evaluation slide 中使用：本项目遵循 materials informatics 中的 benchmark 思路，比较 graph-based model 和 descriptor-based baseline。

---

## 3. 可选补充文献，用于 Q&A

### Deep Learning in Two-Dimensional Materials: Characterization, Prediction, and Design

**链接：** https://link.springer.com/article/10.1007/s11467-024-1394-7
**DOI：** https://doi.org/10.1007/s11467-024-1394-7

如果有人问 deep learning 是否广泛用于二维材料，可以引用这篇较新的综述。它适合用于 broader impact 或 future work。

### SchNet: A Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions

**链接：** https://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions

SchNet 是早期重要的 atomistic neural network。如果你想解释神经网络如何处理连续原子坐标和原子间距离，可以参考它；但对于本项目，CGCNN 和 ALIGNN 更核心。

### Neural Message Passing for Quantum Chemistry

**链接：** https://arxiv.org/abs/1704.01212

这篇论文给出 message passing neural network 的通用数学框架。适合理解 GNN 数学，但短 presentation 中不需要重点展开。

### Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks

**链接：** https://arxiv.org/abs/1909.01315

DGL 是 ALIGNN 底层使用的图学习库。如果老师问 DGL 是什么，可以引用这篇；但它不是 scientific motivation 的核心文献。

---

## 4. 如何把这些论文串成一个完整 story

### Slide 1: Why 2D Materials?

先讲材料问题，不要一开始就讲模型。

**核心信息：**
二维材料在电子学和光电子学中很重要，因为它们的电子性质可以通过组成、厚度、应变、堆垛和局部结构调控。Band gap 是关键性质，因为它决定材料是金属、半导体还是绝缘体。

**支撑文献：**
Choudhary & Garrity 2017; Zhang et al. 2021.

---

### Slide 2: Why Is Band Gap Prediction Difficult?

讲清楚计算瓶颈。

**核心信息：**
DFT 有清晰的物理基础，也比较可靠，但计算成本高。对于大量二维候选材料，尤其是包含缺陷、应变、异质结构或大超胞的体系，逐个做 DFT 太慢。

**支撑文献：**
Butler et al. 2018; JARVIS 2020.

---

### Slide 3: Why Machine Learning?

从瓶颈自然过渡到数据驱动方案。

**核心信息：**
高通量 DFT 数据库让 supervised learning 成为可能。我们可以把 DFT 计算得到的 band gap 当作 label，训练 ML 模型近似“晶体结构 -> band gap”的映射，从而用于快速筛选。

**支撑文献：**
Butler et al. 2018; JARVIS 2020.

---

### Slide 4: Why Not Only Classical ML?

引出 descriptors 的局限。

**核心信息：**
Random Forest 等 classical ML 可以作为有效 baseline，但通常依赖 hand-crafted descriptors。对于晶体结构，这些 descriptors 可能损失局部几何信息，例如 bond angles、local distortions 和 connectivity patterns。

**支撑文献：**
Zhang et al. 2021; Matbench 2020.

---

### Slide 5: Why Graph Neural Networks?

引出结构表示。

**核心信息：**
晶体天然是 graph：原子是 nodes，相邻原子对是 edges。通过 message passing，每个原子聚合局部环境信息，模型学习 structure-aware representation，再用于材料整体性质预测。

**支撑文献：**
CGCNN 2018; MEGNet 2019.

---

### Slide 6: Why ALIGNN?

解释模型选择。

**核心信息：**
标准 crystal GNN 主要捕捉 pairwise atom-neighbor interactions；但 band gap 可能对 local geometry 和 bond angles 敏感。ALIGNN 引入 line graph，让模型显式学习 bond-angle relationships，因此更适合结构敏感的电子性质。

**支撑文献：**
ALIGNN 2021.

---

### Slide 7: What This Project Does

把文献回收到自己的项目。

**核心信息：**
本项目使用 JARVIS-DFT 数据进行 band gap prediction，并比较 descriptor-based baselines 与 ALIGNN/GNN 思路。即使 ALIGNN 自训练受到 DGL/PyTorch/CUDA 兼容性限制，项目仍然有完整的数据处理流程、baseline evaluation 和文献支持的方法动机。

**支撑文献：**
JARVIS 2020; ALIGNN 2021; Matbench 2020.

---

## 5. 英文讲解稿的中文理解版

下面这段对应英文版中的 presentation script，适合你先用中文理解逻辑，再转换成英文讲：

> 本项目的动机从二维材料开始。二维材料的 band gap 对电子和光电子应用非常关键，但 band gap 对原子结构、局域键合、厚度、应变和堆垛方式高度敏感。DFT 可以计算这些性质，但如果要进行大规模筛选，单靠 DFT 的成本太高。
>
> 随着 JARVIS-DFT 这样的高通量数据库发展，我们可以把已有 DFT 结果作为训练数据，建立机器学习模型，学习从晶体结构到 band gap 的映射。Random Forest 等传统模型可以提供有效 baseline，但通常依赖人工设计 descriptors。
>
> 更自然的表示方法是把晶体看成 graph：原子是节点，近邻关系是边。CGCNN 首先将这一思想系统用于晶体材料，MEGNet 进一步说明 graph networks 对分子和晶体都很有用。ALIGNN 在此基础上加入 line graph，让模型不仅学习 pairwise atom-neighbor interactions，还能学习 bond-angle relationships。
>
> 因此，ALIGNN 对本项目来说是一个有物理动机的选择：band gap 是结构敏感的电子性质，而 ALIGNN 可以比普通 crystal graph model 编码更丰富的局域几何信息。

---

## 6. 最小阅读计划

**如果只有 1 小时：**

1. Choudhary & Garrity 2017：读 abstract 和 figures，理解二维材料高通量 DFT 背景。
2. JARVIS 2020：读 abstract 和 dataset 描述，理解数据来源。
3. CGCNN 2018：重点看 Figure 1 和 method section，理解 crystal graph。
4. ALIGNN 2021：重点看 abstract、Figure 1 和 benchmark table。

**如果有 3 小时：**

1. 加读 Zhang et al. 2021，补充 2D band gap ML 背景。
2. 加读 Butler et al. 2018，补充 ML-for-materials 的宏观 motivation。
3. 加读 Matbench 2020，理解 evaluation 和 baseline framing。
4. 快速浏览 MEGNet 2019，理解 ALIGNN 在 crystal GNN 文献脉络中的位置。

---

## 7. Slides 引用对应表

| Slide 主题 | 推荐引用 |
|---|---|
| 为什么二维材料重要 | Choudhary & Garrity 2017 |
| DFT 强大但昂贵，ML 可加速材料发现 | Butler et al. 2018 |
| 数据集来源 | JARVIS 2020 |
| 既有 2D band gap ML 工作 | Zhang et al. 2021 |
| 晶体可以表示为 graph | CGCNN 2018 |
| GNN 是分子和晶体的通用框架 | MEGNet 2019 |
| 为什么选择 ALIGNN | ALIGNN 2021 |
| 为什么要比较 baseline 并报告 MAE/RMSE/R2 | Matbench 2020 |
