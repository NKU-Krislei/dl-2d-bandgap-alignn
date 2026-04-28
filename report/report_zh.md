# 使用 ALIGNN 预测二维材料带隙

**作者**：Xiaoyu Wang
**课程**：INFO5000 — Introduction to Data Science
**日期**：2026 年 4 月 28 日

## 摘要

本项目使用 Atomistic Line Graph Neural Network（ALIGNN）从晶体结构预测材料带隙，并以基于成分的传统回归模型作为基线。已实现的流程包括下载 JARVIS 数据集、构建确定性的训练/验证/测试划分、生成探索性分析图、在 Magpie 成分特征上评估 Random Forest 和 Ridge 基线模型，以及在支持 CUDA 的 V100 实例上训练 graphwise ALIGNN 模型。在最终运行中，处理后的数据集包含 75993 个有效材料，划分规模为 train=60794、validation=7599、test=7600。自训练 ALIGNN 模型在测试集上取得 MAE 0.115 eV、RMSE 0.380 eV、R² 0.922，优于 Random Forest 基线的 MAE 0.273 eV 和 Ridge 的 MAE 0.689 eV。这些结果支持本项目假设：能够感知键角信息的图神经网络可以相比仅使用成分信息的基线模型显著提升带隙预测性能。

## 1. 引言

二维材料具有重要吸引力，因为其电子性质强烈依赖原子排列、化学成分和局域成键几何。本项目将带隙作为核心预测目标，因为带隙决定材料表现为金属、半导体还是绝缘体，并进一步决定其在电子器件和光电子器件中的适用性。

直接进行密度泛函理论计算通常较为准确，但计算成本很高。因此，需要能够更快速地将晶体结构映射到带隙的机器学习模型。ALIGNN 是一个有竞争力的候选方法，因为它同时结合晶体图和线图，使模型能够编码键连接关系与键角信息。

## 2. 方法

本流程遵循项目 proposal 的结构。首先下载 JARVIS 数据集，并筛选出带隙值非负且结构文件可用的有效材料。随后生成确定性的 80/10/10 数据划分，并写入 `data/train_id_prop.csv`、`data/val_id_prop.csv` 和 `data/test_id_prop.csv`；同时生成一个包含 1000 个样本的 quick 子集，用于调试和仅 CPU 的测试。

对于基线模型，本项目使用 Magpie 成分描述符，并训练两个回归器：Random Forest 和 Ridge Regression。对于深度学习部分，最终运行采用直接的 ALIGNN graphwise 训练循环，以绕过不兼容的 CLI 行为，同时保留 ALIGNN 官方的图构建、dataloader 和模型实现。V100 训练运行使用 CUDA、batch size 64、50 个 epoch、LMDB 图缓存，以及确定性的数据划分顺序。

## 3. 数据集

本次运行得到的处理后数据集统计如下：

- 有效材料总数：75993
- 训练 / 验证 / 测试：60794 / 7599 / 7600
- 平均带隙：0.613 eV
- 标准差：1.343 eV
- 金属：53186
- 半导体：16908
- 绝缘体：5899

处理后元数据中识别出的主要材料家族是 Phosphorene。家族标签是启发式标签，由化学式和材料标识符推断得到，用于后续的分组评估。

## 4. 结果

### 4.1 数据探索

探索性分析总结了整体带隙分布、金属/半导体/绝缘体划分、带隙与晶胞大小之间的关系，以及最常见的材料家族。这些结果保存在 `figures/data_exploration.png`。

### 4.2 基线比较

- Pretrained ALIGNN：MAE 不可用，RMSE 不可用，R² 不可用
- Self-trained ALIGNN：MAE 0.115 eV，RMSE 0.380 eV，R² 0.922
- Random Forest：MAE 0.273 eV，RMSE 0.611 eV，R² 0.798
- Ridge Regression：MAE 0.689 eV，RMSE 1.034 eV，R² 0.421

比较图保存为 `figures/method_comparison.png`。

### 4.3 ALIGNN 训练曲线

自训练 ALIGNN 阶段在 V100 GPU 上完成了全部 50 个 epoch。最佳验证 MAE 为 0.117 eV，出现在第 48 个 epoch；最终保留测试集 MAE 为 0.115 eV。最终训练损失为 0.040 eV，表明模型相较于传统基线已经显著收敛到更优性能。训练历史图保存为 `figures/training_history.png`。

### 4.4 预测散点图

每个可用模型都生成了预测值与真实值对比图以及残差图。自训练 ALIGNN 的图保存在 `figures/eval_self_trained.png`，传统基线模型的图分别保存在 `figures/eval_random_forest.png` 和 `figures/eval_ridge.png`。

### 4.5 误差分析

按材料家族划分的误差分析写入 `results/evaluation_report.json`；当有足够的分组数据时，也会在 `figures/per_family_performance.png` 中进行可视化。基线模型的学习曲线结果保存到 `results/learning_curve.json`，并在 `figures/learning_curve.png` 中可视化。

## 5. 讨论

最终结果显示出清晰的性能层级。自训练 ALIGNN 模型相对于 Random Forest 基线将 MAE 降低了约 58%（0.115 eV 对比 0.273 eV），并取得 R² 0.922，高于 Random Forest 的 0.798。这一提升与 ALIGNN 的物理动机一致：键角信息和局域图结构包含仅凭成分描述符无法充分表示的结构信号。

本项目仍然保留传统基线模型作为可靠性后备方案。这一点很有价值，因为深度学习技术栈依赖 CUDA、PyTorch、DGL 和 ALIGNN 之间的版本兼容性。在 V100 实例上安装与 CUDA 匹配的 DGL wheel 后，直接训练循环成功完成，并产生了最强结果。

## 6. 结论

本项目目前已经具备一个可运行、可复现的 JARVIS 材料带隙预测实验流程。本地 CPU 路径覆盖数据下载、数据划分、探索性分析、预训练模型推理尝试、Random Forest 和 Ridge 基线、评估图、概念图以及报告生成。远程 GPU 路径成功训练了 ALIGNN，并达到测试 MAE 0.115 eV，明显低于 0.25 eV 的目标阈值，也清楚优于 Random Forest 基线。

## 生成的图

- `data_exploration.png`：带隙分布和数据集概览
- `alignn_overview.png`：ALIGNN 工作流程图
- `crystal_graph_demo.png`：晶体图和线图概念示意
- `physics_context.png`：物理动机和机器学习流程背景
- `eval_pretrained.png`：本次运行未生成
- `eval_self_trained.png`：自训练 ALIGNN 的预测图和残差图
- `training_history.png`：ALIGNN 训练和验证曲线
- `method_comparison.png`：模型基准比较
- `learning_curve.png`：基线模型学习曲线
- `per_family_performance.png`：按材料家族分组的 MAE 比较

## 参考文献

1. Xie, T., & Grossman, J. C. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.
2. Choudhary, K., et al. Atomistic Line Graph Neural Network for improved materials property predictions.
3. Ward, L., et al. A general-purpose machine learning framework for predicting properties of inorganic materials.
4. Butler, K. T., et al. Machine learning for molecular and materials science.
5. Meng, S., et al. Deep learning in two-dimensional materials research.

## 附录

### 可复现性

1. 安装 `requirements.txt` 中列出的依赖。
2. 运行 `python run_pipeline.py --quick`，执行快速的仅 CPU 调试路径。
3. 运行 `python run_pipeline.py`，执行完整的本地流程。
4. 对于远程 CUDA 训练，使用 `results/gpu_connection.json` 中的凭据，然后运行 `python src/train_direct.py --device cuda --epochs 50 --batch_size 64`。
5. 下载 GPU 结果后，重新运行 `python src/evaluate.py` 和 `python src/visualize.py`，以刷新报告资源。
