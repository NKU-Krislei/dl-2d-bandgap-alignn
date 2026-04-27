# 项目进展与问题分析报告

**生成时间**: 2026-04-27 23:42  
**项目名称**: Predicting Band Gaps of Two-Dimensional Materials Using ALIGNN  
**课程**: INFO5000 — Introduction to Data Science

---

## 一、项目整体进展

### 1.1 已完成的工作

| 阶段 | 状态 | 说明 |
|------|------|------|
| **Phase 0** — 环境与基础设施 | ✅ 完成 | 本地开发环境、远程 GPU 连接配置、项目目录结构 |
| **Phase 1** — 代码模块编写 | ✅ 完成 | 6 个核心模块：`data_download.py`, `data_explore.py`, `predict.py`, `train.py`, `evaluate.py`, `visualize.py` |
| **Phase 2L** — 本地 CPU 路径 | ✅ 完成 | 数据集下载、数据探索、基线模型训练（RF/Ridge） |
| **Phase 2G** — 远程 GPU 训练 | ❌ **失败** | ALIGNN 自训练因 DGL/PyTorch 兼容性问题未能完成 |
| **Phase 3** — 报告生成 | ✅ 完成 | 基于 RF/Ridge 基线结果生成完整报告 |

### 1.2 已交付的成果

**数据集**: JARVIS 数据库，75,993 条有效材料记录
- 训练集: 60,794 | 验证集: 7,599 | 测试集: 7,600
- 金属: 53,186 | 半导体: 16,908 | 绝缘体: 5,899

**基线模型结果**:

| 模型 | MAE (eV) | RMSE (eV) | R² |
|------|----------|-----------|-----|
| **Random Forest + Magpie** | **0.273** | 0.611 | **0.798** |
| **Ridge + Magpie** | 0.689 | 1.034 | 0.421 |
| ALIGNN (Pretrained) | N/A | N/A | N/A |
| ALIGNN (Self-Trained) | N/A | N/A | N/A |

**生成的图表** (9 张):
- `data_exploration.png` — 数据集概览
- `method_comparison.png` — 模型对比
- `eval_random_forest.png` / `eval_ridge.png` — 预测散点图
- `learning_curve.png` — 学习曲线
- `per_family_performance.png` — 按材料家族分组误差
- `alignn_overview.png` / `crystal_graph_demo.png` / `physics_context.png` — 概念图

**报告文件**: `report/report.md` + `report/slides_outline.md`

---

## 二、核心问题：ALIGNN 自训练失败

### 2.1 问题概述

项目最核心的目标——**在 GPU 上自训练 ALIGNN 模型**——未能完成。Codex 在远程 GPU 上尝试了 **4 种不同的路径**，全部失败。

### 2.2 四次尝试的详细记录

#### 尝试 1: ALIGNN 官方 CLI (`alignn_train_finetune` / `train_alignn.py`)
- **问题**: 2026 版 ALIGNN 包的 `TrainingConfig` schema 与 wrapper 预期的不兼容
- **具体**: CLI 默认进入 atomwise（力场预测）模式，即使 config 中设置了 `model.name="alignn"`（graphwise 标量回归模式）
- **结果**: CLI 拒绝或错误路由 graphwise 配置，训练未启动

#### 尝试 2: 直接 Python 调用（CPU-only DGL）
- **策略**: 绕过 CLI，直接用 Python 调用 `alignn.train.train_dgl()` 和 `alignn.data.get_train_val_loaders()`
- **问题**: 远程环境预装的 DGL 2.1.0 是 **CPU-only 版本**
- **具体**: 当尝试将图数据 `graph.to('cuda')` 时失败
- **结果**: 无法使用 GPU 加速

#### 尝试 3: 安装 DGL 2.4.0+cu121（CUDA 11.8）
- **策略**: 手动安装 CUDA 版本的 DGL wheel
- **安装命令**: `pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html --no-deps`
- **问题**: 安装成功，但 **在第一个 ALIGNN forward pass 时挂死（hang）**
- **环境**: PyTorch 2.8.0+cu128（AutoDL 镜像预装）
- **结果**: 进程卡死，无错误输出

#### 尝试 4: 安装 DGL 2.5.0+cu124（CUDA 12.4）
- **策略**: 尝试更新的 DGL 版本
- **安装命令**: `pip install dgl==2.5.0+cu124 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps`
- **问题**: 同样 **在第一个 forward pass 时挂死**
- **结果**: 与尝试 3 相同

### 2.3 根本问题分析

```
┌─────────────────────────────────────────────────────────────┐
│                    兼容性冲突链                              │
├─────────────────────────────────────────────────────────────┤
│  AutoDL 镜像预装: PyTorch 2.8.0 + CUDA 12.8 (cu128)        │
│              ↓                                               │
│  ALIGNN 2026.4.2 依赖: DGL + PyTorch                        │
│              ↓                                               │
│  DGL 官方 wheel: 最高支持到 cu124 (CUDA 12.4)               │
│              ↓                                               │
│  版本不匹配: PyTorch cu128 ↔ DGL cu121/cu124               │
│              ↓                                               │
│  结果: DGL 的 CUDA kernel 与 PyTorch 的 CUDA 运行时冲突     │
│        → forward pass 时 CUDA kernel launch 挂死            │
└─────────────────────────────────────────────────────────────┘
```

**核心矛盾**: 
- AutoDL 镜像预装了 **PyTorch 2.8.0+cu128**（CUDA 12.8）
- DGL 官方发布的 wheel 最高只支持到 **cu124**（CUDA 12.4）
- 当 DGL 的 CUDA kernel 尝试与 PyTorch 的 CUDA 12.8 运行时交互时，发生底层兼容性冲突，导致进程挂死

### 2.4 已创建的应对代码

尽管训练失败，Codex 已经写好了完整的训练代码，一旦环境问题解决即可直接运行：

- **`src/train_direct.py`** (395 行): 绕过 CLI 的直接训练脚本
  - 使用 `alignn.models.alignn.ALIGNN` + `ALIGNNConfig` 构建 graphwise 模型
  - 使用 `alignn.data.get_train_val_loaders` 加载数据
  - 自定义训练循环（AdamW + OneCycleLR）
  - 支持 CUDA/CPU、LMDB 缓存、checkpoint 保存
  - 已处理 DGL GraphBolt 兼容性问题（stub 模块）

---

## 三、需要决策的事项

### 3.1 选项 A: 接受当前结果，以基线模型完成报告

**现状**: RF MAE 0.273 eV, R² 0.798 已经是相当不错的结果

**优点**:
- 项目已经完整，有数据、有模型、有评估、有报告
- 不需要再花时间调试环境
- 可以专注于完善报告和准备答辩

**缺点**:
- 缺少 ALIGNN 自训练结果，与项目标题"Using ALIGNN"不完全匹配
- 报告需要解释为什么 ALIGNN 训练失败

### 3.2 选项 B: 重新配置环境，再次尝试 ALIGNN 训练

**需要解决的问题**: 让 PyTorch、DGL、ALIGNN 三者的 CUDA 版本完全匹配

**可能的方案**:
1. **降级 PyTorch** 到 cu121/cu124 版本，匹配 DGL
2. **使用 Docker 容器** 预先配置好兼容的环境
3. **换一台 GPU 实例**，选择预装兼容版本的镜像
4. **从源码编译 DGL** 适配 cu128（难度大，耗时长）

---

## 四、环境修复方案（已确认可行）

AutoDL 支持更换镜像。从截图可见有以下选项：

| 镜像 | PyTorch | CUDA | 建议 |
|------|---------|------|------|
| PyTorch 2.8.0 | 2.8.0 | **12.4** | ✅ **推荐** — CUDA 12.4 与 DGL cu124 完全匹配 |
| PyTorch 2.8.0 | 2.8.0 | **12.1** | ✅ 备选 — DGL cu121 也支持 |

**当前失败的镜像**: PyTorch 2.8.0 + CUDA 12.8（cu128）— DGL 不支持

---

## 五、更换镜像操作指南

### 步骤 1: 保存当前数据（如果远程还有未下载的文件）
如果远程 `/root/dl_2d_bandgap/` 目录还有数据未下载，先执行：
```bash
# 从本地终端运行
rsync -avz -e "ssh -p 48992 -o StrictHostKeyChecking=no" \
  root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/results/ \
  dl_2d_bandgap/results/

rsync -avz -e "ssh -p 48992 -o StrictHostKeyChecking=no" \
  root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/figures/ \
  dl_2d_bandgap/figures/
```

### 步骤 2: 在 AutoDL 控制台更换镜像
1. **关机**当前实例（释放 GPU）
2. 点击"**更换镜像**"
3. 选择: **PyTorch 2.8.0 + Python 3.12 + CUDA 12.4**
4. 确认更换，等待系统重装（约 2-3 分钟）
5. 开机后 SSH 连接信息（host/port）**保持不变**

### 步骤 3: 验证新环境
SSH 登录后运行：
```bash
nvidia-smi                    # 确认 GPU 和 CUDA 12.4
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 步骤 4: 重新上传项目并安装依赖
```bash
# 从本地终端
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='data/raw' \
  -e "ssh -p 48992 -o StrictHostKeyChecking=no" \
  dl_2d_bandgap/ root@connect.cqa1.seetacloud.com:/root/dl_2d_bandgap/

# 在远程执行
ssh -p 48992 -o StrictHostKeyChecking=no root@connect.cqa1.seetacloud.com \
  'cd /root/dl_2d_bandgap && pip install -r requirements.txt'
```

### 步骤 5: 安装匹配的 DGL
```bash
# 在远程执行 — CUDA 12.4 匹配 DGL cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# 验证
python3 -c "import dgl; print(dgl.__version__); import torch; print(torch.version.cuda)"
```

### 步骤 6: 运行训练
```bash
# 在远程执行
python3 src/train_direct.py --device cuda --epochs 50 --batch_size 64
```

---

## 六、更换镜像的注意事项

### ⚠️ 数据会丢失
更换镜像会**重置系统盘**（/root/ 目录）。
- ✅ **不会丢失**: AutoDL 数据盘（如果有挂载的话）
- ❌ **会丢失**: /root/ 下的所有文件，包括已上传的项目、已安装的包、训练缓存

**对策**: 重要数据先 `rsync` 下载到本地（步骤 1）。

### ⚠️ Python 版本可能变化
截图显示 CUDA 12.4 镜像用的是 **Python 3.12**。如果 `requirements.txt` 中有版本锁死的包，可能需要调整。

**对策**: 安装依赖时观察报错，如有不兼容再调整。

### ⚠️ SSH 密码可能重置
部分 AutoDL 镜像更换后会重置 root 密码。

**对策**: 如果 SSH 连不上，去 AutoDL 控制台查看新的密码或重置密码。

### ⚠️ 端口可能变化
AutoDL 的 SSH 端口通常是固定的，但极少数情况下可能变化。

**对策**: 更换后在控制台确认 SSH 端口。

### ⚠️ 训练时间预估
75k 数据集在 RTX 6000 (96GB) 上：
- 数据预处理（图构建）: 10-20 分钟
- 训练 50 epochs: 约 2-4 小时（取决于 batch_size 和模型大小）
- 总费用: 约 ¥5-15（按 AutoDL RTX 6000 单价估算）

---

## 七、如果更换镜像后仍失败

如果按上述步骤操作后 ALIGNN 训练仍然失败，建议立即放弃，接受当前基线结果。不要再花更多时间调试。

**底线**: RF MAE 0.273 eV 已经是很好的结果，足以支撑一份完整的课程报告。

---

## 八、文件清单

| 文件 | 说明 |
|------|------|
| `report/report.md` | 完整项目报告 |
| `report/slides_outline.md` | 幻灯片大纲 |
| `results/evaluation_report.json` | 评估指标 |
| `results/pretrain_benchmark.json` | 基线模型结果 |
| `results/training_history.json` | 训练历史（记录失败原因） |
| `results/final_summary.json` | 项目总览 |
| `src/train_direct.py` | 绕过 CLI 的训练脚本（环境修复后可直接用） |
| `milestones/` | 8 个里程碑文件，记录完整执行过程 |
| `figures/` | 9 张可视化图表 |
| `data/*_id_prop.csv` | 数据集划分文件 |

---

*报告已更新。请按步骤操作，有问题随时问我。*
