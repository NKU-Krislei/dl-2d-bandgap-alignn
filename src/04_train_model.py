"""
步骤 4: 训练 ALIGNN 模型
=========================
在 JARVIS 数据集上训练 ALIGNN 模型，预测材料带隙。
这是项目的核心实验步骤。

运行方式:
    conda activate dl_2d_bg
    python 04_train_model.py

预计时间:
    - 小规模训练 (1000 样本): 5-15 分钟 (CPU)
    - 全量训练 (40000 样本): 2-6 小时 (CPU)

建议: 先用小规模数据跑通流程，确认无误后再全量训练。
"""

import os
import json
import time
from pathlib import Path
import numpy as np

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42


def train_alignn_model(
    n_epochs=100,
    batch_size=64,
    learning_rate=0.001,
    use_small_dataset=False,
):
    """训练 ALIGNN 模型"""
    print("=" * 60)
    print("  训练 ALIGNN 图神经网络模型")
    print("=" * 60)
    print()

    try:
        from alignn.models.alignn import alignn
        from alignn.config import TrainingConfig
        from alignn.data import get_train_val_loaders
        import torch
        from torch import nn
    except ImportError as e:
        print(f"[ERROR] 无法导入 ALIGNN: {e}")
        print("       请先运行 setup_env.sh 安装依赖")
        return None

    # 选择数据集
    if use_small_dataset:
        train_file = DATA_DIR / "train_small_id_prop.csv"
        val_file = DATA_DIR / "val_small_id_prop.csv"
        print("📌 使用小规模数据集 (快速测试)")
    else:
        train_file = DATA_DIR / "train_id_prop.csv"
        val_file = DATA_DIR / "val_id_prop.csv"
        print("📌 使用全量数据集")

    if not train_file.exists():
        print(f"[ERROR] 训练文件不存在: {train_file}")
        print("       请先运行 02_explore_data.py")
        return None

    # 设置训练配置
    config = TrainingConfig(
        # 模型配置
        model="alignn",
        alignn_layers=3,       # ALIGNN 层数 (消息传递迭代次数)
        gcn_layers=3,          # GCN 层数
        atom_input_features=92, # 原子特征维度 (magpie 描述符)
        # 训练配置
        output_dir=str(RESULTS_DIR / "alignn_training"),
        target="target",
        random_seed=RANDOM_SEED,
        # 优化器配置
        optim="AdamW",
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=n_epochs,
        weight_decay=0.01,
        # 损失函数
        criterion="mse",
        # 数据配置
        distributed=False,
        n_workers=0,           # macOS 下设为 0 避免 multiprocessing 问题
        # 混合精度训练 (加速)
        amp=False,
        # 日志
        store_checkpoint=True,
        save_dataloader=False,
    )

    print(f"\n📋 训练配置:")
    print(f"   模型: ALIGNN ({config.alignn_layers} ALIGNN 层 + {config.gcn_layers} GCN 层)")
    print(f"   训练轮次: {n_epochs}")
    print(f"   批次大小: {batch_size}")
    print(f"   学习率: {learning_rate}")
    print(f"   输出目录: {config.output_dir}")
    print()

    # 创建数据加载器
    print("📦 加载数据...")
    try:
        train_loader, val_loader = get_train_val_loaders(
            train_path=str(train_file),
            val_path=str(val_file),
            config=config,
        )
        print(f"   训练样本: {len(train_loader.dataset)}")
        print(f"   验证样本: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        print("       可能的原因: 结构文件路径不匹配")
        print("       正在尝试使用 ALIGNN 标准数据格式...")
        return train_alignn_standard(config)

    # 初始化模型
    print("\n🏗️  初始化 ALIGNN 模型...")
    model = alignn(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=learning_rate * 0.01
    )
    loss_fn = nn.MSELoss()

    device = torch.device("cpu")
    model = model.to(device)

    # 训练循环
    print(f"\n🏋️  开始训练...")
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "learning_rate": []}

    best_val_loss = float("inf")
    best_model_state = None

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # === 训练 ===
        model.train()
        train_losses = []
        for batch in train_loader:
            graph, target, *_ = batch
            graph = graph.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(graph)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # === 验证 ===
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                graph, target, *_ = batch
                graph = graph.to(device)
                target = target.to(device)

                output = model(graph)
                loss = loss_fn(output, target)

                val_losses.append(loss.item())
                val_preds.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())

        # 计算指标
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 日志
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            elapsed = time.time() - start_time
            print(f"   Epoch [{epoch:3d}/{n_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} eV | "
                  f"Time: {elapsed:.0f}s")

        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n✅ 训练完成! 总耗时: {total_time:.0f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    print(f"   最终验证 MAE: {history['val_mae'][-1]:.4f} eV")

    # 保存结果
    results = {
        "model": "ALIGNN",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_params": n_params,
        "best_val_loss": round(best_val_loss, 6),
        "final_val_mae": round(history["val_mae"][-1], 6),
        "training_time_seconds": round(total_time, 1),
        "history": history,
    }

    # 保存模型
    model_save_path = RESULTS_DIR / "alignn_training" / "best_model.pt"
    os.makedirs(model_save_path.parent, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"   最佳模型保存至: {model_save_path}")

    # 保存训练历史
    history_path = RESULTS_DIR / "alignn_training" / "training_history.json"
    with open(history_path, "w") as f:
        # 转换 numpy 类型
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)

    return results


def train_alignn_standard(config):
    """使用 ALIGNN 的标准训练接口（备选方案）"""
    print("\n🔄 尝试使用 ALIGNN 标准训练接口...")

    try:
        from alignn.train_run import train_dgl
        import subprocess

        # 构建 ALIGNN 命令行参数
        cmd = [
            "alignn_train_finetune",
            f"--config_name=alignn_example",
            f"--root_dir={DATA_DIR}",
            f"--epochs={config.epochs}",
            f"--batch_size={config.batch_size}",
            f"--learning_rate={config.learning_rate}",
            f"--output_dir={config.output_dir}",
        ]

        print(f"   运行命令: {' '.join(cmd)}")
        print(f"   [提示] 如果此命令失败，请参考 ALIGNN 官方文档:")
        print(f"   https://github.com/usnistgov/alignn")

    except Exception as e:
        print(f"   [ERROR] 标准接口也失败了: {e}")
        print("   建议:")
        print("   1. 检查 ALIGNN 版本: pip show alignn")
        print("   2. 查看 ALIGNN 官方教程和示例")
        print("   3. 使用 Colab notebook: https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/Training_ALIGNN_model_example.ipynb")

    return None


if __name__ == "__main__":
    print("=" * 60)
    print("  步骤 4: 训练 ALIGNN 模型")
    print("=" * 60)
    print()

    # 先用小数据集快速验证流程
    # 如果成功，可以修改参数进行全量训练
    results = train_alignn_model(
        n_epochs=50,           # 先用 50 轮快速验证
        batch_size=32,
        learning_rate=0.001,
        use_small_dataset=True,  # 使用小数据集测试
    )

    if results:
        print(f"\n{'='*60}")
        print(f"  训练结果摘要")
        print(f"{'='*60}")
        print(f"  最终验证 MAE: {results['final_val_mae']:.4f} eV")
        print(f"  训练时间: {results['training_time_seconds']:.0f} 秒")
        print(f"{'='*60}")
        print()
        print("  💡 提示: 如果结果看起来合理，可以增加 epochs 和使用全量数据")
        print("     修改脚本中的 n_epochs=100, use_small_dataset=False")
    else:
        print("\n⚠️  训练未成功完成。")
        print("    请检查错误信息，或尝试使用 ALIGNN 的命令行工具。")
        print("    参考: https://github.com/usnistgov/alignn")

    print()
    print("=" * 60)
    print("  下一步: python 05_evaluate.py")
    print("=" * 60)
