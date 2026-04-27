"""
步骤 5: 模型评估
=================
对训练好的模型进行全面评估，计算多种指标，生成评估报告。

运行方式:
    conda activate dl_2d_bg
    python 05_evaluate.py
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def compute_metrics(y_true, y_pred, label=""):
    """计算回归评估指标"""
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        mean_absolute_percentage_error
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_pred - y_true

    metrics = {
        "MAE (eV)": mean_absolute_error(y_true, y_pred),
        "RMSE (eV)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_true + residuals) * 100,
        "Max Error (eV)": np.max(np.abs(residuals)),
        "Mean Error (eV)": np.mean(residuals),
        "Std Error (eV)": np.std(residuals),
    }

    print(f"\n📊 {label} 评估结果:")
    print(f"   {'指标':<20} {'值':>10}")
    print(f"   {'-'*32}")
    for k, v in metrics.items():
        print(f"   {k:<20} {v:>10.4f}")

    return metrics


def plot_predictions(y_true, y_pred, method_name, filename=None):
    """绘制预测值 vs 真实值散点图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_true

    # 1. 散点图 (带对角线)
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, c='#2196F3', s=8, alpha=0.3, edgecolors='none')
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    ax1.set_xlabel('DFT Band Gap (eV)', fontsize=12)
    ax1.set_ylabel('Predicted Band Gap (eV)', fontsize=12)
    ax1.set_title(f'(a) {method_name}: Predicted vs. Actual', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.2, max_val)
    ax1.set_ylim(-0.2, max_val)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. 残差分布图
    ax2 = axes[1]
    ax2.hist(residuals, bins=60, color='#4CAF50', edgecolor='white', alpha=0.8, density=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=1.5,
                label=f'Mean={np.mean(residuals):.3f} eV')
    ax2.set_xlabel('Residual (Predicted - Actual) (eV)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'(b) Residual Distribution', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 误差 vs 真实值
    ax3 = axes[2]
    abs_errors = np.abs(residuals)
    ax3.scatter(y_true, abs_errors, c=abs_errors, cmap='hot_r', s=8, alpha=0.3, edgecolors='none')
    ax3.set_xlabel('DFT Band Gap (eV)', fontsize=12)
    ax3.set_ylabel('Absolute Error (eV)', fontsize=12)
    ax3.set_title(f'(c) Absolute Error vs. Band Gap', fontsize=13)
    ax3.axhline(np.mean(abs_errors), color='blue', linestyle='--',
                label=f'MAE={np.mean(abs_errors):.3f} eV')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is None:
        filename = f"evaluation_{method_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📈 评估图保存至: {FIGURES_DIR / filename}")


def plot_training_history(history_path=None, history_dict=None):
    """绘制训练历史曲线"""
    if history_dict is None and history_path:
        with open(history_path, "r") as f:
            history_dict = json.load(f)

    if history_dict is None:
        print("[WARNING] 无训练历史数据，跳过绘制。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history_dict["train_loss"]) + 1)

    # 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, history_dict["train_loss"], 'b-', linewidth=1.5, label='Training Loss')
    ax1.plot(epochs, history_dict["val_loss"], 'r-', linewidth=1.5, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('(a) Training & Validation Loss', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # MAE 曲线
    ax2 = axes[1]
    ax2.plot(epochs, history_dict["val_mae"], 'g-', linewidth=1.5, label='Validation MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE (eV)', fontsize=12)
    ax2.set_title('(b) Validation MAE', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 训练历史图保存至: {FIGURES_DIR / 'training_history.png'}")


def plot_comparison_benchmark(results_dir):
    """对比不同方法的性能"""
    benchmark_file = results_dir / "pretrain_benchmark.json"
    if not benchmark_file.exists():
        return

    with open(benchmark_file, "r") as f:
        benchmarks = json.load(f)

    if len(benchmarks) < 2:
        return

    methods = list(benchmarks.keys())
    maes = [benchmarks[m].get("MAE_eV", 0) for m in methods]
    rmses = [benchmarks[m].get("RMSE_eV", 0) for m in methods]
    r2s = [benchmarks[m].get("R2", 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE 对比
    bars1 = axes[0].bar(range(len(methods)), maes, color=['#2196F3', '#FF9800', '#4CAF50'])
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([benchmarks[m]["method"] for m in methods], rotation=15, ha='right', fontsize=10)
    axes[0].set_ylabel('MAE (eV)', fontsize=12)
    axes[0].set_title('(a) Mean Absolute Error', fontsize=13)
    for bar, val in zip(bars1, maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', fontsize=10)

    # RMSE 对比
    bars2 = axes[1].bar(range(len(methods)), rmses, color=['#2196F3', '#FF9800', '#4CAF50'])
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels([benchmarks[m]["method"] for m in methods], rotation=15, ha='right', fontsize=10)
    axes[1].set_ylabel('RMSE (eV)', fontsize=12)
    axes[1].set_title('(b) Root Mean Square Error', fontsize=13)
    for bar, val in zip(bars2, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', fontsize=10)

    # R² 对比
    bars3 = axes[2].bar(range(len(methods)), r2s, color=['#2196F3', '#FF9800', '#4CAF50'])
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels([benchmarks[m]["method"] for m in methods], rotation=15, ha='right', fontsize=10)
    axes[2].set_ylabel('R² Score', fontsize=12)
    axes[2].set_title('(c) R² Score', fontsize=13)
    for bar, val in zip(bars3, r2s):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 方法对比图保存至: {FIGURES_DIR / 'method_comparison.png'}")


def run_full_evaluation():
    """运行完整评估流程"""
    print("=" * 60)
    print("  模型评估")
    print("=" * 60)
    print()

    all_metrics = {}

    # === 评估预训练模型结果 ===
    pretrained_file = RESULTS_DIR / "pretrained_predictions.npz"
    if pretrained_file.exists():
        data = np.load(pretrained_file, allow_pickle=True)
        y_true = data["targets"]
        y_pred = data["predictions"]

        metrics = compute_metrics(y_true, y_pred, "ALIGNN (Pre-trained)")
        all_metrics["ALIGNN (Pre-trained)"] = metrics
        plot_predictions(y_true, y_pred, "ALIGNN_Pretrained", "eval_pretrained.png")

    # === 评估自训练模型 ===
    train_history_path = RESULTS_DIR / "alignn_training" / "training_history.json"
    if train_history_path.exists():
        with open(train_history_path, "r") as f:
            history = json.load(f)
        plot_training_history(history_dict=history)
        final_mae = history["val_mae"][-1]
        print(f"\n📊 自训练 ALIGNN 最终验证 MAE: {final_mae:.4f} eV")
        all_metrics["ALIGNN (Self-trained)"] = {
            "MAE (eV)": final_mae,
            "Final Epoch": len(history["train_loss"]),
        }

    # === 绘制方法对比图 ===
    plot_comparison_benchmark(RESULTS_DIR)

    # === 保存评估报告 ===
    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n💾 评估报告保存至: {report_path}")
    print()

    return all_metrics


if __name__ == "__main__":
    metrics = run_full_evaluation()

    print("=" * 60)
    print("  评估完成! 下一步: python 06_visualize.py")
    print("=" * 60)
