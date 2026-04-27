#!/usr/bin/env python3
"""
一键运行脚本 — 数据探索 + 可视化 (不需要 ALIGNN 安装即可运行)
==============================================================
本脚本可以直接在当前环境中运行，生成项目报告所需的基础图表。
无需 ALIGNN/PyTorch 安装。

运行方式:
    python run_quick.py

生成:
    - figures/data_exploration.png: 数据分布图
    - figures/alignn_overview.png: ALIGNN 架构概览
    - figures/crystal_graph_demo.png: 晶体图表示示例
    - figures/physics_context.png: 物理背景图
"""

import sys
import os
from pathlib import Path

# 确保在项目根目录运行
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    print("=" * 60)
    print("  DL for 2D Materials Band Gap Prediction")
    print("  Quick Start Script")
    print("=" * 60)
    print()

    # 步骤 1: 下载 (如果还没有)
    data_dir = PROJECT_ROOT / "data" / "jarvis_dft_3d"
    if not data_dir.exists():
        print("📦 数据集尚未下载。正在下载...")
        print("   (首次下载约 500MB，需要几分钟)")
        from download_data import download_jarvis_dataset
        download_jarvis_dataset()
    else:
        print("✅ 数据集已存在，跳过下载。")

    # 步骤 2: 数据探索 (生成数据分布图)
    print("\n📊 正在生成数据探索图...")
    try:
        from explore_data import load_jarvis_data, filter_valid_bandgap, plot_bandgap_distribution
        rows, header, struct_dir = load_jarvis_data()
        materials = filter_valid_bandgap(rows, header)
        plot_bandgap_distribution(materials)
    except Exception as e:
        print(f"   [WARNING] 数据探索失败: {e}")

    # 步骤 3: 概念图 (不需要数据)
    print("\n🎨 正在生成概念图...")
    try:
        from visualize import (
            create_overview_figure,
            create_crystal_graph_demo,
            create_physics_context_figure,
        )
        create_overview_figure()
        create_crystal_graph_demo()
        create_physics_context_figure()
    except Exception as e:
        print(f"   [WARNING] 概念图生成失败: {e}")

    # 完成
    figures_dir = PROJECT_ROOT / "figures"
    print(f"\n{'='*60}")
    print(f"  ✅ 快速可视化完成!")
    print(f"  📁 生成的图片: {figures_dir}")
    print(f"")
    print(f"  下一步:")
    print(f"    1. 安装 ALIGNN: bash setup_env.sh")
    print(f"    2. 训练模型: conda activate dl_2d_bg && cd src && python 04_train_model.py")
    print(f"    3. 生成完整评估: python 05_evaluate.py && python 06_visualize.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
