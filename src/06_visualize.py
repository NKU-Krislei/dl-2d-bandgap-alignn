"""
步骤 6: 综合可视化与报告生成
=============================
生成项目报告中所需的所有高质量图表，以及生成数据摘要。

运行方式:
    conda activate dl_2d_bg
    python 06_visualize.py
"""

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def create_overview_figure():
    """创建项目概览图: ALIGNN 架构示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('ALIGNN Workflow for Band Gap Prediction', fontsize=16, fontweight='bold', pad=20)

    # 方框定义
    boxes = [
        (0.3, 2.5, 2.5, 1.5, 'Crystal\nStructure\n(POSCAR)', '#E3F2FD', '#1565C0'),
        (3.5, 2.5, 2.5, 1.5, 'Crystal\nGraph\n(Atoms=Bonds)', '#E8F5E9', '#2E7D32'),
        (6.7, 2.5, 2.5, 1.5, 'Line Graph\n(Bond Angles)', '#FFF3E0', '#E65100'),
        (9.9, 2.5, 2.5, 1.5, 'ALIGNN\n(GNN Layers)', '#FCE4EC', '#C62828'),
        (13.0, 2.0, 0.8, 2.5, 'Band\nGap\n(eV)', '#F3E5F5', '#6A1B9A'),
    ]

    for x, y, w, h, text, fc, ec in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=ec)

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#555555', lw=2)
    ax.annotate('', xy=(3.5, 3.25), xytext=(2.8, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(6.7, 3.25), xytext=(6.0, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(9.9, 3.25), xytext=(9.2, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(13.0, 3.25), xytext=(12.4, 3.25), arrowprops=arrow_style)

    # 底部说明
    descriptions = [
        (1.55, 2.0, 'Input: Atomic\npositions + species'),
        (4.75, 2.0, 'Two-body\ninteractions'),
        (7.95, 2.0, 'Three-body\ninteractions'),
        (11.15, 2.0, 'Message passing\nneural network'),
        (13.4, 1.2, 'DFT-level\naccuracy'),
    ]
    for x, y, text in descriptions:
        ax.text(x, y, text, ha='center', va='top', fontsize=8, color='#666666',
                style='italic')

    # 顶部标签
    ax.text(1.55, 4.3, '① Input', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
    ax.text(4.75, 4.3, '② Representation', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')
    ax.text(7.95, 4.3, '② Representation', fontsize=10, ha='center', color='#E65100', fontweight='bold')
    ax.text(11.15, 4.3, '③ Model', fontsize=10, ha='center', color='#C62828', fontweight='bold')
    ax.text(13.4, 4.8, '④ Output', fontsize=10, ha='center', color='#6A1B9A', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "alignn_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 概览图保存至: {FIGURES_DIR / 'alignn_overview.png'}")


def create_crystal_graph_demo():
    """创建晶体图表示示例 (概念图)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) 晶体结构
    ax1 = axes[0]
    # 简化的 hBN 晶格示意图
    a = 1.0
    # B 原子
    bx = [0, a, a/2]
    by = [0, 0, a*np.sqrt(3)/2]
    # N 原子
    nitx = [0, a, a/2]
    nity = [a*np.sqrt(3)/3, a*np.sqrt(3)/3, -a*np.sqrt(3)/6]

    ax1.scatter(bx, by, c='#FF6B6B', s=200, zorder=5, label='Boron', edgecolors='darkred', linewidth=1.5)
    ax1.scatter(nitx, nity, c='#4ECDC4', s=200, zorder=5, label='Nitrogen', edgecolors='teal', linewidth=1.5)

    # 画键
    for i in range(len(bx)):
        for j in range(len(nitx)):
            dist = np.sqrt((bx[i]-nitx[j])**2 + (by[i]-nity[j])**2)
            if dist < a * 0.8:
                ax1.plot([bx[i], nitx[j]], [by[i], nity[j]], 'k-', linewidth=2, alpha=0.5)

    ax1.set_title('(a) Crystal Structure of h-BN', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.5, 1.8)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # (b) 晶体图 (Graph)
    ax2 = axes[1]
    # 节点 = 原子, 边 = 化学键
    import networkx as nx
    try:
        G = nx.Graph()
        # 添加原子节点
        for i, (x, y) in enumerate(zip(bx, by)):
            G.add_node(f"B{i}", pos=(x*2, y*2), color='red', type='B')
        for i, (x, y) in enumerate(zip(nitx, nity)):
            G.add_node(f"N{i}", pos=(x*2, y*2), color='blue', type='N')

        # 添加键边
        for i in range(len(bx)):
            for j in range(len(nitx)):
                dist = np.sqrt((bx[i]-nitx[j])**2 + (by[i]-nity[j])**2)
                if dist < a * 0.8:
                    G.add_edge(f"B{i}", f"N{j}")

        pos = nx.get_node_attributes(G, 'pos')
        colors = ['#FF6B6B' if 'B' in n else '#4ECDC4' for n in G.nodes()]
        nx.draw(G, pos, ax=ax2, node_color=colors, node_size=300,
                with_labels=True, font_weight='bold', edge_color='gray', width=2)
    except ImportError:
        ax2.text(0.5, 0.5, 'NetworkX\nnot installed', ha='center', va='center', fontsize=14)

    ax2.set_title('(b) Crystal Graph\n(Atoms = Nodes, Bonds = Edges)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # (c) Line Graph (键角信息)
    ax3 = axes[2]
    ax3.text(0.5, 0.7, 'Line Graph', ha='center', va='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.5, 'Nodes = Bonds\nEdges = Bond Angles\n(B → N → B)', ha='center', va='center', fontsize=11)
    ax3.text(0.5, 0.25, 'Captures 3-body\ninteractions!', ha='center', va='center',
             fontsize=12, color='red', fontweight='bold')

    # 画一个简单的键角示意
    ax3.annotate('', xy=(0.25, 0.15), xytext=(0.5, 0.35), arrowprops=dict(arrowstyle='->', lw=2))
    ax3.annotate('', xy=(0.75, 0.15), xytext=(0.5, 0.35), arrowprops=dict(arrowstyle='->', lw=2))
    ax3.text(0.5, 0.2, 'θ', fontsize=14, ha='center')

    ax3.set_title('(c) Line Graph\n(Bond Angle Information)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crystal_graph_demo.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 晶体图示例保存至: {FIGURES_DIR / 'crystal_graph_demo.png'}")


def create_physics_context_figure():
    """创建物理背景图: 2D 材料能带结构示例"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) 典型 2D 半导体能带 (概念示意)
    ax1 = axes[0]
    k = np.linspace(0, 4, 100)
    # 导带
    cb = 2.0 + 0.5 * np.cos(k) + 0.2 * np.sin(2*k)
    # 价带
    vb = -0.3 - 0.8 * np.cos(k) - 0.1 * np.sin(2*k)
    ax1.plot(k, cb, 'b-', linewidth=2, label='Conduction Band')
    ax1.plot(k, vb, 'r-', linewidth=2, label='Valence Band')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(k, vb, -3, alpha=0.1, color='red')
    ax1.fill_between(k, cb, 5, alpha=0.1, color='blue')
    # 标注带隙
    ax1.annotate('', xy=(2, 0), xytext=(2, 1.5),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
    ax1.text(2.3, 0.75, '$E_g$', fontsize=16, color='green', fontweight='bold')
    ax1.set_xlabel('Wave Vector k', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('(a) Band Structure of a 2D Semiconductor', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-3, 5)

    # (b) 不同材料的带隙对比
    ax2 = axes[1]
    materials = ['Graphene', 'MoS$_2$', 'h-BN', 'BP', 'MoSe$_2$']
    bandgaps = [0.0, 1.89, 5.97, 0.3, 1.55]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    bars = ax2.barh(materials, bandgaps, color=colors, edgecolor='gray', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, bandgaps):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f} eV', va='center', fontsize=10)

    ax2.set_xlabel('Band Gap (eV)', fontsize=12)
    ax2.set_title('(b) Band Gaps of Typical 2D Materials', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 7)

    # (c) ML 在材料科学中的位置
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title('(c) ML for Materials Science Pipeline', fontsize=12, fontweight='bold')

    pipeline = [
        ('Material\nDatabase', '#E3F2FD'),
        ('Feature\nExtraction', '#E8F5E9'),
        ('ML Model\n(GNN)', '#FCE4EC'),
        ('Property\nPrediction', '#F3E5F5'),
        ('Material\nDesign', '#FFF3E0'),
    ]

    for i, (text, color) in enumerate(pipeline):
        x = 0.1 + i * 0.18
        rect = mpatches.FancyBboxPatch(
            (x, 0.35), 0.15, 0.3, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='#555', linewidth=1.5,
            transform=ax3.transAxes
        )
        ax3.add_patch(rect)
        ax3.text(x + 0.075, 0.5, text, ha='center', va='center',
                 fontsize=8, fontweight='bold', transform=ax3.transAxes)

        if i < len(pipeline) - 1:
            ax3.annotate('', xy=(x + 0.17, 0.5), xytext=(x + 0.16, 0.5),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physics_context.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 物理背景图保存至: {FIGURES_DIR / 'physics_context.png'}")


def generate_summary_table():
    """生成结果汇总表"""
    print("\n" + "=" * 60)
    print("  项目结果汇总")
    print("=" * 60)

    # 加载所有可用结果
    all_results = {}

    # 预训练模型结果
    benchmark_file = RESULTS_DIR / "pretrain_benchmark.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            all_results.update(json.load(f))

    # 自训练模型结果
    history_file = RESULTS_DIR / "alignn_training" / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
        all_results["ALIGNN (Self-trained)"] = {
            "method": "ALIGNN (Self-trained)",
            "MAE_eV": round(history["val_mae"][-1], 4),
            "n_epochs": len(history["train_loss"]),
        }

    # 数据集统计
    stats_file = RESULTS_DIR / "dataset_stats.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    # 打印汇总表
    print(f"\n📊 数据集: JARVIS DFT-3D")
    print(f"   总材料数: {stats.get('total_materials', 'N/A')}")
    print(f"   金属: {stats.get('n_metal', 'N/A')} | 半导体: {stats.get('n_semiconductor', 'N/A')} | 绝缘体: {stats.get('n_insulator', 'N/A')}")

    print(f"\n📋 模型性能对比:")
    print(f"   {'方法':<35} {'MAE (eV)':<12} {'RMSE (eV)':<12} {'R²':<10}")
    print(f"   {'-'*70}")
    for key, val in all_results.items():
        name = val.get("method", key)
        mae = val.get("MAE_eV", "N/A")
        rmse = val.get("RMSE_eV", "N/A")
        r2 = val.get("R2", "N/A")
        print(f"   {name:<35} {str(mae):<12} {str(rmse):<12} {str(r2):<10}")

    # 保存汇总
    summary = {
        "dataset": stats,
        "model_results": all_results,
    }
    summary_path = RESULTS_DIR / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 汇总保存至: {summary_path}")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("  步骤 6: 综合可视化")
    print("=" * 60)
    print()

    # 生成概念图
    create_overview_figure()
    create_crystal_graph_demo()
    create_physics_context_figure()

    # 生成汇总表
    summary = generate_summary_table()

    print()
    print("=" * 60)
    print("  所有图表生成完成! 可以查看 figures/ 目录")
    print("=" * 60)
    print(f"   📁 图片目录: {FIGURES_DIR}")
    print(f"   📁 结果目录: {RESULTS_DIR}")
