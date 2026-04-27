"""
步骤 2: 数据探索与预处理
===========================
探索 JARVIS 数据集，筛选 2D 相关材料，准备训练/测试数据。

运行方式:
    conda activate dl_2d_bg
    python 02_explore_data.py

输出:
    - results/data_exploration.png: 数据分布图
    - data/train.json / val.json / test.json: ALIGNN 训练格式
    - data/train_id_prop.csv / val_id_prop.csv / test_id_prop.csv: 备用格式
"""

import os
import json
import csv
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 设置 matplotlib 中文字体 (macOS)
plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_jarvis_data():
    """加载 JARVIS 数据集"""
    prop_file = DATA_DIR / "jarvis_dft_3d" / "id_prop.csv"
    struct_dir = DATA_DIR / "jarvis_dft_3d"

    materials = []
    with open(prop_file, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    print(f"📦 加载了 {len(rows)} 种材料的元数据")
    print(f"   列名: {header[:5]}... (共 {len(header)} 列)")
    return rows, header, struct_dir


def filter_valid_bandgap(rows, header):
    """筛选有效带隙数据的材料"""
    valid = []
    invalid_count = 0

    for row in rows:
        try:
            bg = float(row.get("optb88vdw_bandgap", 0))
            if bg >= 0:  # 合法带隙值 (>= 0)
                valid.append({
                    "jid": row["jid"],
                    "bandgap": bg,
                    "formation_energy": float(row.get("optb88vdw_energy_per_atom", 0)),
                    "volume": float(row.get("optb88vdw_volume", 0)),
                    "nsites": int(row.get("nsites", 0)),
                })
            else:
                invalid_count += 1
        except (ValueError, KeyError, TypeError):
            invalid_count += 1

    print(f"✅ 有效材料: {len(valid)} (带隙 >= 0)")
    print(f"❌ 无效/负值: {invalid_count}")
    return valid


def identify_2d_materials(rows, header):
    """尝试识别可能的 2D 材料 (基于材料 ID 和晶格参数)"""
    # JARVIS 中 2D 材料通常有 "2D-" 前缀或在 JARVIS-2D 数据库中
    # 这里我们用启发式方法：nsites 较少 + 有特定关键词
    two_d_keywords = ["2D-", "JVASP-2D", "monolayer", "bilayer", "graphene",
                       "BN", "MoS2", "MoS", "WS2", "WS", "MoSe", "WSe",
                       "phosphorene", "MXene", "TMD"]

    two_d = []
    for row in rows:
        jid = row.get("jid", "")
        # 方法1: ID 匹配
        for kw in two_d_keywords:
            if kw.lower() in jid.lower():
                two_d.append(jid)
                break

    # 同时检查是否有单独的 2D 数据集
    dir2d = DATA_DIR / "jarvis_dft_2d"
    if dir2d.exists():
        prop2d = dir2d / "id_prop.csv"
        if prop2d.exists():
            print(f"   [发现] JARVIS 2D 数据集: {dir2d}")

    print(f"🔍 识别到可能的 2D 材料: {len(two_d)} 个 (基于关键词匹配)")
    return two_d


def plot_bandgap_distribution(materials, two_d_ids=None):
    """绘制带隙分布图"""
    bg = np.array([m["bandgap"] for m in materials])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 带隙直方图
    ax1 = axes[0, 0]
    ax1.hist(bg[bg > 0], bins=80, color='steelblue', edgecolor='white', alpha=0.8, label=f'n={np.sum(bg > 0)}')
    ax1.axvline(bg.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={bg.mean():.2f} eV')
    ax1.axvline(np.median(bg[bg > 0]), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(bg[bg > 0]):.2f} eV')
    ax1.set_xlabel('Band Gap (eV)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('(a) Band Gap Distribution (E_g > 0)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 10)

    # 2. 金属 vs 半导体 比例
    ax2 = axes[0, 1]
    n_metal = np.sum(bg < 0.01)
    n_semi = np.sum((bg >= 0.01) & (bg < 3.0))
    n_insulator = np.sum(bg >= 3.0)
    labels = [f'Metal\n(n={n_metal})', f'Semiconductor\n(n={n_semi})', f'Insulator\n(n={n_insulator})']
    sizes = [n_metal, n_semi, n_insulator]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11})
    ax2.set_title('(b) Material Type Distribution', fontsize=13)

    # 3. 带隙 vs 原子数
    ax3 = axes[1, 0]
    nsites = np.array([m["nsites"] for m in materials])
    scatter = ax3.scatter(nsites, bg, c=bg, cmap='viridis', s=3, alpha=0.3)
    ax3.set_xlabel('Number of Atoms in Unit Cell', fontsize=12)
    ax3.set_ylabel('Band Gap (eV)', fontsize=12)
    ax3.set_title('(c) Band Gap vs. Unit Cell Size', fontsize=13)
    plt.colorbar(scatter, ax=ax3, label='Band Gap (eV)')

    # 4. 带隙 vs 形成能
    ax4 = axes[1, 1]
    fe = np.array([m["formation_energy"] for m in materials])
    mask = (fe > -3) & (fe < 2)  # 过滤极端值
    ax4.scatter(fe[mask], bg[mask], c=bg[mask], cmap='plasma', s=3, alpha=0.3)
    ax4.set_xlabel('Formation Energy per Atom (eV)', fontsize=12)
    ax4.set_ylabel('Band Gap (eV)', fontsize=12)
    ax4.set_title('(d) Band Gap vs. Formation Energy', fontsize=13)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "data_exploration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 数据探索图保存至: {FIGURES_DIR / 'data_exploration.png'}")


def prepare_alignn_dataset(materials, train_ratio=0.8, val_ratio=0.1):
    """准备 ALIGNN 训练格式数据集"""
    # 随机打乱
    random.shuffle(materials)
    n = len(materials)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = materials[:n_train]
    val = materials[n_train:n_train + n_val]
    test = materials[n_train + n_val:]

    print(f"\n📂 数据集划分:")
    print(f"   训练集: {len(train)} ({train_ratio*100:.0f}%)")
    print(f"   验证集: {len(val)} ({val_ratio*100:.0f}%)")
    print(f"   测试集: {len(test)} {(1-train_ratio-val_ratio)*100:.0f}%")

    # 保存 id_prop.csv 格式 (ALIGNN 可直接使用)
    splits = {"train": train, "val": val, "test": test}
    for split_name, split_data in splits.items():
        csv_path = DATA_DIR / f"{split_name}_id_prop.csv"
        json_path = DATA_DIR / f"{split_name}.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "target"])
            for m in split_data:
                writer.writerow([m["jid"], f"{m['bandgap']:.6f}"])

        # 同时保存 json 格式
        with open(json_path, "w") as f:
            json.dump(split_data, f, indent=2)

        bg_vals = [m["bandgap"] for m in split_data]
        print(f"   {split_name}: E_g ∈ [{min(bg_vals):.3f}, {max(bg_vals):.3f}] eV, "
              f"mean={np.mean(bg_vals):.3f} eV")

    return train, val, test


def save_dataset_stats(materials, train, val, test):
    """保存数据集统计信息"""
    stats = {
        "total_materials": len(materials),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "bandgap_mean": float(np.mean([m["bandgap"] for m in materials])),
        "bandgap_std": float(np.std([m["bandgap"] for m in materials])),
        "bandgap_max": float(max(m["bandgap"] for m in materials)),
        "bandgap_min": float(min(m["bandgap"] for m in materials)),
        "n_metal": int(sum(1 for m in materials if m["bandgap"] < 0.01)),
        "n_semiconductor": int(sum(1 for m in materials if 0.01 <= m["bandgap"] < 3.0)),
        "n_insulator": int(sum(1 for m in materials if m["bandgap"] >= 3.0)),
    }
    stats_path = RESULTS_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📈 数据集统计保存至: {stats_path}")
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("  步骤 2: 数据探索与预处理")
    print("=" * 60)
    print()

    # 加载数据
    rows, header, struct_dir = load_jarvis_data()

    # 筛选有效数据
    materials = filter_valid_bandgap(rows, header)

    # 识别 2D 材料
    two_d_ids = identify_2d_materials(rows, header)

    # 绘制分布图
    plot_bandgap_distribution(materials)

    # 划分数据集
    train, val, test = prepare_alignn_dataset(materials)

    # 保存统计信息
    stats = save_dataset_stats(materials, train, val, test)

    print()
    print("=" * 60)
    print("  数据预处理完成! 下一步: python 03_predict_premodel.py")
    print("=" * 60)
