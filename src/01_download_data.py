"""
步骤 1: 下载 JARVIS 数据集
===========================
从 JARVIS 数据库下载材料数据 (结构 + 带隙)，作为 ALIGNN 训练/测试的数据源。

JARVIS 是 NIST 维护的公开材料数据库，包含约 60,000+ 种材料的 DFT 计算结果。
数据格式已经与 ALIGNN 兼容。

运行方式:
    conda activate dl_2d_bg
    python 01_download_data.py

预计时间: 5-10 分钟 (取决于网络速度)
"""

import os
import json
import csv
import zipfile
import urllib.request
from pathlib import Path

# ============================================================
# 配置
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# JARVIS ALIGNN 数据集下载链接 (NIST 官方)
# 包含约 53,000 种材料，每种材料包含:
# - POSCAR (晶体结构)
# - 带隙 (eV), 形成能 (eV), 体积模量 (GPa) 等性质
JARVIS_URL = "https://ndownloader.figshare.com/files/27732228"

# C2DB 2D 材料数据集 (补充用)
# 包含约 4,000 种二维材料的电子结构数据

# ============================================================
# 主流程
# ============================================================

def download_jarvis_dataset():
    """下载 JARVIS 的 ALIGNN 格式数据集"""
    zip_path = DATA_DIR / "jarvis_alignn.zip"

    if (DATA_DIR / "jarvis_dft_3d" / "id_prop.csv").exists():
        print("[SKIP] JARVIS 数据集已存在，跳过下载。")
        return

    print(f"[1/3] 正在下载 JARVIS 数据集...")
    print(f"      URL: {JARVIS_URL}")
    print(f"      大小: ~500MB")

    # 使用流式下载，显示进度
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r      下载进度: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="")

    urllib.request.urlretrieve(JARVIS_URL, zip_path, reporthook=report_progress)
    print("\n      下载完成!")

    print(f"[2/3] 正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(DATA_DIR)
    print(f"      解压完成!")

    # 清理 zip 文件
    zip_path.unlink()

    # 检查解压结果
    print(f"[3/3] 检查数据...")
    extracted_dir = DATA_DIR / "jarvis_dft_3d"
    if extracted_dir.exists():
        count = sum(1 for p in extracted_dir.glob("*.json"))
        print(f"      解压目录: {extracted_dir}")
        print(f"      材料文件数: {count}")
        if (extracted_dir / "id_prop.csv").exists():
            with open(extracted_dir / "id_prop.csv", "r") as f:
                lines = f.readlines()
            print(f"      id_prop.csv 行数: {len(lines)} (含表头)")
    print()


def explore_data_structure():
    """探索数据集结构，帮助用户理解数据格式"""
    print("=" * 60)
    print("  数据集结构探索")
    print("=" * 60)

    # 读取 id_prop.csv
    prop_file = DATA_DIR / "jarvis_dft_3d" / "id_prop.csv"
    if not prop_file.exists():
        print("[WARNING] id_prop.csv 未找到，请先运行下载。")
        return

    # 读取前几行看格式
    with open(prop_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"\n📋 id_prop.csv 格式:")
    print(f"   列名: {header}")
    print(f"   总材料数: {len(rows)}")

    # 显示前 3 行
    print(f"\n   前 3 个示例:")
    for i, row in enumerate(rows[:3]):
        print(f"   [{i+1}] ID={row[0]}, 性质={row[1:]}")

    # 看一个材料结构文件
    if rows:
        material_id = rows[0][0]
        json_file = DATA_DIR / "jarvis_dft_3d" / f"{material_id}.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                structure = json.load(f)
            print(f"\n📄 材料结构文件示例 ({material_id}.json):")
            print(f"   键: {list(structure.keys())}")
            if "atomic_numbers" in structure:
                print(f"   原子数: {len(structure['atomic_numbers'])}")
            if "lattice" in structure:
                print(f"   晶格: {structure['lattice']}")
            if "coords" in structure:
                print(f"   坐标数: {len(structure['coords'])}")

    # 统计带隙分布
    print(f"\n📊 带隙 (bandgap) 分布统计:")
    print(f"   假设第 {header.index('optb88vdw_bandgap') + 1} 列为带隙")

    try:
        bg_idx = header.index("optb88vdw_bandgap")
        bg_values = []
        for row in rows:
            try:
                bg_values.append(float(row[bg_idx]))
            except (ValueError, IndexError):
                continue

        import numpy as np
        bg_arr = np.array(bg_values)
        print(f"   有效数据点: {len(bg_arr)}")
        print(f"   带隙范围: [{bg_arr.min():.3f}, {bg_arr.max():.3f}] eV")
        print(f"   平均带隙: {bg_arr.mean():.3f} eV")
        print(f"   中位数: {np.median(bg_arr):.3f} eV")
        print(f"   带隙 = 0 (金属): {(bg_arr < 0.01).sum()} 个 ({(bg_arr < 0.01).sum()/len(bg_arr)*100:.1f}%)")
        print(f"   带隙 > 0 (半导体/绝缘体): {(bg_arr >= 0.01).sum()} 个 ({(bg_arr >= 0.01).sum()/len(bg_arr)*100:.1f}%)")

    except ValueError as e:
        print(f"   [ERROR] 无法解析带隙数据: {e}")
        print(f"   可用列: {header}")

    print()
    return header, rows


if __name__ == "__main__":
    print("=" * 60)
    print("  步骤 1: 下载 JARVIS 材料数据集")
    print("=" * 60)
    print()

    download_jarvis_dataset()
    header, rows = explore_data_structure()

    print("=" * 60)
    print("  下载完成! 下一步: python 02_explore_data.py")
    print("=" * 60)
