"""
步骤 3: 使用 ALIGNN 预训练模型进行预测
=========================================
利用 ALIGNN 官方提供的预训练模型，直接对测试集进行带隙预测。
这是 "零训练" 基准 — 直接展示预训练模型的能力。

运行方式:
    conda activate dl_2d_bg
    python 03_predict_premodel.py

预计时间: 5-15 分钟 (取决于测试集大小)
"""

import os
import json
import csv
import time
from pathlib import Path
import numpy as np

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_test_set():
    """加载测试集"""
    test_csv = DATA_DIR / "test_id_prop.csv"
    struct_dir = DATA_DIR / "jarvis_dft_3d"

    if not test_csv.exists():
        print("[ERROR] 测试集文件不存在，请先运行 02_explore_data.py")
        return None, None, None

    materials = []
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = row["id"]
            target = float(row["target"])
            # 检查结构文件是否存在
            json_file = struct_dir / f"{jid}.json"
            if json_file.exists():
                materials.append({
                    "jid": jid,
                    "target": target,
                    "struct_file": str(json_file),
                })

    print(f"📦 加载测试集: {len(materials)} 个材料")
    return materials, struct_dir, test_csv


def predict_with_alignn_premodel(materials, struct_dir):
    """使用 ALIGNN 预训练模型进行预测"""
    print("\n" + "=" * 60)
    print("  方法 1: ALIGNN 预训练模型 (Pre-trained ALIGNN)")
    print("=" * 60)

    try:
        from alignn.models.alignn import alignn
        from alignn.config import TrainingConfig
        from jarvis.core.atoms import Atoms
        from jarvis.db.jsonutils import loadjson
        import torch
    except ImportError as e:
        print(f"[ERROR] 无法导入 ALIGNN: {e}")
        print("       请先运行 setup_env.sh 安装依赖")
        return None

    # 加载预训练配置
    # ALIGNN 预训练模型的默认路径
    model_name = "alignn_ffdb"
    print(f"\n🔍 查找预训练模型: {model_name}")

    try:
        from alignn import pretrained
        pretrained_model = pretrained.get_alignn_ffdb_model()
        print(f"✅ 预训练模型加载成功!")
        model = pretrained_model
        model.eval()
    except Exception as e:
        print(f"[WARNING] 无法加载预训练模型: {e}")
        print("         将尝试手动加载...")
        try:
            # 手动下载模型
            config = TrainingConfig(
                model="alignn",
                output_dir=str(RESULTS_DIR / "pretrained_run"),
                target="optb88vdw_bandgap",
                epoch=1,
            )
            model = alignn(config)
            # 下载预训练权重
            model_path = pretrained.get_alignn_ffdb_path()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            print(f"✅ 手动加载成功!")
        except Exception as e2:
            print(f"[ERROR] 预训练模型加载失败: {e2}")
            print("       请检查网络连接，ALIGNN 需要下载预训练权重。")
            print("       跳过预训练模型预测，将直接进入自训练步骤。")
            return None

    # 进行预测
    print(f"\n🔬 开始预测 {len(materials)} 个材料的带隙...")
    predictions = []
    targets = []
    jids = []

    device = torch.device("cpu")  # 使用 CPU
    model = model.to(device)

    start_time = time.time()
    for i, mat in enumerate(materials):
        try:
            # 读取晶体结构
            atoms_dict = loadjson(mat["struct_file"])
            atoms = Atoms.from_dict(atoms_dict)

            # 使用 ALIGNN 的预测接口
            with torch.no_grad():
                pred = model.pred_from_atoms(atoms)

            predictions.append(float(pred))
            targets.append(mat["target"])
            jids.append(mat["jid"])

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"   进度: {i+1}/{len(materials)} ({rate:.1f} 材料/秒)")

        except Exception as e:
            # 跳过预测失败的材料
            if i < 5:
                print(f"   [WARN] 材料 {mat['jid']} 预测失败: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n✅ 预测完成! 共 {len(predictions)} 个材料, 耗时 {elapsed:.1f} 秒")

    results = {
        "jids": jids,
        "targets": targets,
        "predictions": predictions,
        "elapsed_seconds": elapsed,
        "method": "ALIGNN-pretrained",
        "model_name": model_name,
    }

    return results


def predict_with_simple_baseline(materials):
    """简单基线: 基于成分描述符的随机森林"""
    print("\n" + "=" * 60)
    print("  方法 2: 成分描述符 + 随机森林 (Baseline)")
    print("=" * 60)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError:
        print("[WARNING] matminer/sklearn 未安装，跳过基线预测")
        return None

    from jarvis.core.atoms import Atoms
    from jarvis.db.jsonutils import loadjson

    # 提取成分描述符
    print("   提取成分特征...")
    features = []
    targets = []
    jids = []
    valid_materials = []

    for mat in materials:
        try:
            atoms_dict = loadjson(mat["struct_file"])
            atoms = Atoms.from_dict(atoms_dict)
            comp = Composition(atoms.elements)
            feat = ElementProperty.from_preset("magpie").featurize(comp)
            features.append(feat)
            targets.append(mat["target"])
            jids.append(mat["jid"])
            valid_materials.append(mat)
        except Exception:
            continue

    if len(features) < 10:
        print(f"   [ERROR] 有效特征太少: {len(features)}")
        return None

    X = np.array(features)
    y = np.array(targets)
    print(f"   特征维度: {X.shape}")
    print(f"   有效样本数: {len(y)}")

    # 简单 80/20 划分
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, j_train, j_test = train_test_split(
        X, y, jids, test_size=0.2, random_state=RANDOM_SEED
    )

    # 训练随机森林
    print("   训练随机森林模型...")
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    # 预测
    y_pred = rf.predict(X_test)

    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n   📊 随机森林结果:")
    print(f"   MAE = {mae:.4f} eV")
    print(f"   RMSE = {rmse:.4f} eV")
    print(f"   R² = {r2:.4f}")
    print(f"   训练时间: {train_time:.1f} 秒")

    # 特征重要性
    try:
        featurizer = ElementProperty.from_preset("magpie")
        feat_names = featurizer.feature_labels()
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"\n   📋 Top 10 重要特征:")
        for idx in top_idx:
            print(f"     {feat_names[idx]}: {importances[idx]:.4f}")
    except Exception:
        pass

    results = {
        "jids": j_test,
        "targets": y_test.tolist(),
        "predictions": y_pred.tolist(),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_time": train_time,
        "method": "RandomForest+Magpie",
    }

    return results


def save_pretrain_results(alignn_results, rf_results):
    """保存预测结果"""
    all_results = {}

    if alignn_results:
        all_results["alignn_pretrained"] = {
            "method": "ALIGNN (Pre-trained)",
            "n_predictions": len(alignn_results["predictions"]),
            "elapsed_seconds": alignn_results["elapsed_seconds"],
        }

        # 计算指标
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_true = np.array(alignn_results["targets"])
        y_pred = np.array(alignn_results["predictions"])
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        all_results["alignn_pretrained"].update({
            "MAE_eV": round(mae, 4),
            "RMSE_eV": round(rmse, 4),
            "R2": round(r2, 4),
        })
        print(f"\n📊 ALIGNN 预训练模型性能:")
        print(f"   MAE  = {mae:.4f} eV")
        print(f"   RMSE = {rmse:.4f} eV")
        print(f"   R²   = {r2:.4f}")

        # 保存详细结果
        np.savez(
            RESULTS_DIR / "pretrained_predictions.npz",
            jids=alignn_results["jids"],
            targets=y_true,
            predictions=y_pred,
        )

    if rf_results:
        all_results["random_forest"] = {
            "method": "Random Forest + Magpie Descriptors",
            "MAE_eV": round(rf_results["mae"], 4),
            "RMSE_eV": round(rf_results["rmse"], 4),
            "R2": round(rf_results["r2"], 4),
        }

    # 保存汇总
    with open(RESULTS_DIR / "pretrain_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 结果保存至: {RESULTS_DIR / 'pretrain_benchmark.json'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  步骤 3: 预训练模型预测")
    print("=" * 60)
    print()

    materials, struct_dir, test_csv = load_test_set()
    if materials is None:
        exit(1)

    # 方法 1: ALIGNN 预训练模型
    alignn_results = predict_with_alignn_premodel(materials, struct_dir)

    # 方法 2: 随机森林基线
    rf_results = predict_with_simple_baseline(materials)

    # 保存结果
    save_pretrain_results(alignn_results, rf_results)

    print()
    print("=" * 60)
    print("  预测完成! 下一步: python 04_train_model.py")
    print("=" * 60)
