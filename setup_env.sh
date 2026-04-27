#!/bin/bash
# ============================================================
# 环境搭建脚本 - DL for 2D Materials Band Gap Prediction
# 运行方式: bash setup_env.sh
# ============================================================

set -e

echo "=========================================="
echo "  Creating conda environment: dl_2d_bg"
echo "=========================================="

# 创建专用环境 (Python 3.11, 因为 ALIGNN 兼容性最好)
conda create -n gnn_2d_mat python=3.11 -y

echo ""
echo "=========================================="
echo "  Activating environment..."
echo "=========================================="

# 激活环境 (在脚本中需要 source)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dl_2d_bg

echo ""
echo "=========================================="
echo "  Installing PyTorch (Apple Silicon optimized)..."
echo "=========================================="

# macOS Apple Silicon: 使用 pip 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=========================================="
echo "  Installing ALIGNN and dependencies..."
echo "=========================================="

# 安装 ALIGNN (NIST 的图神经网络材料性质预测框架)
pip install alignn

# 安装材料科学工具包
pip install pymatgen ase matminer

# 安装数据分析和可视化
pip install pandas numpy matplotlib seaborn scikit-learn tqdm

# 安装 JARVIS tools (数据集访问)
pip install jarvis-tools

echo ""
echo "=========================================="
echo "  Verifying installation..."
echo "=========================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
import alignn
print(f'ALIGNN version: {alignn.__version__}')
import pymatgen
print(f'pymatgen version: {pymatgen.__version__}')
import matminer
print(f'matminer version: {matminer.__version__}')
print()
print('All packages installed successfully!')
"

echo ""
echo "=========================================="
echo "  Setup complete! Next steps:"
echo "=========================================="
echo "  1. conda activate dl_2d_bg"
echo "  2. cd src && python 01_download_data.py"
echo "  3. python 02_explore_data.py"
echo "  4. python 03_predict_premodel.py"
echo "  5. python 04_train_model.py"
echo "  6. python 05_evaluate.py"
echo "  7. python 06_visualize.py"
echo "=========================================="
