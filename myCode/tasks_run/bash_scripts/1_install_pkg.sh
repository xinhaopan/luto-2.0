#!/bin/bash

# 确保 Conda/Mamba 路径正确加载
path="/g/data/jk53/LUTO_XH/apps"
export PATH="${path}/miniforge3/bin:$PATH"
source "${path}/miniforge3/etc/profile.d/conda.sh"

# 定义 Python 版本
py_ver='python=3.12'
py_ver_clean=$(echo $py_ver | sed 's/=//g')  # 将 "python=3.12" 转换为 "python3.12"

# 创建新的 Conda 环境并安装指定版本的 Python
if mamba env list | grep -q "^luto"; then
    echo "The 'luto' environment already exists. Skipping environment creation."
else
    echo "Creating the 'luto' environment with ${py_ver}..."
    mamba create -c conda-forge -n luto ${py_ver} -y
fi

# 激活 Conda 环境
echo "Activating the 'luto' environment..."
conda activate luto

# 检查并安装 Conda 依赖
if [[ -f requirements_conda.txt ]]; then
    echo "Installing Conda packages from requirements_conda.txt..."
    mamba install -n luto -c conda-forge -y $(cat requirements_conda.txt)
else
    echo "Warning: requirements_conda.txt not found. Skipping Conda package installation."
fi

# 检查并安装 Pip 依赖
if [[ -f requirements_pip.txt ]]; then
    echo "Installing Pip packages from requirements_pip.txt..."
    pip install --target=${path}/miniforge3/envs/luto/lib/${py_ver_clean}/site-packages $(cat requirements_pip.txt)
else
    echo "Warning: requirements_pip.txt not found. Skipping Pip package installation."
fi

echo "Environment setup complete. The 'luto' environment is ready."