#!/bin/bash

# 自定义安装路径
path="/g/data/jk53/LUTO_XH/apps"
# 将 Conda 的二进制文件添加到 PATH
export PATH="${path}/miniforge3/bin:$PATH"

# 检查是否已安装 Miniforge
if [[ ! -d ${path}/miniforge3 ]]; then
    echo "Miniforge not found in ${path}/miniforge3. Installing..."
    # 下载最新的 Miniforge 安装脚本
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    # 安装到自定义路径
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${path}/miniforge3
    # 删除安装脚本
    rm Miniforge3-$(uname)-$(uname -m).sh
else
    echo "Miniforge already installed in ${path}/miniforge3."
fi

# 初始化 mamba 或 conda
if [[ -x "$(command -v mamba)" ]]; then
    echo "Initializing mamba..."
    mamba init
else
    echo "Mamba not found, initializing conda..."
    conda init
fi

# 提示用户重启 shell 以激活环境
echo "Initialization complete. Please restart your shell or source your shell configuration file."