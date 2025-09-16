import os
import subprocess
import shutil

# ==============================================================================
#  配置区域 (已大幅简化)
# ==============================================================================

CPU_CORES = 90
MEMORY_GB = "1440G"
TIME_LIMIT = "0-720:00:00"

# 要运行的Python脚本
PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
# 要激活的Conda环境名称
CONDA_ENV_NAME = "xpluto-fixed"

# 生成的提交脚本名称
SUBMISSION_SCRIPT_NAME = "submit_preprocess.sh"

# ==============================================================================

def create_and_submit_hpc_job():
    """
    动态创建Slurm提交脚本并使用sbatch提交作业。
    【采纳您的建议】: 使用 `source ~/.bashrc` 来初始化环境，简单可靠。
    """
    print("--- HPC作业启动脚本 (根据您的建议优化) ---")

    # 1. 定义Slurm提交脚本的内容
    slurm_script_content = f"""#!/bin/bash

#SBATCH --job-name=carbon_preprocess
#SBATCH --partition=mem
#SBATCH --nodelist=hpc-fc-b-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_CORES}
#SBATCH --mem={MEMORY_GB}
#SBATCH --time={TIME_LIMIT}
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err

echo "========================================================="
echo "作业ID: $SLURM_JOB_ID"
echo "开始时间: $(date)"
echo "========================================================="

# --- 步骤 1: 加载您的个人 shell 配置文件 (推荐方法) ---
# 这会加载 Conda 初始化以及您所有的个人设置。
echo "正在加载用户的 shell 配置文件: source ~/.bashrc"
source ~/.bashrc
if [ $? -ne 0 ]; then
    echo "警告: 'source ~/.bashrc' 执行时遇到问题，但这可能不影响后续步骤。"
fi

# --- 步骤 2: 激活您的 xpluto 环境 ---
echo "正在激活 Conda 环境: {CONDA_ENV_NAME}"
conda activate {CONDA_ENV_NAME}
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 Conda 环境 '{CONDA_ENV_NAME}'。"
    echo "请确认 'conda init' 已在您的 ~/.bashrc 文件中正确配置。"
    exit 1
fi

echo "Conda 环境已激活。当前 Python 路径: $(which python)"
echo "---------------------------------------------------------"

# --- 步骤 3: 运行您的 Python 脚本 ---
echo "开始执行 Python 脚本: {PYTHON_SCRIPT_TO_RUN}"
python {PYTHON_SCRIPT_TO_RUN}

echo "========================================================="
echo "Python 脚本执行完毕。"
echo "结束时间: $(date)"
echo "========================================================="
"""

    # 2. 创建并提交作业 (这部分逻辑不变)
    try:
        with open(SUBMISSION_SCRIPT_NAME, "w") as f:
            f.write(slurm_script_content)
        os.chmod(SUBMISSION_SCRIPT_NAME, 0o755)
        print(f"✅ 成功创建HPC提交脚本: '{SUBMISSION_SCRIPT_NAME}'")
    except IOError as e:
        print(f"❌ 错误: 无法创建脚本文件. {e}")
        return

    print(f"\n🚀 正在使用 'sbatch {SUBMISSION_SCRIPT_NAME}' 提交作业...")
    try:
        result = subprocess.run(
            ["sbatch", SUBMISSION_SCRIPT_NAME],
            capture_output=True, text=True, check=True
        )
        print("\n--- sbatch 命令输出 ---")
        print(result.stdout.strip())
        print("---")
        print(f"✅ 作业已成功提交! 您可以使用 'squeue -u {os.getlogin()}' 命令查看作业状态。")
    except subprocess.CalledProcessError as e:
        print("❌ 错误: 'sbatch' 命令执行失败。")
        print("--- sbatch 返回的错误信息 ---")
        print(e.stderr)
        print("---")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    create_and_submit_hpc_job()