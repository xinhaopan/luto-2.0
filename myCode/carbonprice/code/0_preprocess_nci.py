import os
import subprocess

# ==============================================================================
#  配置区域
# ==============================================================================
CPU_CORES = 45
MEMORY_GB = "1400GB"          # PBS 用 GB
TIME_LIMIT = "48:00:00"       # 一天
PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
CONDA_ENV_NAME = "xpluto"
SUBMISSION_SCRIPT_NAME = "submit_preprocess.pbs"


def create_and_submit_hpc_job():
    """
    动态创建 PBS 提交脚本并用 qsub 提交。
    """
    print("--- PBS 作业启动脚本 ---")

    pbs_script_content = f"""#!/bin/bash
#PBS -N carbon_preprocess
#PBS -q hugemem
#PBS -l ncpus={CPU_CORES}
#PBS -l mem={MEMORY_GB}
#PBS -l walltime={TIME_LIMIT}
#PBS -o pbs_output_$PBS_JOBID.out
#PBS -e pbs_error_$PBS_JOBID.err
#PBS -l wd

# === 关键：声明项目与存储资源 ===
#PBS -l storage=gdata/jk53

echo "========================================================="
echo "JobID: $PBS_JOBID"
echo "Host : $(hostname)"
echo "Start: $(date)"
echo "WD   : $PWD"
echo "========================================================="

# 确保在提交目录运行（-l wd 已经会这样做，这里再次确保）
cd "$PBS_O_WORKDIR" || exit 1

# --- 步骤 1: 加载你的 shell 配置（含 conda 初始化） ---
echo "source ~/.bashrc"
source ~/.bashrc || echo "警告: source ~/.bashrc 失败，继续尝试激活 conda"

# --- 步骤 2: 激活 Conda 环境 ---
echo "activating conda env: {CONDA_ENV_NAME}"
conda activate {CONDA_ENV_NAME}
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 Conda 环境 '{CONDA_ENV_NAME}'"
    exit 1
fi

echo "Python: $(which python)"
python --version
echo "---------------------------------------------------------"

# --- 步骤 3: 执行 Python 脚本 ---
echo "运行: python {PYTHON_SCRIPT_TO_RUN}"
python {PYTHON_SCRIPT_TO_RUN}
rc=$?

echo "========================================================="
echo "结束时间: $(date)"
echo "退出码: $rc"
echo "========================================================="
exit $rc
"""

    # 写入脚本文件
    try:
        with open(SUBMISSION_SCRIPT_NAME, "w") as f:
            f.write(pbs_script_content)
        os.chmod(SUBMISSION_SCRIPT_NAME, 0o755)
        print(f"✅ 已生成 PBS 提交脚本: {SUBMISSION_SCRIPT_NAME}")
    except IOError as e:
        print(f"❌ 无法创建脚本文件: {e}")
        return

    # 提交作业
    print(f"\n🚀 正在提交: qsub {SUBMISSION_SCRIPT_NAME}")
    try:
        result = subprocess.run(
            ["qsub", SUBMISSION_SCRIPT_NAME],
            capture_output=True, text=True, check=True
        )
        print("\n--- qsub 输出 ---")
        print(result.stdout.strip())
        print("---")
        user = os.environ.get("USER", "your_username")
        print(f"✅ 提交成功! 使用 'qstat -u {user}' 查看状态。")
    except subprocess.CalledProcessError as e:
        print("❌ 错误: qsub 执行失败。")
        print("--- qsub 错误信息 ---")
        print(e.stderr)
        print("---")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    create_and_submit_hpc_job()
