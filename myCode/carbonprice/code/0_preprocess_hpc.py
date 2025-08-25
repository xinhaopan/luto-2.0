import os
import subprocess
import shutil

# ==============================================================================
#  配置区域: 请根据您的HPC环境修改下面的占位符
# ==============================================================================


# 【可选修改】申请的计算资源
CPU_CORES = 13          # 申请的CPU核心数
MEMORY_GB = "208G"      # 申请的内存 (例如 "64G", "128G")
TIME_LIMIT = "0-08:00:00" # 作业运行时间上限 (天-时:分:秒)

# 作业和脚本文件名
PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
SUBMISSION_SCRIPT_NAME = "submit_preprocess.sh"

# ==============================================================================

def create_and_submit_hpc_job():
    """
    动态创建Slurm提交脚本并使用sbatch提交作业。
    """
    print("--- HPC作业启动脚本 ---")

    # 1. 定义Slurm提交脚本的内容
    #    使用f-string将上面的配置动态插入脚本中
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
echo "作业名称: $SLURM_JOB_NAME"
echo "运行节点: $SLURMD_NODENAME"
echo "申请核心数: $SLURM_CPUS_PER_TASK"
echo "开始时间: $(date)"
echo "========================================================="



# 运行您的Python脚本
echo "开始执行 Python 脚本: {PYTHON_SCRIPT_TO_RUN}"
python {PYTHON_SCRIPT_TO_RUN}

echo "========================================================="
echo "Python 脚本执行完毕。"
echo "结束时间: $(date)"
echo "========================================================="
"""

    # 2. 创建提交脚本文件
    try:
        with open(SUBMISSION_SCRIPT_NAME, "w") as f:
            f.write(slurm_script_content)
        # 赋予脚本执行权限
        os.chmod(SUBMISSION_SCRIPT_NAME, 0o755)
        print(f"✅ 成功创建HPC提交脚本: '{SUBMISSION_SCRIPT_NAME}'")
    except IOError as e:
        print(f"❌ 错误: 无法创建脚本文件. {e}")
        return

    # 3. 检查`sbatch`命令是否存在
    if not shutil.which("sbatch"):
        print("\n❌ 错误: 'sbatch' 命令未找到。")
        print("请确认您正在HPC的登录节点上运行此脚本，并且Slurm已正确安装。")
        return

    # 4. 使用subprocess提交作业
    print(f"\n🚀 正在使用 'sbatch {SUBMISSION_SCRIPT_NAME}' 提交作业...")
    try:
        # 执行sbatch命令并捕获输出
        result = subprocess.run(
            ["sbatch", SUBMISSION_SCRIPT_NAME],
            capture_output=True,
            text=True,
            check=True  # 如果sbatch返回非零退出码则抛出异常
        )
        print("\n--- sbatch 命令输出 ---")
        print(result.stdout.strip())
        print("---")
        print("✅ 作业已成功提交! 您可以使用 'squeue -u {HPC_USERNAME}' 命令查看作业状态。")

    except FileNotFoundError:
        print("❌ 错误: 'sbatch' 命令未找到。这不应该发生，因为我们已经检查过了。")
    except subprocess.CalledProcessError as e:
        print("❌ 错误: 'sbatch' 命令执行失败。")
        print("--- sbatch 返回的错误信息 ---")
        print(e.stderr)
        print("---")
        print("请检查您的Slurm配置或提交脚本中的参数。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    create_and_submit_hpc_job()