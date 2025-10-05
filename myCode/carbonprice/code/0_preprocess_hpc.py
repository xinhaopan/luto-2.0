import os
import subprocess
import shutil

# ==============================================================================
#  配置区域
# ==============================================================================
CPU_CORES = 30 #90
MEMORY_GB = "1350G" # "1500G"
TIME_LIMIT = "0-720:00:00"

PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
CONDA_ENV_NAME = "xpluto-fixed"

SUBMISSION_SCRIPT_NAME = "submit_preprocess.sh"

# ==============================================================================

def create_and_submit_hpc_job():
    """
    动态创建Slurm提交脚本并使用sbatch提交作业。
    重点：加入并行/临时目录/句柄数等保护，缓解 joblib/loky 报错（_SemLock FileNotFound 等）。
    """
    print("--- HPC作业启动脚本（增强稳健性） ---")

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
# 可选：如果集群支持，避免 SMT 线程干扰
##SBATCH --hint=nomultithread

echo "========================================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点  : $(hostname)"
echo "开始时间: $(date)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "TMPDIR: ${{SLURM_TMPDIR:-/tmp}}"
echo "========================================================="

# --- 环境保护：限制内部库并行、设置本地临时目录、提升句柄上限 ---
# 这些变量能显著降低 joblib/loky 出错概率（SemLock丢失/进程被杀）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# joblib 临时文件与 memmap 放到节点本地盘，避免 NFS 竞争
export JOBLIB_TEMP_FOLDER="{{${{SLURM_TMPDIR}}:-/tmp}}"
export TMPDIR="{{${{SLURM_TMPDIR}}:-/tmp}}"

# 告诉 loky 不要超配核数
export LOKY_MAX_CPU_COUNT="${{SLURM_CPUS_PER_TASK}}"

# （可选）允许更多打开文件句柄（大量 memmap 时有用）
ulimit -n 65535 2>/dev/null || true

# 打印可用内存与磁盘，便于排障
command -v free >/dev/null 2>&1 && free -h || true
df -h "$JOBLIB_TEMP_FOLDER" "$TMPDIR" 2>/dev/null || true

# --- 步骤 1: 加载你的 shell 配置（含 conda 初始化） ---
echo "source ~/.bashrc"
source ~/.bashrc || echo "警告: source ~/.bashrc 失败（继续执行）"

# --- 步骤 2: 激活 Conda 环境 ---
echo "conda activate {CONDA_ENV_NAME}"
conda activate {CONDA_ENV_NAME}
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 Conda 环境 '{CONDA_ENV_NAME}'"
    exit 1
fi

echo "Python: $(which python)"
python --version
echo "---------------------------------------------------------"

# （可选）若 0_Preprocess.py 使用 joblib：
#   - 请读取环境变量 n_jobs = min(int(os.getenv("LOKY_MAX_CPU_COUNT", 1))-5, 40)
#   - Parallel(..., max_nbytes="256M", temp_folder=os.getenv("JOBLIB_TEMP_FOLDER"))
#   - 避免内外层并行叠加，内层库线程保持 1（已在上面环境变量限制）

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
