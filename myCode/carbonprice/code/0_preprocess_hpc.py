import os
import subprocess
import shutil
import time

# ==============================================================================
#  配置区域
# ==============================================================================
CPU_CORES = 42 #90
MEMORY_GB = "1200G" # "1500G"
TIME_LIMIT = "0-72:00:00"

PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
CONDA_ENV_NAME = "xpluto-fixed"

SUBMISSION_SCRIPT_NAME = "submit_preprocess.sh"
CHECK_INTERVAL = 600  # 检查间隔（秒）

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

def wait_for_other_jobs_to_complete():
    """
    检查服务器上当前用户的Slurm任务状态，如果有运行中或排队的任务，则等待它们完成。
    只有当所有任务都完成后，才会返回。
    """
    user = os.environ.get("USER", "")
    if not user:
        print("⚠️  无法获取用户名，跳过任务检查")
        return

    print("=" * 60)
    print("🔍 检查服务器上的任务状态...")
    print("=" * 60)

    while True:
        try:
            result = subprocess.run(
                ["squeue", "-u", user, "--noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0 or not result.stdout.strip():
                print("✅ 没有检测到正在运行的任务，可以提交新任务")
                return

            # 解析 squeue 输出
            # 格式: JOBID PARTITION NAME USER ST TIME NODES NODELIST
            # ST: R=运行中, PD=排队, CG=completing
            active_jobs = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    job_id   = parts[0]
                    job_name = parts[2]
                    status   = parts[4]
                    if status in ('R', 'PD', 'CG'):
                        label = {'R': '运行中', 'PD': '排队', 'CG': '结束中'}.get(status, status)
                        active_jobs.append({'id': job_id, 'name': job_name, 'status': label})

            if not active_jobs:
                print("✅ 没有检测到正在运行的任务，可以提交新任务")
                return

            print(f"\n⏳ 检测到 {len(active_jobs)} 个活跃任务:")
            for job in active_jobs:
                print(f"   - 任务 {job['id']}: {job['name']} [{job['status']}]")

            print(f"\n⏳ 等待上述任务完成，下次检查时间：{CHECK_INTERVAL} 秒后")
            print(f"   当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(CHECK_INTERVAL)

        except subprocess.TimeoutExpired:
            print("⚠️  squeue 命令超时，继续等待...")
            time.sleep(CHECK_INTERVAL)
        except FileNotFoundError:
            print("⚠️  未找到 'squeue' 命令，可能不在HPC环境中，跳过检查")
            return
        except Exception as e:
            print(f"⚠️  检查任务状态时出错: {e}")
            print("   继续等待...")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # 先检查是否有任务在运行，如果有则等待
    wait_for_other_jobs_to_complete()

    # 所有任务完成后，提交新的 HPC 任务
    print("\n" + "=" * 60)
    print("🎯 所有任务已完成，现在提交新任务")
    print("=" * 60 + "\n")
    create_and_submit_hpc_job()
