import os
import subprocess
import time

# ==============================================================================
#  配置区域
# ==============================================================================
CPU_CORES = 50
MEMORY_GB = "200GB"          # PBS 用 GB
TIME_LIMIT = "12:00:00"       # 一天
PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
CONDA_ENV_NAME = "xpluto"
SUBMISSION_SCRIPT_NAME = "submit_preprocess.pbs"
queue_name = "normalsr" # "hugmem" "normalsr" https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits
CHECK_INTERVAL = 600  # 检查间隔（秒）

def wait_for_other_jobs_to_complete():
    """
    检查服务器上的其他任务状态，如果有运行中或排队的任务，则等待它们完成。
    只有当所有其他任务都完成后，才会返回。
    """
    user = os.environ.get("USER", "")
    if not user:
        print("⚠️  无法获取用户名，跳过任务检查")
        return
    
    print("=" * 60)
    print("🔍 检查服务器上的其他任务状态...")
    print("=" * 60)
    
    while True:
        try:
            # 获取当前用户的所有任务
            result = subprocess.run(
                ["qstat", "-u", user],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # 如果没有任务，qstat 可能返回非零状态
                print("✅ 没有检测到其他正在运行的任务，可以提交新任务")
                return
            
            # 解析 qstat 输出，获取任务信息
            lines = result.stdout.strip().split('\n')
            
            # 过滤出运行中或排队的任务
            # qstat 的输出格式: Job id, Name, User, Time Use, S, Queue
            # S (状态): R = 运行中, Q = 排队, C = 已完成
            active_jobs = []
            
            for line in lines:
                if line.strip() == "" or "Job id" in line or "-" in line[:10]:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    job_id = parts[0]
                    job_name = parts[1]
                    job_status = parts[4]  # 状态列
                    
                    # 只关注运行中(R)和排队(Q)的任务
                    if job_status in ['R', 'Q']:
                        active_jobs.append({
                            'id': job_id,
                            'name': job_name,
                            'status': '运行中' if job_status == 'R' else '排队'
                        })
            
            if not active_jobs:
                print("✅ 没有检测到其他正在运行的任务，可以提交新任务")
                return
            
            # 显示当前活跃的任务
            print(f"\n⏳ 检测到 {len(active_jobs)} 个活跃任务:")
            for job in active_jobs:
                print(f"   - 任务 {job['id']}: {job['name']} [{job['status']}]")
            
            print(f"\n⏳ 等待上述任务完成，下次检查时间：{CHECK_INTERVAL} 秒后")
            print(f"   当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 等待指定时间后再检查
            time.sleep(CHECK_INTERVAL)
            
        except subprocess.TimeoutExpired:
            print("⚠️  qstat 命令超时，继续等待...")
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"⚠️  检查任务状态时出错: {e}")
            print("   继续等待...")
            time.sleep(CHECK_INTERVAL)

def create_and_submit_hpc_job():
    """
    动态创建 PBS 提交脚本并用 qsub 提交。
    """
    print("--- PBS 作业启动脚本 ---")

    pbs_script_content = f"""#!/bin/bash
#PBS -N carbon_preprocess
#PBS -q {queue_name}
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
    # 先检查是否有其他任务在运行，如果有则等待
    wait_for_other_jobs_to_complete()
    
    # 所有其他任务完成后，提交新的 HPC 任务
    print("\n" + "=" * 60)
    print("🎯 所有其他任务已完成，现在提交新任务")
    print("=" * 60 + "\n")
    create_and_submit_hpc_job()
