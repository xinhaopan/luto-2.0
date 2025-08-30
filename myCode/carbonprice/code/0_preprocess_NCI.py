import os
import subprocess

# ==============================================================================
# 配置（按需修改）
# ==============================================================================

QUEUE        = "hugemem"       # 建议：megamem（500GB更省SU）；或 "hugemem"/"normal"
NCPUS        = 16               # 建议：500GB在megamem下 ≤8核同价，8核最划算
MEMORY       = "500GB"         # 申请内存
WALLTIME     = "48:00:00"      # 形如 "HH:MM:SS"（或 "DD:HH:MM:SS" 也可）
JOBFS        = None            # 例如 "200GB"；无需要可设 None


# Python脚本与Conda环境
PYTHON_SCRIPT_TO_RUN = "0_Preprocess.py"
CONDA_ENV_NAME       = "xpluto"

# 生成的提交脚本名
SUBMISSION_SCRIPT_NAME = "submit_preprocess.pbs"

# 作业名
JOB_NAME = "carbon_preprocess"

# 节点约束（一般不需要手动指定节点名；PBS会根据资源自动分配）
NODELIST = None   # 例如 "gadi-cpu-clx-001"; 不需要就设 None

# ==============================================================================

def create_and_submit_pbs_job():
    print("--- 生成 PBS 提交脚本（NCI/Gadi） ---")


    # 组装 -l 资源行
    l_fields = [f"ncpus={NCPUS}", f"mem={MEMORY}", f"walltime={WALLTIME}"]
    if JOBFS:
        l_fields.append(f"jobfs={JOBFS}")
    l_line = ",".join(l_fields)

    nodelist_line = f"#PBS -l host={NODELIST}\n" if NODELIST else ""

    pbs_script_content = f"""#!/bin/bash
#PBS -q {QUEUE}
#PBS -N {JOB_NAME}
#PBS -l {l_line}
#PBS -l storage=gdata/jk53+scratch/jk53
#PBS -l wd
#PBS -j oe
{nodelist_line}# ===== 作业信息 =====
WORKDIR="/g/data/jk53/LUTO_XH/LUTO2/myCode/carbonprice/code"

echo "========================================================="
echo "PBS_JOBID: $PBS_JOBID"
echo "开始时间: $(date)"
echo "切换到: $WORKDIR"
cd "$WORKDIR" || {{ echo "ERROR: 目录不存在: $WORKDIR"; exit 1; }}
echo "当前工作目录: $PWD"
echo "========================================================="

set -euo pipefail

# 激活 Conda（尽量兼容你的 miniforge/miniconda 路径）
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$HOME/.bashrc" || true
fi

echo "激活 Conda 环境: {CONDA_ENV_NAME}"
conda activate {CONDA_ENV_NAME}

echo "Python 路径: $(which python)"
python --version

echo "开始执行 Python 脚本: {PYTHON_SCRIPT_TO_RUN}"
python -u "{PYTHON_SCRIPT_TO_RUN}"

echo "========================================================="
echo "脚本执行完毕。结束时间: $(date)"
echo "========================================================="
"""

    # 写文件
    with open(SUBMISSION_SCRIPT_NAME, "w") as f:
        f.write(pbs_script_content)
    os.chmod(SUBMISSION_SCRIPT_NAME, 0o755)
    print(f"✅ 已生成: {SUBMISSION_SCRIPT_NAME}")

    # 提交
    print(f"🚀 使用 qsub 提交作业…")
    try:
        out = subprocess.check_output(["qsub", SUBMISSION_SCRIPT_NAME], text=True)
        print("--- qsub 输出 ---")
        print(out.strip())
        print("-----------------")
        print(f"✅ 作业已提交。用 `qstat -u $USER` 或 `qstat -f <JOBID>` 查看状态。")
    except subprocess.CalledProcessError as e:
        print("❌ qsub 提交失败：")
        print(e.output)
    except FileNotFoundError:
        print("❌ 未找到 qsub 命令，请在NCI登录节点上运行此脚本。")

if __name__ == "__main__":
    create_and_submit_pbs_job()
