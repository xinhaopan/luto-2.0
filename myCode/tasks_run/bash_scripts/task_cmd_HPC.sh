#!/usr/bin/env bash
########################################################################
# submit_with_retry.sh —— 提交任务到 Slurm，并在失败时自动重试
# 每次都会重新评估可用节点，直到任务成功开始运行
########################################################################

set -euo pipefail
exec > slurm.log 2>&1
echo "[$(date '+%F %T')] Script started"
script_name="$1"
echo "script_name: $script_name"


WORKDIR=$(pwd)
echo "[$(date '+%F %T')] Current working directory: $WORKDIR"

# ------------------------------------------------------------
# 1. 读取你的用户设置
# ------------------------------------------------------------
source luto/settings_bash.py

REQUIRED_MEM_MB=$(echo "$MEM" | awk '/GB/{printf "%d", $1*1024} /MB/{print $1}')
REQUIRED_CPU_CORES=$NCPUS

# ---------------- 原始节点优先级列表 ----------------------------
ORIGINAL_NODE_LIST=(
  "hpc-fc-b-1:mem"
  "hpc-gc-b-1:normal"
  "hpc-gc-b-2:normal"
  "hpc-dgx-b-3:dgx"
  "hpc-dgx-b-2:dgx"
  "hpc-dgx-b-1:dgx"
)

# 当前可用节点列表（会在循环中动态修改）
NODE_LIST=("${ORIGINAL_NODE_LIST[@]}")

########################################################################
# 函数 choose_node —— 选出满足资源要求的可用节点
########################################################################
choose_node () {
  echo "Required: ${REQUIRED_MEM_MB} MB, ${REQUIRED_CPU_CORES} CPU" >&2
  echo "Checking available nodes (${#NODE_LIST[@]} remaining)…" >&2

  for entry in "${NODE_LIST[@]}"; do
    node=${entry%%:*}
    partition=${entry##*:}

    info=$(scontrol show node "$node" 2>/dev/null) || {
      echo "  ! Cannot query $node" >&2
      continue
    }

    real_mem=$(grep -o 'RealMemory=[0-9]*' <<<"$info" | cut -d= -f2)
    alloc_mem=$(grep -o 'AllocMem=[0-9]*' <<<"$info" | cut -d= -f2)
    free_mem=$((real_mem - alloc_mem))

    cpus_tot=$(grep -o 'CPUTot=[0-9]*' <<<"$info" | cut -d= -f2)
    cpus_use=$(grep -o 'CPUAlloc=[0-9]*' <<<"$info" | cut -d= -f2)
    free_cpus=$((cpus_tot - cpus_use - 2)); ((free_cpus<0)) && free_cpus=0
    state=$(grep -o 'State=[A-Z+]*' <<<"$info" | cut -d= -f2 | tr '[:upper:]' '[:lower:]')

    echo "  - $node  mem=${free_mem}M  cpu=${free_cpus}  state=$state" >&2

    [[ $state == *down* || $state == *drain* || $state == *planned* ]] && continue

    if [[ $free_mem -ge $REQUIRED_MEM_MB && $free_cpus -ge $REQUIRED_CPU_CORES ]]; then
      echo "    -> Selected $node ($partition)" >&2
      echo "${node}|${partition}"
      return 0
    fi
  done

  return 1
}

########################################################################
# 函数 remove_node_from_list —— 从当前节点列表中移除指定节点
########################################################################
remove_node_from_list() {
  local node_to_remove="$1"
  local new_list=()

  for entry in "${NODE_LIST[@]}"; do
    node=${entry%%:*}
    if [[ "$node" != "$node_to_remove" ]]; then
      new_list+=("$entry")
    fi
  done

  NODE_LIST=("${new_list[@]}")
  echo "[$(date '+%F %T')] 已从节点列表中移除 $node_to_remove，剩余 ${#NODE_LIST[@]} 个节点" >&2
}

########################################################################
# 函数 reset_node_list —— 重置节点列表为原始完整列表
########################################################################
reset_node_list() {
  NODE_LIST=("${ORIGINAL_NODE_LIST[@]}")
  echo "[$(date '+%F %T')] 任务成功运行，重置节点列表为完整列表 (${#NODE_LIST[@]} 个节点)" >&2
}

# 激活环境只需一次
source ~/.bashrc
conda activate luto
PYTHON=$(which python)
echo "[$(date '+%F %T')] Python executable: $PYTHON"

########################################################################
# 循环重试直到提交成功
########################################################################
MAX_RETRIES=20
LAST_JOB_ID=""

for RETRY_COUNT in $(seq 1 $MAX_RETRIES); do
  echo "[$(date '+%F %T')] 第 $RETRY_COUNT 次尝试提交任务..."

  # 检查是否还有可用节点
  if [[ ${#NODE_LIST[@]} -eq 0 ]]; then
    echo "[$(date '+%F %T')] 所有节点都已尝试过，重置节点列表..."
    reset_node_list
  fi

  # 调用 choose_node 函数选择节点
  echo "[$(date '+%F %T')] 正在选择满足资源要求的节点..."
  node_info=$(choose_node)

  if [[ $? -ne 0 ]]; then
    echo "[$(date '+%F %T')] 第 $RETRY_COUNT 次尝试：没有找到满足要求的节点，等待 10 秒后重试..."
    sleep 10
    continue
  fi

  SELECTED_NODE=$(echo "$node_info" | cut -d'|' -f1)
  SELECTED_PARTITION=$(echo "$node_info" | cut -d'|' -f2)

  echo "[$(date '+%F %T')] 第 $RETRY_COUNT 次尝试提交 SLURM 任务：node=$SELECTED_NODE partition=$SELECTED_PARTITION"

  # 设置对应许可文件路径
  export GRB_LICENSE_FILE="$HOME/apps/gurobi1103/licenses/gurobi-${SELECTED_NODE}.lic"
  echo "Using license: $GRB_LICENSE_FILE"

  SCRIPT_SLURM=$(mktemp)
  cat << EOF > "$SCRIPT_SLURM"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SELECTED_PARTITION}
#SBATCH --nodelist=${SELECTED_NODE}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NCPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd "$WORKDIR"


# Create temporary directory (add job ID to avoid conflicts)
export CUSTOM_TMPDIR="/home/remote/s222552331/LUTO2_XH/TMPDIR/job_\${SLURM_JOB_ID}"
mkdir -p "\$CUSTOM_TMPDIR"
# Set all temporary directory environment variables
export TMPDIR="\$CUSTOM_TMPDIR"
export TMP="\$CUSTOM_TMPDIR"
export TEMP="\$CUSTOM_TMPDIR"
# Cleanup function (prevent accumulation of temporary files)
cleanup() {
    echo "Cleaning up temporary files: \$CUSTOM_TMPDIR"
    rm -rf "\$CUSTOM_TMPDIR"
}
trap cleanup EXIT
echo "Using temporary directory: \$TMPDIR"


echo "SLURM job started at \$(date '+%F %T')"
${PYTHON} ${script_name}
EOF

  echo "========== SLURM 脚本内容 BEGIN =========="
  cat "$SCRIPT_SLURM"
  echo "=========== SLURM 脚本内容 END ==========="

  if ! job_submit_out=$(sbatch "$SCRIPT_SLURM" 2>&1); then
    echo "[$(date '+%F %T')] sbatch failed: $job_submit_out"
    remove_node_from_list "$SELECTED_NODE"
    rm -f "$SCRIPT_SLURM"
    sleep 10
    continue
  fi

  job_id=$(echo "$job_submit_out" | awk '{print $4}')
  LAST_JOB_ID="$job_id"  # 记录最后一个成功提交的任务ID
  echo "[$(date '+%F %T')] 提交任务 $job_id → $SELECTED_NODE"
  sleep 5

  # 等待任务开始运行
  MAX_WAIT=30
  WAITED=0
  job_started=false

  while [[ $WAITED -lt $MAX_WAIT ]]; do
    state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)

    if [[ "$state" == "RUNNING" ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 正在运行，脚本成功完成！"
      reset_node_list  # 成功时重置节点列表
      rm -f "$SCRIPT_SLURM"
      exit 0
    elif [[ "$state" == "FAILED" || "$state" == "CANCELLED" || "$state" == "COMPLETED" ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 快速结束 (state=$state)，将重试..."
      # 任务快速失败时移除该节点
      remove_node_from_list "$SELECTED_NODE"
      rm -f "$SCRIPT_SLURM"
      job_started=false
      break
    elif [[ "$state" == "PENDING" ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 仍在等待中 (PENDING)，继续等待..."
    else
      echo "[$(date '+%F %T')] 任务 $job_id 状态: $state"
    fi

    sleep 5
    ((WAITED+=5))
  done

  # 处理超时情况
  if [[ $job_started == false && $WAITED -ge $MAX_WAIT ]]; then
    if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 在 $MAX_WAIT 秒内未能开始运行，将重试..."
      # 等待超时时移除该节点
      remove_node_from_list "$SELECTED_NODE"
      scancel "$job_id" 2>/dev/null || true
      rm -f "$SCRIPT_SLURM"
    else
      echo "[$(date '+%F %T')] 任务 $job_id 在 $MAX_WAIT 秒内未能开始运行，这是最后一次尝试，保留任务继续运行..."
      echo "[$(date '+%F %T')] 任务ID: $job_id，您可以使用 'squeue -j $job_id' 查看任务状态"
      rm -f "$SCRIPT_SLURM"
      exit 0  # 成功退出，让任务继续运行
    fi
  fi

done

# 如果执行到这里，说明已经达到最大重试次数
if [[ -n "$LAST_JOB_ID" ]]; then
  echo "[$(date '+%F %T')] 已达到最大重试次数 $MAX_RETRIES，最后一个任务 $LAST_JOB_ID 已提交，保留该任务继续运行"
  echo "[$(date '+%F %T')] 任务ID: $LAST_JOB_ID，您可以使用 'squeue -j $LAST_JOB_ID' 查看任务状态"
  exit 0  # 成功退出，让最后的任务继续运行
else
  echo "[$(date '+%F %T')] 已达到最大重试次数 $MAX_RETRIES，但没有成功提交任何任务"
  exit 1
fi