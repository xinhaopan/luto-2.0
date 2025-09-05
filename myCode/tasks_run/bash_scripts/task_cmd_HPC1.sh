#!/usr/bin/env bash
########################################################################
# submit_with_retry.sh —— 提交任务到 Slurm，并在失败时自动重试
# 每次都会重新评估可用节点，直到任务成功开始运行
########################################################################

set -euo pipefail
exec > slurm.log 2>&1
echo "[$(date '+%F %T')] Script started"

WORKDIR=$(pwd)
echo "[$(date '+%F %T')] Current working directory: $WORKDIR"

# ------------------------------------------------------------
# 1. 读取你的用户设置
# ------------------------------------------------------------
source luto/settings_bash.py

REQUIRED_MEM_MB=$(echo "$MEM" | awk '/GB/{printf "%d", $1*1024} /MB/{print $1}')
REQUIRED_CPU_CORES=$NCPUS

# ---------------- 节点优先级列表 ----------------------------
NODE_LIST=(
  "hpc-fc-b-1:mem"
  "hpc-gc-b-1:normal"
  "hpc-gc-b-2:normal"
  "hpc-dgx-b-3:dgx"
  "hpc-dgx-b-2:dgx"
  "hpc-dgx-b-1:dgx"
)

########################################################################
# 函数 choose_node —— 选出满足资源要求的可用节点
########################################################################
choose_node () {
  echo "Required: ${REQUIRED_MEM_MB} MB, ${REQUIRED_CPU_CORES} CPU" >&2
  echo "Checking available nodes…" >&2

  for entry in "${NODE_LIST[@]}"; do
    node=${entry%%:*}
    partition=${entry##*:}

    info=$(scontrol show node "$node" 2>/dev/null) || {
      echo "  ! Cannot query $node" >&2
      continue
    }

    free_mem=$(grep -o 'FreeMem=[0-9]*' <<<"$info" | cut -d= -f2)
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

# 激活环境只需一次
source ~/.bashrc
conda activate luto
PYTHON=$(which python)
echo "[$(date '+%F %T')] Python executable: $PYTHON"

########################################################################
# 循环重试直到提交成功
########################################################################
MAX_RETRIES=20

for RETRY_COUNT in $(seq 1 $MAX_RETRIES); do
  echo "[$(date '+%F %T')] 第 $RETRY_COUNT 次尝试提交任务..."

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
echo "SLURM job started at \$(date '+%F %T')"
${PYTHON} python_script.py
EOF

  echo "========== SLURM 脚本内容 BEGIN =========="
  cat "$SCRIPT_SLURM"
  echo "=========== SLURM 脚本内容 END ==========="

  job_submit_out=$(sbatch "$SCRIPT_SLURM" 2>&1)
  if [[ $? -ne 0 ]]; then
    echo "[$(date '+%F %T')] sbatch failed: $job_submit_out"
    rm -f "$SCRIPT_SLURM"
    sleep 10
    continue
  fi

  job_id=$(echo "$job_submit_out" | awk '{print $4}')
  echo "[$(date '+%F %T')] 提交任务 $job_id → $SELECTED_NODE"
  sleep 10

  # 等待任务开始运行
  MAX_WAIT=60
  WAITED=0
  job_started=false

  while [[ $WAITED -lt $MAX_WAIT ]]; do
    state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)

    if [[ "$state" == "RUNNING" ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 正在运行，脚本成功完成！"
      rm -f "$SCRIPT_SLURM"
      exit 0
    elif [[ "$state" == "FAILED" || "$state" == "CANCELLED" || "$state" == "COMPLETED" ]]; then
      echo "[$(date '+%F %T')] 任务 $job_id 快速结束 (state=$state)，将重试..."
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

echo "[$(date '+%F %T')] 已达到最大重试次数 $MAX_RETRIES，但没有成功提交任务。"
exit 1