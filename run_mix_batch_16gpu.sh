#!/bin/bash
#PBS -P CFP03-CF-035
#PBS -j oe
#PBS -N rae-qwen-16gpu
#PBS -q auto
#PBS -l select=2:ngpus=8
#PBS -l walltime=48:00:00
#PBS -V

set -euo pipefail

cd "$PBS_O_WORKDIR"

readonly NNODES=2
readonly GPUS_PER_NODE=8
readonly MASTER_PORT=29500
readonly RESULTS_DIR="/scratch/e1539128/ckpt-16gpu"
readonly RESUME_PATH="/scratch/e1539128/ckpt-8gpu/dit_training/checkpoints/step-0000016000.pt"
readonly WANDB_EXP_NAME="dit_stage2_mixed_16gpu_exp01"
readonly SIF_IMAGE="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"

export HF_HOME="/scratch/e1539128/cache/huggingface"
export XDG_CACHE_HOME="/scratch/e1539128/cache"

mapfile -t NODELIST < <(awk '!seen[$0]++' "$PBS_NODEFILE")
if [ "${#NODELIST[@]}" -ne "$NNODES" ]; then
  echo "[launcher] expected ${NNODES} unique nodes, got ${#NODELIST[@]}"
  echo "[launcher] PBS_NODEFILE:"
  cat "$PBS_NODEFILE"
  exit 2
fi

readonly MASTER_ADDR="${NODELIST[0]}"

declare -a NODE_SLOTS=()
for host in "${NODELIST[@]}"; do
  slot_idx="$(awk -v h="$host" '$0==h {print NR-1; exit}' "$PBS_NODEFILE")"
  if [ -z "$slot_idx" ]; then
    echo "[launcher] failed to map host $host to PBS slot index."
    exit 2
  fi
  NODE_SLOTS+=("$slot_idx")
done

echo "[launcher] nodes: ${NODELIST[*]}"
echo "[launcher] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
echo "[launcher] world_size=$((NNODES * GPUS_PER_NODE))"

if [ ! -f "${RESUME_PATH}" ]; then
  echo "[launcher] resume checkpoint not found: ${RESUME_PATH}"
  exit 2
fi
echo "[launcher] resuming from checkpoint: ${RESUME_PATH}"

declare -a PIDS=()
for ((node_rank=0; node_rank<NNODES; node_rank++)); do
  slot="${NODE_SLOTS[$node_rank]}"
  host="${NODELIST[$node_rank]}"
  echo "[launcher] launching node_rank=${node_rank} on host=${host} (pbs slot=${slot})"

  pbsdsh -n "$slot" /usr/bin/bash -lc "
    set -euo pipefail
    cd '$PBS_O_WORKDIR'
    module load singularity

    export HF_HOME='/scratch/e1539128/cache'
    export HF_HUB_CACHE='/scratch/e1539128/cache/hub'
    export TRANSFORMERS_CACHE='/scratch/e1539128/cache/transformers'
    export XDG_CACHE_HOME='/scratch/e1539128/cache'
    export HF_HUB_DOWNLOAD_TIMEOUT=60
    export HF_HUB_ETAG_TIMEOUT=30

    singularity exec -e '$SIF_IMAGE' bash -lc '
      set -euo pipefail
      source /scratch/e1539128/virtualenvs/rae/bin/activate
      echo \"[node_rank=${node_rank} host=${host}] starting torchrun\"
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=${node_rank} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        t2i_training_mixed_ds/train.py \
        --config t2i_training_mixed_ds/config/dit_training.yaml \
        --results-dir ${RESULTS_DIR} \
        --precision bf16 \
        --resume-path ${RESUME_PATH} \
        --wandb \
        --wandb-name ${WANDB_EXP_NAME}
    '
  " &

  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
