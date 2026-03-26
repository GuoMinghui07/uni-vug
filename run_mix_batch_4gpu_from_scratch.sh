#!/bin/bash
#PBS -P CFP03-CF-035          
#PBS -j oe                    
#PBS -N rae-qwen           
#PBS -q auto                  
#PBS -l select=1:ngpus=4      
#PBS -l walltime=48:00:00   
#PBS -V  

cd $PBS_O_WORKDIR;
readonly NUM_GPUS=4

export HF_HOME="/scratch/e1539128/cache/huggingface"
export XDG_CACHE_HOME="/scratch/e1539128/cache"

image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"

module load singularity

singularity exec -e $image bash << EOF
    source /scratch/e1539128/virtualenvs/rae/bin/activate

    echo "Running with HF_HOME: \$HF_HOME"
    export HF_HOME=/scratch/e1539128/cache
    export HF_HUB_CACHE=/scratch/e1539128/cache/hub
    export TRANSFORMERS_CACHE=/scratch/e1539128/cache/transformers
    export XDG_CACHE_HOME=/scratch/e1539128/cache
    # Raise hub timeouts to reduce false-positive stream failures under transient network jitter.
    export HF_HUB_DOWNLOAD_TIMEOUT=60
    export HF_HUB_ETAG_TIMEOUT=30
    WANDB_EXP_NAME="dit_stage2_mixed_4gpu_continue2"
    # CKPT_DIR="/scratch/e1539128/ckpt-4gpu-new/dit_training/checkpoints"
    # RESUME_CKPT="\$(ls -1 "\${CKPT_DIR}"/step-*.pt 2>/dev/null | sort | tail -n 1)"
    # if [[ -z "\${RESUME_CKPT}" || ! -f "\${RESUME_CKPT}" ]]; then
    #   echo "[fatal] No resume checkpoint found under: \${CKPT_DIR}" >&2
    #   exit 1
    # fi
    # echo "Resuming full training state from checkpoint: \${RESUME_CKPT}"
    RESUME_CKPT="/scratch/e1539128/ckpt-4gpu-new/dit_training/checkpoints/step-0000009000.pt"

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
      --nnodes=1 \
      --nproc_per_node=$NUM_GPUS \
      --master_port=29500 \
      t2i_training_mixed_ds/train.py \
      --config t2i_training_mixed_ds/config/dit_training.yaml \
      --results-dir /scratch/e1539128/ckpt-4gpu-new2 \
      --precision bf16 \
      --resume-path "\${RESUME_CKPT}" \
      --wandb \
      --wandb-name "\${WANDB_EXP_NAME}"
EOF
