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
readonly RESUME_CKPT="/scratch/e1539128/ckpt2/dit_training/checkpoints/step-0000006000.pt"
readonly RESULTS_DIR="/scratch/e1539128/ckpt4"
readonly RESUME_MODE="model_only_reset_step"  # full | model_only | model_only_reset_step

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
    WANDB_EXP_NAME="dit_stage2_4gpu_exp11_512"

    case "$RESUME_MODE" in
      full)
        RESUME_ARGS=(--resume-path "$RESUME_CKPT")
        ;;
      model_only)
        RESUME_ARGS=(--resume-path "$RESUME_CKPT" --resume-model-only)
        ;;
      model_only_reset_step)
        RESUME_ARGS=(--resume-path "$RESUME_CKPT" --resume-model-only --resume-reset-progress)
        ;;
      *)
        echo "Invalid RESUME_MODE: $RESUME_MODE"
        exit 2
        ;;
    esac
    echo "Resume mode: $RESUME_MODE"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
      --nnodes=1 \
      --nproc_per_node=$NUM_GPUS \
      --master_port=29500 \
      t2i_training_single_ds/train.py \
      --config training/config/dit_training.yaml \
      --results-dir $RESULTS_DIR \
      "\${RESUME_ARGS[@]}" \
      --no-maintain-ema \
      --precision bf16 \
      --wandb \
      --wandb-name "\${WANDB_EXP_NAME}"
EOF
