WANDB_EXP_NAME="dit_video_4gpu"
INIT_CKPT="/scratch/e1539128/ckpt-4gpu-new/dit_training/checkpoints/step-0000018000.pt"
T2V_RESULTS_DIR="/scratch/e1539128/ckpt-video"
T2V_SANITY_RESULTS_DIR="/scratch/e1539128/ckpt-video-sanity-check"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29500 \
  t2v_training_single_ds/train.py \
  --config t2v_training_single_ds/config/dit_training.yaml \
  --results-dir "${T2V_RESULTS_DIR}" \
  --precision bf16 \
  --resume-path "${INIT_CKPT}" \
  --resume-model-only \
  --resume-reset-progress \
  --wandb \
  --wandb-name "${WANDB_EXP_NAME}"

# Sanity check (single GPU example):
CUDA_VISIBLE_DEVICES=0 python t2v_training_single_ds/sanity_check.py \
  --config t2v_training_single_ds/config/sanity_check.yaml \
  --results-dir "${T2V_SANITY_RESULTS_DIR}" \
  --precision bf16 \
  --resume-path /scratch/e1539128/ckpt-8gpu-new/dit_training/checkpoints/step-0000007000.pt \
  --resume-model-only \
  --resume-reset-progress \
  --wandb
