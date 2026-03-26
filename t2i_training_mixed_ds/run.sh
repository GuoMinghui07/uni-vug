WANDB_EXP_NAME="dit_stage2_mixed_4gpu_exp03"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29500 \
  t2i_training_mixed_ds/train.py \
  --config t2i_training_mixed_ds/config/dit_training.yaml \
  --results-dir ckpts \
  --precision bf16 \
  --resume-path /scratch/e1539128/ckpt4/dit_training/checkpoints/step-0000014000.pt \
  --resume-model-only \
  --resume-reset-progress \
  --wandb \
  --wandb-name "${WANDB_EXP_NAME}"
