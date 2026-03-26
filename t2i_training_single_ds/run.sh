WANDB_EXP_NAME="dit_stage2_4gpu_exp06"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_port=29500 \
  training/train.py \
  --config training/config/dit_training.yaml \
  --results-dir ckpts \
  --precision bf16 \
  --wandb \
  --wandb-name "${WANDB_EXP_NAME}"


# CUDA_VISIBLE_DEVICES=0 python training/train_sanity_check.py --wandb
