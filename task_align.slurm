#!/bin/sh
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4


EPOCHS=1
BATCH_SIZE=6
LR=1e-4
MIN_LR=8e-5
WARMUP_STEPS=3000
WARMUP_START_LR=1e-6


CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 scripts/train_align.py \
    --out_dir ./checkpoints/align_stage \
    --data_root "" \
    --llama_model "" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --warmup_start_lr $WARMUP_START_LR \
    --amp \
    --distributed \
    --seed 42