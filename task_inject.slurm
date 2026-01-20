#!/bin/bash
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4

EPOCHS=1
BATCH_SIZE=6
LR=1e-4
MIN_LR=8e-5
WARMUP_STEPS=3000
WARMUP_START_LR=1e-6

COCO_DATA_ROOT=""
LLAMA_MODEL=""
Q_FORMER_MODEL=""
EXT_PATH=""
PROMPT_PATH=""
OUTPUT_DIR=""
STAGE1_CHECKPOINT=""


export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

torchrun \
    --nproc_per_node=4 \
    scripts/train_inject.py \
    --output_dir $OUTPUT_DIR \
    --data_root $COCO_DATA_ROOT \
    --ext_path $EXT_PATH \
    --llama_model $LLAMA_MODEL \
    --q_former_model $Q_FORMER_MODEL \
    --prompt_path $PROMPT_PATH \
    --stage1_checkpoint $STAGE1_CHECKPOINT \
    --topn 9 \
    --num_query_token_txt 8 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --warmup_start_lr $WARMUP_START_LR \
    --amp \
    --distributed \
    --seed 42