#!/bin/bash

set -e
set -u

# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
source /mnt/miniconda3/etc/profile.d/conda.sh

conda activate rt1

# module load cuda/11.8.0-lpttyok
# module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

ITER=5
SPLIT_TYPE='task_split'
TEST_SCENE=""
# LOW_DIV='--low_div' $LOW_DIV
# LOAD_CHECKPOINT='/users/ajaafar/data/shared/lanmp/pretrained_rt1_ckpt/checkpoint_best.pt'
# CHECKPOINT_DIR='/users/ajaafar/data/ajaafar/LaNMP-Dataset/models/main_models/rt1/checkpoints/new_data/rt1-nodist-${SPLIT_TYPE}-${TEST_SCENE}-${ITER}'
LOAD_CHECKPOINT='/mnt/ahmed/rt1/pretrain_ckpts/checkpoint_best.pt'
CHECKPOINT_DIR='/home/ahmedjaafar/LaNMP-Dataset/models/main_models/rt1/checkpoints/new_data/rt1-nodist--${SPLIT_TYPE}-${TEST_SCENE}-${ITER}'
VAL_LOSS_DIR='val_losses/new_data/rt1-nodist-${SPLIT_TYPE}-${TEST_SCENE}-${ITER}'
EPOCHS=30
EVAL_FREQ=50
CHECKPOINT_FREQ=0
TRAIN_BATCH=6
EVAL_BATCH=6
TRAIN_SUBBATCH=24
EVAL_SUBBATCH=24
LR=1e-4
LR_SCHED="plateau"
GAMMA=0.99
FACTOR=0.75
PATIENCE=1


python main_ft.py --use-dist --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --load-checkpoint "$LOAD_CHECKPOINT" --val_loss_dir "$VAL_LOSS_DIR" --wandb --checkpoint-freq "$CHECKPOINT_FREQ" --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr "$LR" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE" --train-subbatch "$TRAIN_SUBBATCH" --eval-subbatch "$EVAL_SUBBATCH" --test-scene "$TEST_SCENE"