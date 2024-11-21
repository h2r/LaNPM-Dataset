#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

SPLIT_TYPE='task_split'
# EVAL_SCENE=4
# LOW_DIV='--low_div' $LOW_DIV
EPOCHS=30
LOAD_CHECKPOINT='/users/ajaafar/data/shared/lanmp/pretrained_rt1_ckpt/checkpoint_best.pt'
CHECKPOINT_DIR='/users/ajaafar/data/ajaafar/LaNMP-Dataset/models/main_models/rt1/checkpoints/new_data/rt1_ft_task2'
VAL_LOSS_DIR='val_losses/new_data/rt1_ft_task2'
EVAL_FREQ=50
CHECKPOINT_FREQ=0
TRAIN_BATCH=5
EVAL_BATCH=5
TRAIN_SUBBATCH=38
EVAL_SUBBATCH=38
LR=1e-5
LR_SCHED="plateau"
GAMMA=0.999
FACTOR=0.9
PATIENCE=1


python main_ft.py --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --load-checkpoint "$LOAD_CHECKPOINT" --val_loss_dir "$VAL_LOSS_DIR" --wandb --checkpoint-freq "$CHECKPOINT_FREQ" --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr "$LR" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE" --train-subbatch "$TRAIN_SUBBATCH" --eval-subbatch "$EVAL_SUBBATCH"