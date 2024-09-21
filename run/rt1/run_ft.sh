#!/bin/bash

set -e
set -u

source /mnt/miniconda3/etc/profile.d/conda.sh
conda activate rt1

# module load cuda/11.8.0-lpttyok
# module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

SPLIT_TYPE='k_fold_scene'
# LOW_DIV='--low_div' $LOW_DIV
EPOCHS=3
CHECKPOINT_DIR='/mnt/ahmed/rt1/ft_ckpt8'
EVAL_FREQ=75
CHECKPOINT_FREQ=250
TRAIN_SUBBATCH=50
EVAL_SUBBATCH=50
TRAIN_BATCH=3
EVAL_BATCH=3
# EVAL_SCENE=4
LOAD_CHECKPOINT='/mnt/ahmed/rt1/pretrain_ckpts/checkpoint_231336.pt'
VAL_LOSS_DIR='val_losses/fractal_ft8'
LR=1e-4
GAMMA=0.99


python main_ft.py --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --load-checkpoint "$LOAD_CHECKPOINT" --val_loss_dir "$VAL_LOSS_DIR" --wandb --checkpoint-freq "$CHECKPOINT_FREQ" --lr "$LR" --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr_sched --gamma "$GAMMA"