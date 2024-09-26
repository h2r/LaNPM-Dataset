#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

CHECKPOINT_DIR='/users/ajaafar/scratch/rt1_pretrain_ckpts6'
CHECKPOINT_FREQ=2142
EVAL_FREQ=80
VAL_LOSS_DIR='val_losses/rt1_pretrain_new6'
TRAIN_BATCH=12
EVAL_BATCH=12
LR=1e-4
LR_SCHED="plateau"
GAMMA=0.999
FACTOR=0.1
PATIENCE=1


python main_pretrain.py --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-freq "$CHECKPOINT_FREQ" --val_loss_dir "$VAL_LOSS_DIR" --eval-freq "$EVAL_FREQ" --lr "$LR" --wandb --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE"