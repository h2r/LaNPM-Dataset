#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

CHECKPOINT_DIR='/users/ajaafar/scratch/rt1_pretrain_ckpts2'
CHECKPOINT_FREQ=2142
EVAL_FREQ=2142
VAL_LOSS_DIR='val_losses/rt1_pretrain_new2'
LR=1e-5

python main_pretrain.py --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-freq "$CHECKPOINT_FREQ" --val_loss_dir "$VAL_LOSS_DIR" --eval-freq "$EVAL_FREQ" --lr "$LR" --wandb