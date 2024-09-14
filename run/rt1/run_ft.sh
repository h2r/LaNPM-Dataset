#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd models/main_models/rt1

SPLIT_TYPE='k_fold_scene'
# LOW_DIV='--low_div' $LOW_DIV
EPOCHS=2
CHECKPOINT_DIR='checkpoints/rt-1_moma_ft'
EVAL_FREQ=100
TRAIN_SUBBATCH=8
EVAL_SCENE=4
LOAD_CHECKPOINT='checkpoints/rt1_fractal_pretrain/checkpoint_23600_loss_135.182.pt'
VAL_LOSS_DIR='val_losses/kfold'


python main_ft.py --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --train-subbatch "$TRAIN_SUBBATCH" --eval-scene "$EVAL_SCENE" --load-checkpoint "$LOAD_CHECKPOINT" --val_loss_dir "$VAL_LOSS_DIR"