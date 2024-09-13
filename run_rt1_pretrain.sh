#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd models/main_models/rt1

CHECKPOINT_DIR='checkpoints/rt-1_pretrain_new'
# VAL_LOSS_DIR='val_losses/rt1_pretrain_new'

python main.py --checkpoint-dir "$CHECKPOINT_DIR"