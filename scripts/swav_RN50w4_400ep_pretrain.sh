# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --job-name=swav_RN50w4_400ep_pretrain
#SBATCH --time=72:00:00
#SBATCH --mem=450G
#SBATCH --constraint=volta32gb

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH="/path/to/imagenet/train"
EXPERIMENT_PATH="./experiments/swav_RN50w4_400ep_pretrain"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 4 \
--size_crops 224 96 \
--min_scale_crops 0.08 0.08 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 2560 \
--epoch_queue_starts 0 \
--epochs 400 \
--batch_size 40 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--dist_url $dist_url \
--arch resnet50w4 \
--hidden_mlp 8192 \
--use_fp16 true \
--sync_bn apex \
--dump_path $EXPERIMENT_PATH
