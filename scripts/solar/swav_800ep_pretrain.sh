#!bin/bash 
#SBATCH -p scavenger-gpu
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -o ./experiments/solar/slurm.out
#SBATCH -e ./experiments/solar/slurm.err
#SBATCH --mem=64G 

DATASET_PATH="/hpc/group/energy/zdc6/data/solar-pv"
EXPERIMENT_PATH="./experiments/solar/swav_800ep_pretrain"
mkdir -p $EXPERIMENT_PATH
 
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
    --data_path $DATASET_PATH \
    --nmb_crops 2 6 \
    --size_crops 224 96 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.14 \
    --crops_for_assign 0 1 \
    --temperature 0.1 \
    --epsilon 0.05 \
    --sinkhorn_iterations 3 \
    --feat_dim 128 \
    --nmb_prototypes 3000 \
    --queue_length 0 \
    --epochs 800 \
    --batch_size 32 \
    --base_lr 0.6 \
    --final_lr 0.0006 \
    --freeze_prototypes_niters 313 \
    --wd 0.000001 \
    --warmup_epochs 0 \
    --start_warmup 0.3 \
    --dist_url $dist_url \
    --arch resnet50 \
    --use_fp16 true \
    --sync_bn apex \
    --queue_length 3840 \
    --task solar \
    --dump_path $EXPERIMENT_PATH