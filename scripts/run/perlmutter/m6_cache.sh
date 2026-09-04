#!/bin/bash -l
# Precompute frozen-backbone features for the paper's m6 (repo d9_m5_k30_p20, 175M params).
#
# --combine_layers is REQUIRED at this width. The head's only use of the per-layer stack is
# one convex combination, so applying it at cache time collapses the layer axis:
#     all 12 layers   36,864 bytes/point   2.17 TB for 70k events
#     combined         3,072 bytes/point     185 GB
# Cost: weighted_avg_weights stops adapting -- 12 scalars of 5,565,326, measured at no
# accuracy penalty (0.8508 combined vs 0.8480 learned, 10k events / 40 epochs).
#SBATCH -A m4722_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -J fm4npp_m6_cache
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load pytorch/2.6.0
source $SCRATCH/fm4npp/venv/bin/activate
cd $SCRATCH/fm4npp/PP_collision

# check the quota before writing 185 GB
myquota | head -5

srun python scripts/cache_features.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m5_k30_p20 \
    --checkpoint $SCRATCH/fm4npp/assets/pp_nerf_m5_k30.ckpt \
    --out $SCRATCH/fm4npp/cache/m6_70k \
    --eventnumber 70000 \
    --combine_layers

srun python scripts/cache_features.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m5_k30_p20 \
    --checkpoint $SCRATCH/fm4npp/assets/pp_nerf_m5_k30.ckpt \
    --out $SCRATCH/fm4npp/cache/m6_val \
    --eventnumber 500 --split test \
    --data_root $SCRATCH/fm4npp/data/val/ \
    --combine_layers
