#!/bin/bash -l
# Track finding, paper's m6 (repo d9_m5_k30_p20, 175M params), 70k events, from cache.
# Run slurm/m6_cache.sh first.
#SBATCH -A m4722_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -J fm4npp_m6_70k
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load pytorch/2.6.0
source $SCRATCH/fm4npp/venv/bin/activate
cd $SCRATCH/fm4npp/PP_collision

# On a GB10 (unified host+device memory) this ran at batch 16, ~16 min/epoch, and
# early-stopped at epoch 105. An A100 has 80 GB of dedicated device memory -- raise
# --train_batch_size if it fits.
srun python train/downstream/train_track_finding.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m5_k30_p20 \
    --root_dir $SCRATCH/fm4npp/runs/ \
    --eventnumber 70000 \
    --train_batch_size 16 \
    --run_num m6_70k \
    --seed 42 \
    --feature_cache $SCRATCH/fm4npp/cache/m6_70k \
    --feature_cache_val $SCRATCH/fm4npp/cache/m6_val
