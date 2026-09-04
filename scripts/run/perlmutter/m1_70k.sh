#!/bin/bash -l
# Track finding, paper's m3 (repo d9_m1_k30_p20, 5.3M params), 70k labeled events.
# No feature cache: the backbone is small enough that on-the-fly is fine.
#SBATCH -A m4722_g            # check `sacctmgr -n show assoc user=$USER format=account%20`
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -J fm4npp_m1_70k
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load pytorch/2.6.0
source $SCRATCH/fm4npp/venv/bin/activate
cd $SCRATCH/fm4npp/PP_collision

srun python train/downstream/train_track_finding.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m1_k30_p20 \
    --root_dir $SCRATCH/fm4npp/runs/ \
    --eventnumber 70000 \
    --train_batch_size 32 \
    --run_num m1_70k \
    --seed 42
