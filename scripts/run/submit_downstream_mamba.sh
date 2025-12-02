#!/bin/bash -l
#SBATCH --time=24:00:00              # Wall time
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=4          # MPI tasks per node
#SBATCH --cpus-per-task=32           # CPU cores per task
#SBATCH --gpus-per-node=4            # GPUs per node (adjust based on cluster)
#SBATCH -J mamba_downstream          # Job name
#SBATCH -o mamba_downstream_%j.out   # Standard output
#SBATCH -e mamba_downstream_%j.err   # Standard error
#SBATCH -A YOUR_ACCOUNT              # Account/project (UPDATE THIS)
#SBATCH -q regular                   # Queue/partition

# ============================================================================
# Mamba Track Finding (Downstream) Submission Script
# Anonymized for publication
# ============================================================================

# === Configuration ===
config_file="./scripts/configs/mamba_tracking.yaml"
config="mamba_5m_downstream"  # Options: mamba_5m_downstream, mamba2_5m_downstream, mamba_5m_scratch
run_num="run0"                # Run identifier

# === Python Environment (UPDATE THIS) ===
PYTHON_BIN="/path/to/conda/envs/fm4npp/bin/python"

# Or activate conda environment:
# source activate fm4npp
# PYTHON_BIN="python"

# === Distributed Training Setup ===
export MASTER_ADDR=$(hostname)

# Change to training directory
cd train/downstream

# Command to execute
cmd="$PYTHON_BIN train_downstream.py \
    --yaml_config=../../$config_file \
    --config=$config \
    --run_num=$run_num"

# === CUDA Environment Setup (if needed) ===
# Uncomment and adjust based on your cluster:
# export CUDA_HOME=/path/to/cuda
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH

# === Launch Training ===
set -x  # Print commands for debugging

# Using srun for distributed training
srun -l \
    bash -c "
    # Export DDP environment variables
    export LOCAL_RANK=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export MASTER_PORT=29500

    # Run training
    $cmd
    "

# Alternative: Direct execution for single-node
# $cmd

echo "Downstream training job completed"
