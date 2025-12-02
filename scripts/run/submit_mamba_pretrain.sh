#!/bin/bash -l
#SBATCH --time=24:00:00              # Wall time
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=4          # MPI tasks per node
#SBATCH --cpus-per-task=32           # CPU cores per task
#SBATCH --gpus-per-node=4            # GPUs per node (adjust based on cluster)
#SBATCH -J mamba_pretrain            # Job name
#SBATCH -o mamba_pretrain_%j.out     # Standard output
#SBATCH -e mamba_pretrain_%j.err     # Standard error
#SBATCH -A YOUR_ACCOUNT              # Account/project (UPDATE THIS)
#SBATCH -q regular                   # Queue/partition

# ============================================================================
# Mamba Pretraining Submission Script
# Anonymized for publication
# ============================================================================

# === Configuration ===
config_file="./scripts/configs/mamba_pretrain.yaml"
config="mamba_5m"           # Options: mamba_5m, mamba2_5m, mamba_small
run_num="run0"              # Run identifier

# === Python Environment (UPDATE THIS) ===
PYTHON_BIN="/path/to/conda/envs/fm4npp/bin/python"

# Or activate conda environment:
# source activate fm4npp
# PYTHON_BIN="python"

# === Distributed Training Setup ===
export MASTER_ADDR=$(hostname)

# Command to execute
cmd="$PYTHON_BIN -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
    --yaml_config=$config_file \
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

echo "Pretraining job completed"
