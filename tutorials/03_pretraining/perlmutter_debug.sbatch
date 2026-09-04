#!/bin/bash -l
# FM4NPP tutorial -- module 03, pretraining the paper's m3 on Perlmutter's debug queue.
#
#   INPUT   $FM4NPP_PRETRAIN_ROOT (module 01)
#   OUTPUT  $FM4NPP_RUNS/tutorial_m3/debug0/{training_checkpoints/ckpt.tar, *.csv}
#
#   sbatch perlmutter_debug.sbatch
#
# ---------------------------------------------------------------------------
# FILL IN THE ACCOUNT BELOW. Everything else is set.
# Find yours with:  iris   (or)  sacctmgr show assoc user=$USER format=account
# ---------------------------------------------------------------------------
#SBATCH -A
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -J fm4npp_m3_debug
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# NOTE ON -C gpu: the repo's own submit scripts omit this. Without a constraint your job
# is not routed to the GPU partition. It is not optional on Perlmutter.
#
# NOTE ON THE DEBUG QUEUE: max 10 minutes, max 8 nodes, and typically a short wait.
# Budget below assumes ~1 min dataset scan + ~4 min for 200 steps at width 256 on 4xA100.

set -euo pipefail
mkdir -p logs

# --------------------------------------------------------------------- environment
module load python cudatoolkit

HERE="$SLURM_SUBMIT_DIR"
source "$HERE/../common/paths.sh"
source "$FM4NPP_WORK/.venv/bin/activate"

# --------------------------------------------------------------------- config
# Expand __FM4NPP_*__ placeholders into a real config (fails loudly if paths are wrong).
RENDERED="$SCRATCH/tutorial_m3.rendered.yaml"
python "$HERE/configs/render_config.py" "$HERE/configs/tutorial_m3.yaml" -o "$RENDERED"

# --------------------------------------------------------------------- distributed
# The vendored trainer calls init_process_group when WORLD_SIZE > 1. These four
# variables are what make that work; the repo's scripts leave them inside a commented-out
# block, which is why the original runs on one GPU no matter what SLURM was asked for.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

echo "=== $(date) starting on $SLURM_NNODES node(s), $SLURM_NTASKS task(s) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun -l bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  python '"$HERE"'/pretrain_m3.py \
      --yaml_config '"$RENDERED"' \
      --config tutorial_m3 \
      --root_dir '"$FM4NPP_RUNS"' \
      --run_num debug0 \
      --max_steps 200 \
      --mup
'

echo "=== $(date) done ==="
echo "loss curve:  python $HERE/plot_loss.py $FM4NPP_RUNS/tutorial_m3/debug0/config_tutorial_m3_run_debug0.csv"
