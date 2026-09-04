# Setup and Usage Guide

Detailed instructions for setting up and running FM4NPP experiments.

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: A100, V100, or RTX 30xx/40xx)
- **Memory**: 40GB+ GPU memory recommended for full-scale training
- **Storage**: 100GB+ for preprocessed data

### Software
- Python 3.10 or higher
- CUDA 12.1 or higher
- Linux or macOS (Windows via WSL2)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd FM4NPP_Public
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n fm4npp python=3.10
conda activate fm4npp

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Mamba dependencies.
# NOT `pip install mamba-ssm causal-conv1d`: PyPI has no wheels for either package, so
# that compiles CUDA extensions for 20-60 minutes and fails without --no-build-isolation
# (setup.py imports torch). This script fetches the matching prebuilt wheel instead.
pip install triton
bash tutorials/01_environment/install_kernels.sh

# Install other requirements
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('Mamba installed successfully')"
```

## Data Preparation

### Data Format

The code reads memory-mapped `RaggedMmap` directories. Note the split names: the
TRAINING loader is hardcoded to `split='pretrain'` and the eval loader to
`split='test'` (see `get_data_loader` in `fm4npp/datasets/dataset.py`). So your
labeled *training* data must be written with the `pretrain` suffix. The name is
historical -- it is not the unlabeled pretraining corpus.

```
data_root/                       # training data (params.data_root)
├── features_pretrain/           # (n_i, 4) float32
├── seg_target_pretrain/         # (n_i,)   int64  track ids
├── pid_target_pretrain/         # (n_i,)   int64  raw PDG codes
└── reg_target_pretrain/         # (n_i, 8) float32
data_root_test/                  # eval data (params.data_root_test)
├── features_test/  seg_target_test/  pid_target_test/  reg_target_test/
```

### Feature Format (4D per point)

Each spacepoint has **4** features, not 30:

| index | meaning |
|---|---|
| 0 | `E`  energy deposition |
| 1 | `x` |
| 2 | `y` |
| 3 | `z` |

The dataset converts (x, y, z) to polar (eta, phi, r) internally and normalizes with
constants hardcoded in `TPCBatchDataset.__init__`: `E` is z-normalized with
mean 253.0982 / std 268.7093, `eta` min-maxed over [-2, 2], `phi` over [-pi, pi],
and `r` over [31.372, 75.385].

### Target Formats

- `seg_target` -- integer track id per point. Track finding clusters points by these.
- `pid_target` -- **raw PDG codes**, not class indices.
  `downstream_util.get_pidlabel()` maps them itself
  (211 -> 1 pion, 321 -> 2 kaon, 2212 -> 3 proton, 11 -> 4 electron, else 0).
- `reg_target` -- `(n, 8)` = `px, py, pz, vtx_x, vtx_y, vtx_z, q, e`.
  `downstream_util.get_trackinfo_noiselabel()` derives `noise_labels`
  (pt < 0.06), `valid_tracks` (vertex within 1 cm) and `track_info` from it.

### Statistics Files

`stat_dir` must contain:
- `bin_edges_v3_nbins_8_8_6.pkl` -- voxelizer bin edges (`fm4npp/datasets/voxelizer.py`)
- `loss_bin_pp.pkl`, `loss_weight_pp.pkl` -- loaded unconditionally by the trainers

These now ship in [`stats/`](stats/) — set `stat_dir` to that directory. They were
previously absent because the blanket `*.pkl` rule in `.gitignore` excluded them, which
is worth knowing about: if the bin-edges file is missing the voxelizer silently
*recomputes* bins from your dataset, producing a different tokenization from the one the
released checkpoints were pretrained with, with no error raised. The two loss pickles
have no fallback and raise `FileNotFoundError`.

### Getting the Data Without a 118 GB Download

The Zenodo release is one 118.5 GB zip, but the downstream tasks only need the labeled
splits, which come to **under 1 GB**. Zenodo serves HTTP range requests, so you can pull
just those members:

```bash
pip install remotezip
python scripts/fetch_labeled_data.py --out ./TPCpp-10M
# fetching 32 members, 0.97 GB (the full archive is 118.53 GB)
```

Download the whole archive only if you intend to pretrain from scratch -- the other
117 GB is the unlabeled pretraining corpus.

Note that `labeled/train` is **sharded** (`spacepoints_000.npz` ... `_006.npz`) while
`labeled/test` and `labeled/validation` are single files. `prepare_data.py` handles both.

### Preparing the Public Dataset

The Zenodo release ships flat `.npz` files, not `RaggedMmap` directories. Use the
converter in this repo:

```bash
# Zenodo: https://doi.org/10.5281/zenodo.16970029
# labeled/<split>/{spacepoints,track_ids,pid_labels,noise_tags}.npz

python scripts/prepare_data.py \
    --in_dir /path/to/TPCpp-10M/labeled/train \
    --out    /path/to/mmap_train \
    --split  pretrain          # 'pretrain' == the TRAINING loader, see above

python scripts/prepare_data.py \
    --in_dir /path/to/TPCpp-10M/labeled/test \
    --out    /path/to/mmap_test \
    --split  test
```

Two caveats, both documented in `scripts/prepare_data.py`:

1. Zenodo's `pid_labels` are already *class indices* (0-4), whereas the code expects
   raw PDG codes and maps them itself. The converter inverts the mapping so the
   round-trip is exact. Passing the Zenodo labels through unchanged collapses every
   point to class 0.
2. Zenodo publishes no `reg_target` -- only a boolean `noise_tag`. The converter
   synthesizes an 8-column `reg_target` that reproduces `noise_labels` faithfully and
   sets `valid_tracks = 1`; `track_info` is a **placeholder**. Track finding and noise
   tagging do not read `track_info`, but any task regressing track kinematics needs
   the real regression targets from the collaboration.

## Configuration

### 1. Update Paths in Config Files

Edit `scripts/configs/mamba_pretrain.yaml`:
```yaml
data_root: /your/path/to/preprocessed/data
stat_dir: /your/path/to/statistics
checkpoint_dir: /your/path/to/checkpoints
```

Edit `scripts/configs/mamba_tracking.yaml`:
```yaml
data_root: /your/path/to/preprocessed/data
stat_dir: /your/path/to/statistics
pretrained_ckpt: /your/path/to/pretrain/checkpoint.tar
checkpoint_dir: /your/path/to/downstream/checkpoints
```

### 2. Update SLURM Scripts (if using)

Edit `scripts/run/submit_mamba_pretrain.sh`:
```bash
#SBATCH -A YOUR_ACCOUNT              # Your cluster account
#SBATCH --gpus-per-node=4            # Number of GPUs

PYTHON_BIN="/path/to/conda/envs/fm4npp/bin/python"
```

## Running Experiments

### Method 1: SLURM (Recommended for Clusters)

#### Pretraining
```bash
# Edit submit script with your paths
nano scripts/run/submit_mamba_pretrain.sh

# Submit job
sbatch scripts/run/submit_mamba_pretrain.sh

# Monitor job
squeue -u $USER
tail -f mamba_pretrain_*.out
```

#### Downstream
```bash
# Edit submit script with your paths
nano scripts/run/submit_downstream_mamba.sh

# Submit job
sbatch scripts/run/submit_downstream_mamba.sh

# Monitor job
tail -f mamba_downstream_*.out
```

### Method 2: Direct Execution (Single Node)

#### Pretraining
```bash
python scripts/run/train_mamba_direct.py \
    --mode pretrain \
    --config mamba_5m \
    --run_num run0 \
    --num_gpus 4
```

#### Downstream
```bash
python scripts/run/train_mamba_direct.py \
    --mode downstream \
    --config mamba_5m_downstream \
    --run_num run0 \
    --num_gpus 4
```

### Method 3: Manual Execution

#### Pretraining
```bash
cd FM4NPP_Public

python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
    --yaml_config=scripts/configs/mamba_pretrain.yaml \
    --config=mamba_5m \
    --run_num=run0
```

#### Downstream
```bash
cd FM4NPP_Public/train/downstream

python track_finding_trainer.py \
    --yaml_config=../../scripts/configs/mamba_tracking.yaml \
    --config=mamba_5m_downstream \
    --run_num=run0
```

## Monitoring Training

### Check Logs
```bash
# Pretraining logs
tail -f /path/to/checkpoints/mamba_5m/run0/training.log

# Downstream logs
tail -f /path/to/downstream/logs/performance0.log
```

### Checkpoints
```bash
# Pretrain checkpoints
ls /path/to/checkpoints/mamba_5m/run0/training_checkpoints/

# Downstream checkpoints
ls /path/to/downstream/logs/mamba_5m_downstream/run0/checkpoints/
```

### Tensorboard (Optional)
```bash
# If you add tensorboard logging
tensorboard --logdir=/path/to/checkpoints
```

## Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size in config
batch_size: 128  # instead of 256
local_batch_size: 8  # instead of 16
```

### CUDA Errors
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH=/path/to/FM4NPP_Public:$PYTHONPATH

# Or use absolute imports
python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 ...
```

### Checkpoint Not Found
```bash
# Check checkpoint path exists
ls /path/to/pretrain/checkpoint.tar

# Use absolute paths in config
pretrained_ckpt: /absolute/path/to/checkpoint.tar
```

## Expected Results

### Pretraining (50K steps)
- **Time**: ~24-48 hours on 4x A100 GPUs
- **Final Loss**: ~0.5-1.0 (reconstruction loss)
- **Checkpoint Size**: ~50MB (Mamba 5M)

### Downstream (Track Finding)
- **Time**: ~4-8 hours on 4x A100 GPUs
- **Metrics**:
  - ARI (Adjusted Rand Index): 0.85-0.95
  - Precision: 0.90-0.95
  - Recall: 0.85-0.92

## Model Architectures

### Mamba 5M (~4.6M parameters)
```yaml
embed_dim: 256
num_layers: 12
d_state: 16
d_conv: 4
expand: 2
```

### Mamba2 5M (~5.1M parameters)
```yaml
embed_dim: 256
num_layers: 12
d_state: 128
headdim: 64
ngroups: 1
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{fm4npp2025,
  title={Foundation Models for Particle Physics},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## Support

For questions or issues:
1. Check this guide and README.md
2. Open a GitHub issue
3. Contact: [contact email]

## Running the larger models (m3 / m4 / m6)

The four published checkpoints now each have a config. Note the naming: these are the
repository's internal names and they do **not** match the paper's — see README.md.

| config | width | params | checkpoint | paper calls it |
|---|---|---|---|---|
| `d9_m1_k30_p20` | 256 | 5.3M | `pp_nerf_m1_k30.ckpt` | m3 |
| `d9_m3_k30_p20` | 512 | 21M | `pp_nerf_m3_k30.ckpt` | m4 |
| `d9_m4_k30_p20` | 1024 | 84M | `pp_nerf_m4_k30.ckpt` | m5 |
| `d9_m5_k30_p20` | 1536 | 175M | `pp_nerf_m5_k30.ckpt` | m6 |

### Precompute the frozen-backbone features first

The foundation model is frozen for every downstream task, so it recomputes identical
features on every epoch. Measured on one GB10 at batch 32, sequence length ~2000:

| stage | ms/batch | share of step |
|---|---|---|
| frozen backbone forward | 786 | ~45% |
| dataloading (`num_data_workers: 0`) | 379 | ~22% |
| matcher + loss (x3) | 261 | ~15% |
| adapter fwd/bwd — the only part that trains | 140 | ~8% |

Caching removes the first two. For the 175M model that is roughly **48 min/epoch → 8 min/epoch**.

```bash
python scripts/cache_features.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m5_k30_p20 \
    --checkpoint ./checkpoints/pp_nerf_m5_k30.ckpt \
    --out ./cache/m6_10k --eventnumber 10000

python train/downstream/train_track_finding.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config d9_m5_k30_p20 --eventnumber 10000 --seed 42 \
    --train_batch_size 8 \
    --feature_cache ./cache/m6_10k
```

Cache size is `12 * width * 2` bytes per spacepoint: ~310 GB per 10k events at width 1536,
~53 GB at width 256. The cache is exact — features match a fresh forward to ~3e-08 — but it
is only valid while the backbone is frozen, and it is invalidated by a different checkpoint
or different preprocessing. It cannot be used with LoRA or backbone fine-tuning.

### Two things that will cost you a run if you don't know them

**Batch size.** `--train_batch_size 16` exhausts memory at width 1536 on a 128 GB
unified-memory part. Use 8, or accumulate gradients.

**Do not shorten `max_epochs` for a quick test.** Until recently the cosine cycle length was
hardwired to `max_epochs` and the scheduler steps once per epoch, so lowering it compressed
the entire LR schedule while `warmup_steps` stayed fixed — a 40-epoch run spent 50% of itself
in warmup and then annealed to `min_lr` while the training loss was still falling. Set
`first_cycle_steps` explicitly instead; it is now honoured as its own key. For reference,
tracking runs in the original work reached their best validation loss around **epoch 118-126**.
