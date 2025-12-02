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

# Install Mamba dependencies
pip install mamba-ssm causal-conv1d triton

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

The code expects preprocessed data in memory-mapped format:

```
data_root/
├── features_train/     # RaggedMmap of point features [N_events, N_points, 30]
├── features_test/
├── seg_target_train/   # RaggedMmap of segmentation labels [N_events, N_points]
└── seg_target_test/
```

### Feature Format (30D per point)

Each point has 30 features:
1. Position (3D): x, y, z
2. Momentum (3D): px, py, pz
3. Energy (1D): E
4. Time (1D): t
5. Detector metadata (22D): layer, module, sensor, cell, etc.

### Statistics Files

Normalization statistics should be in `stat_dir`:
- `bin_edges_v3_nbins_8_8_6.pkl`: Binning for features
- `loss_bin_pp.pkl`: Loss binning for track finding
- `loss_weight_pp.pkl`: Loss weights

### Data Preprocessing Script (Example)

```python
import numpy as np
from mmap_ninja import RaggedMmap

def preprocess_data(raw_data_dir, output_dir):
    """
    Convert raw particle data to memory-mapped format.

    Args:
        raw_data_dir: Directory with raw event files
        output_dir: Directory to save preprocessed data
    """
    # Load raw events
    events = load_raw_events(raw_data_dir)  # Your function

    # Extract features (30D per point)
    features = []
    targets = []

    for event in events:
        # Extract point features
        event_features = extract_features(event)  # [N_points, 30]
        event_targets = extract_targets(event)    # [N_points]

        features.append(event_features)
        targets.append(event_targets)

    # Save as RaggedMmap
    RaggedMmap.from_generator(
        out_dir=f"{output_dir}/features_train",
        sample_generator=iter(features),
        batch_size=1000
    )

    RaggedMmap.from_generator(
        out_dir=f"{output_dir}/seg_target_train",
        sample_generator=iter(targets),
        batch_size=1000
    )

    print(f"Preprocessed {len(events)} events to {output_dir}")

# Usage
preprocess_data('/path/to/raw/data', '/path/to/preprocessed/data')
```

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
