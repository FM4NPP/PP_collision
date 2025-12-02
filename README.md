# Foundation Models for Particle Physics (FM4NPP)

**Publication Repository**: Minimal implementation for reproducibility

This repository contains the essential code for:
1. **Pretraining**: State space models (Mamba, Mamba2) on particle physics data
2. **Downstream Task**: Track reconstruction using pretrained representations

**Paper**: [Foundation Models for Particle Physics](https://arxiv.org/abs/2508.14087)

## Repository Structure

```
FM4NPP_Public/
├── fm4npp/
│   ├── models/          # Model architectures (Mamba, Mamba2)
│   ├── datasets/        # Data loading and preprocessing
│   └── utils.py         # Utilities and configuration
├── train/
│   ├── pretrain/
│   │   └── nppmamba/    # Pretraining scripts
│   └── downstream/      # Track reconstruction training
├── scripts/
│   ├── configs/         # Configuration files
│   └── run/             # SLURM submission scripts
└── README.md
```

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.4+
- CUDA 12.1+
- mamba-ssm (for Mamba models)
- causal-conv1d
- triton

### Setup
```bash
# Create conda environment
conda create -n fm4npp python=3.10
conda activate fm4npp

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Mamba dependencies
pip install mamba-ssm causal-conv1d
pip install triton

# Install other requirements
pip install pyyaml numpy scipy tqdm mmap-ninja
```

## Usage

### 1. Pretraining

Train Mamba or Mamba2 models on particle physics data:

```bash
# Configure paths in scripts/configs/mamba_pretrain.yaml
# Edit: data_root, checkpoint_dir, stat_dir

# Submit pretraining job (SLURM)
sbatch scripts/run/submit_mamba_pretrain.sh

# Or run directly
python -m train.pretrain.nppmamba.train_multi_gpu \
    --yaml_config=scripts/configs/mamba_pretrain.yaml \
    --config=mamba_5m \
    --run_num=run0
```

### 2. Track Reconstruction (Downstream)

Fine-tune pretrained model for track finding:

```bash
# Configure paths in scripts/configs/mamba_tracking.yaml
# Edit: data_root, pretrained_ckpt, checkpoint_dir

# Submit downstream job (SLURM)
sbatch scripts/run/submit_downstream_mamba.sh

# Or run directly
python train/downstream/track_finding_trainer.py \
    --yaml_config=scripts/configs/mamba_tracking.yaml \
    --config=mamba_5m_downstream \
    --run_num=run0
```

## Configuration

### Key Parameters

**Mamba 5M Model**:
- `embed_dim`: 256
- `num_layers`: 12
- `d_state`: 16 (state space dimension)
- `d_conv`: 4 (convolutional kernel size)
- `expand`: 2 (expansion factor)

**Mamba2 5M Model**:
- `embed_dim`: 256
- `num_layers`: 12
- `d_state`: 128 (state space dimension)
- `headdim`: 64
- `ngroups`: 1

**Training**:
- `batch_size`: 256 (distributed across GPUs)
- `max_lr`: 2e-4
- `warmup_steps`: 1000
- `total_steps`: 50000

## Dataset

### TPCpp-10M Dataset

We provide the preprocessed dataset used in our paper on Zenodo:

**Dataset**: [TPCpp-10M: Simulated proton-proton collisions in Time Projection Chamber for AI Foundation Models](https://doi.org/10.5281/zenodo.16970029)

**Dataset Statistics**:
- **Unlabeled data**: 10M events (100 files) for pretraining
- **Labeled training**: 70k events for downstream tasks
- **Labeled validation**: 13k events
- **Labeled test**: 7k events
- **Total size**: ~118.5 GB (compressed)

**Data Format**: NumPy compressed format (.npz)

### Download Dataset

```bash
# Download from Zenodo
wget https://zenodo.org/records/16970029/files/TPCpp-10M.tar.gz

# Extract
tar -xzf TPCpp-10M.tar.gz

# Dataset structure after extraction:
TPCpp-10M/
├── unlabeled/        # 10M events for pretraining (100 files)
├── labeled_train/    # 70k labeled events (7 file sets)
│   ├── spacepoints/
│   ├── track_ids/
│   ├── particle_ids/
│   └── noise_tags/
├── labeled_val/      # 13k validation events
└── labeled_test/     # 7k test events
```

### Data Format Details

Each spacepoint includes:
- **Position**: (x, y, z) coordinates in TPC
- **Energy**: Energy deposition at the point
- **Labels** (for downstream tasks):
  - Track IDs: Segmentation labels for track reconstruction
  - Particle IDs: 5 classes (electron, photon, pion, kaon, proton)
  - Noise tags: Binary labels (signal/noise)

**Feature dimensions**: 30D per point
- Position: (x, y, z)
- Momentum: (px, py, pz)
- Energy, time, detector metadata

### Usage with Code

After downloading, update config paths:

```yaml
# In scripts/configs/mamba_pretrain.yaml
data_root: /path/to/TPCpp-10M/unlabeled
stat_dir: /path/to/TPCpp-10M/statistics

# In scripts/configs/mamba_tracking.yaml
data_root: /path/to/TPCpp-10M/labeled_train
data_root_test: /path/to/TPCpp-10M/labeled_test
```

See `demo.ipynb` in the dataset for data exploration and visualization examples.

## Citation

If you use this code or dataset, please cite both papers:

```bibtex
@article{fm4npp2025,
  title={Foundation Models for Particle Physics},
  author={Li, Shuhang and Ren, Yihui and Luo, Xihaier and Park, David and Yoo, Shinjae},
  journal={arXiv preprint arXiv:2508.14087},
  year={2025}
}

@article{tpcpp10m2025,
  title={TPCpp-10M: Simulated proton-proton collisions in a Time Projection Chamber for AI Foundation Models},
  author={Li, Shuhang and Huang, Yi and Park, David and Luo, Xihaier and Yu, Haiwang and Go, Yeonju and Pinkenburg, Christopher and Lin, Yuewei and Yoo, Shinjae and Osborn, Joseph and Roland, Christof and Huang, Jin and Ren, Yihui},
  journal={arXiv preprint arXiv:2509.05792},
  year={2025}
}
```

Papers:
- Model: https://arxiv.org/abs/2508.14087
- Dataset: https://arxiv.org/abs/2509.05792

## Contact

For questions or issues, please open a GitHub issue.
