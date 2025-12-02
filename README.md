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

## Data Format

The code expects preprocessed data in memory-mapped format:

```
data_root/
├── features_train/    # Point cloud features (N_events × N_points × D_features)
├── features_test/
├── seg_target_train/  # Segmentation labels for track finding
└── seg_target_test/
```

**Feature dimensions**: 30D per point
- Position: (x, y, z)
- Momentum: (px, py, pz)
- Energy, time, detector metadata

## Citation

If you use this code, please cite our paper:

```bibtex
@article{fm4npp2025,
  title={Foundation Models for Particle Physics},
  author={Li, Shuhang and Ren, Yihui and Luo, Xihaier and Park, David and Yoo, Shinjae},
  journal={arXiv preprint arXiv:2508.14087},
  year={2025}
}
```

Paper: https://arxiv.org/abs/2508.14087

## Contact

For questions or issues, please:
- Open a GitHub issue
- Contact: dpark1@bnl.gov

## Acknowledgments

The authors would like to express their sincere gratitude to the sPHENIX Collaboration for sharing the simulation data and experimental knowledge, as well as Jubin Choi, Abhay Deshpande, Alexei Klimentov, Michael Begel, Torre Wenaus, Nicholas D'Imperio, James Dunlop, and John Hill from Brookhaven National Lab for their valuable support and feedback.

This work was supported by the Laboratory Directed Research and Development (LDRD) Program at Brookhaven National Laboratory, LDRD 25-045, which is operated and managed for the U.S. Department of Energy (DOE) Office of Science by Brookhaven Science Associates under contract No. DE-SC0012704. Shuhang Li was partially supported by the DOE Office of Science through the Office of Nuclear Physics under Award No. DE-FG02-86ER40281. Yihui Ren, Xihaier Luo and Shinjae Yoo were partially supported by the DOE Office of Science through the Office of Advanced Scientific Computing Research and the Scientific Discovery through Advanced Computing (SciDAC) program.

This research also utilized resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility, under NERSC Award No. DDRERCAP-m4722. The authors are grateful to the NERSC staff for their support, particularly Shashank Subramanian and Wahid Bhimji.
