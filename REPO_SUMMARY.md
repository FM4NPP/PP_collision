# FM4NPP Public Repository - Summary

**Created**: December 2, 2025
**Purpose**: Publication-ready minimal implementation
**Location**: `/Users/davidpark/Documents/Claude/FM4NPP_Public`

## What's Included

### Core Functionality
✅ **Mamba/Mamba2 Pretraining**
- Full training code for state space models
- Multi-GPU distributed training support
- Memory-mapped data loading for efficiency

✅ **Track Reconstruction Downstream Task**
- Fine-tuning pretrained models
- Attention-based downstream heads
- Track finding with segmentation

### Code Structure
```
FM4NPP_Public/
├── README.md                    # Main documentation
├── SETUP.md                     # Detailed setup guide
├── PUBLICATION_CHECKLIST.md     # Pre-publication tasks
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── example_usage.py             # Usage examples
│
├── fm4npp/
│   ├── models/
│   │   ├── mambagpt.py         # Mamba1 implementation
│   │   ├── mamba2.py           # Mamba2 implementation
│   │   └── embed.py            # Embedding layers
│   ├── datasets/
│   │   └── dataset.py          # Data loading
│   └── utils.py                # Utilities
│
├── train/
│   ├── pretrain/
│   │   └── nppmamba/
│   │       └── train_multi_gpu_mamba1.py
│   └── downstream/
│       ├── track_finding_trainer.py
│       └── model.py            # Downstream heads
│
└── scripts/
    ├── configs/
    │   ├── mamba_pretrain.yaml
    │   └── mamba_tracking.yaml
    └── run/
        ├── submit_mamba_pretrain.sh
        ├── submit_downstream_mamba.sh
        └── train_mamba_direct.py
```

## What's Anonymized

### Paths
- ❌ `/global/u1/d/dpark1` → ✅ `/path/to`
- ❌ `/pscratch/sd/d/dpark1` → ✅ `/path/to/scratch`
- ❌ `/global/cfs/cdirs/m4722` → ✅ `/path/to/data`

### Identifiers
- ❌ Account `m4722` → ✅ `YOUR_ACCOUNT`
- ❌ User `dpark1` → ✅ `USER` or removed
- ❌ Real names → ✅ `[Authors]` placeholder
- ❌ Emails → ✅ Generic placeholders

### Cluster-Specific
- ❌ Perlmutter-specific settings → ✅ Generic SLURM examples
- ❌ Module load commands → ✅ Conda environment instructions
- ❌ Specific job configurations → ✅ Adaptable templates

## What's NOT Included

To keep the repository minimal:

❌ **Alternative Models**
- Longformer, Linformer (not core contribution)
- Other baseline models

❌ **Additional Downstream Tasks**
- PID classification
- Noise tagging
- Other tasks beyond track finding

❌ **Experimental Code**
- Ablation studies code
- Hyperparameter sweeps
- Debugging scripts

❌ **Data Files**
- Preprocessed datasets
- Statistics files
- Example events

❌ **Checkpoints**
- Pretrained model weights
- Downstream model weights

❌ **Analysis Code**
- Plotting scripts
- Result analysis notebooks
- Performance comparison tools

❌ **Internal Documentation**
- Team-specific notes
- Private development docs
- Meeting notes

## Files Requiring Updates Before Publication

See `PUBLICATION_CHECKLIST.md` for complete list.

**Critical Updates Needed:**
1. **CITATION.cff**: Add real author names, DOI, journal info
2. **README.md**: Add real contact info, funding sources
3. **LICENSE**: Add copyright holder
4. **Config files**: Verify parameters match paper
5. **All .py files**: Check for any remaining personal info

## Model Configurations

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

## Usage Examples

### Pretraining
```bash
# With SLURM
sbatch scripts/run/submit_mamba_pretrain.sh

# Direct execution
python scripts/run/train_mamba_direct.py \
    --mode pretrain \
    --config mamba_5m \
    --run_num run0 \
    --num_gpus 4
```

### Downstream
```bash
# With SLURM
sbatch scripts/run/submit_downstream_mamba.sh

# Direct execution
python scripts/run/train_mamba_direct.py \
    --mode downstream \
    --config mamba_5m_downstream \
    --run_num run0 \
    --num_gpus 4
```

## Testing the Repository

Before publication, test on a clean system:

```bash
# 1. Clone repository
git clone <repo-url>
cd FM4NPP_Public

# 2. Setup environment
conda create -n fm4npp python=3.10
conda activate fm4npp
pip install -r requirements.txt

# 3. Update paths in configs
nano scripts/configs/mamba_pretrain.yaml
nano scripts/configs/mamba_tracking.yaml

# 4. Test training
python scripts/run/train_mamba_direct.py --mode pretrain --config mamba_5m
```

## Next Steps

1. **Review all files** for any remaining personal information
2. **Complete PUBLICATION_CHECKLIST.md** items
3. **Test on fresh environment** to verify setup instructions
4. **Get co-author approval** before publication
5. **Create GitHub repository** and push code
6. **Consider releasing**:
   - Sample dataset
   - Pretrained checkpoints
   - Jupyter notebooks with examples
7. **Update paper** with repository link
8. **Monitor issues** and maintain after publication

## Differences from Original Repository

### Removed
- Multi-model comparison code (Longformer, Linformer)
- Additional downstream tasks (PID, noise tagging)
- Interactive training scripts
- Cluster-specific optimizations
- Personal development tools
- Experimental features
- Internal documentation

### Simplified
- Configuration files (only essential parameters)
- Run scripts (generic SLURM templates)
- Documentation (focused on reproducibility)

### Added
- Clear setup instructions
- Example usage scripts
- Publication checklist
- Citation metadata
- Comprehensive README

## File Sizes

```
Total repository size: ~50KB (code only)
With sample data: ~1GB (if added)
With checkpoints: ~100MB per model (if added)
```

## Maintenance

After publication:
- Monitor GitHub issues
- Respond to questions
- Fix bugs as reported
- Consider adding tutorials/notebooks
- Update for new PyTorch/Mamba versions
- Keep CHANGELOG updated

## Contact

For questions about this repository:
- See `PUBLICATION_CHECKLIST.md` for items to update
- Original repository: `/Users/davidpark/Documents/Claude/FM4NPP`
- Public repository: `/Users/davidpark/Documents/Claude/FM4NPP_Public`

---

**Note**: Delete or move this file to internal docs before final publication.
