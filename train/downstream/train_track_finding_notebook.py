"""
Jupyter notebook-friendly downstream training script.
Copy this code into a Jupyter notebook for interactive experimentation.

Example usage in notebook:
    %run train_track_finding_notebook.py
"""

import os
import sys
import gc
import torch

# Setup paths
sys.path.append('../..')
from fm4npp.utils import YParams
from track_finding_trainer import DownstreamTrainer

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Choose model: 'longformer', 'linformer', or 'mamba'
MODEL = 'longformer'

# Training parameters
NUM_EVENTS = 5000      # Reduced for quick testing
BATCH_SIZE = 16        # Smaller batch for single GPU
MAX_EPOCHS = 20        # Fewer epochs for testing
USE_PRETRAIN = True    # Set False to train from scratch
RUN_ID = 'notebook_test1'

# Paths (adjust if needed)
CONFIG_FILE = '../../scripts/configs/mamba_tracking.yaml'
OUTPUT_DIR = '/pscratch/sd/d/dpark1/NPFN/DOWNSTREAM_TEST/'

# ============================================================================
# Model configurations
# ============================================================================

MODEL_CONFIGS = {
    'longformer': {
        'config': 'longformer_5m_downstream',
        'mambaversion': 'longformer',
        'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/longformer_5m/try1/training_checkpoints/ckpt_best.tar',
        'description': 'Longformer 5M (256d, 10 layers, window=256)'
    },
    'linformer': {
        'config': 'linformer_5m_downstream',
        'mambaversion': 'linformer',
        'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/linformer_5m/try0/training_checkpoints/ckpt_best.tar',
        'description': 'Linformer 5M (128d, 6 layers, proj_dim=128)'
    },
    'mamba': {
        'config': 'mamba_5m_downstream',
        'mambaversion': 'mamba1',
        'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/mamba_5m/try0/training_checkpoints/ckpt_best.tar',
        'description': 'Mamba 5M (256d, 12 layers, d_state=16)'
    }
}

# ============================================================================
# Setup and validation
# ============================================================================

def setup_training():
    """Setup training configuration and validate environment."""

    # Validate model choice
    if MODEL not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model: {MODEL}. Choose from: {list(MODEL_CONFIGS.keys())}")

    model_cfg = MODEL_CONFIGS[MODEL]

    # Print configuration
    print("="*80)
    print("NOTEBOOK DOWNSTREAM TRAINING")
    print("="*80)
    print(f"Model: {model_cfg['description']}")
    print(f"Config: {model_cfg['config']}")
    print(f"Training events: {NUM_EVENTS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Pretrained: {'Yes' if USE_PRETRAIN else 'No (train from scratch)'}")
    if USE_PRETRAIN:
        print(f"Checkpoint: {model_cfg['checkpoint']}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Run ID: {RUN_ID}")
    print("="*80)
    print()

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ ERROR: No GPU available! This requires a GPU to run.")

    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    return model_cfg

def create_params(model_cfg):
    """Create training parameters."""

    # Initialize parameters
    params = YParams(os.path.abspath(CONFIG_FILE), model_cfg['config'])

    # Override with notebook settings
    params.continue_from_best = True
    params.batch_size = BATCH_SIZE
    params.limit_data = True
    params.limit_size = NUM_EVENTS
    params.valid_batch_size = 1
    params.max_epochs = MAX_EPOCHS
    params.mambaversion = model_cfg['mambaversion']
    params.log_to_screen = True  # Show logs in notebook

    # Set checkpoint
    if USE_PRETRAIN:
        params.pretrained_ckpt = model_cfg['checkpoint']

    # Log file name
    base_name = f"{MODEL}_5m_downstream_{NUM_EVENTS}events_{RUN_ID}"
    if USE_PRETRAIN:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"

    # Loss weights
    params.loss_matched_ce_weight = 0.5
    params.loss_unmatched_ce_weight = 0.1
    params.loss_dice_weight = 1
    params.loss_focal_weight = 30
    params.num_embedder_layers = 0

    # Print key parameters
    print(f"Model parameters:")
    print(f"  embed_dim: {params.embed_dim}")
    print(f"  num_layers: {params.num_layers_backbone}")
    print(f"  klen: {params.klen}")
    print(f"  dropout: {params.dropout}")
    print()

    return params

def create_args(model_cfg):
    """Create argument namespace for trainer."""

    class Args:
        def __init__(self):
            self.yaml_config = CONFIG_FILE
            self.config = model_cfg['config']
            self.run_num = RUN_ID
            self.root_dir = OUTPUT_DIR
            self.global_log_dir = 'globallogs'
            self.eventnumber = NUM_EVENTS
            self.usepretrain = USE_PRETRAIN
            self.train_batch_size = BATCH_SIZE
            self.mambaversion = model_cfg['mambaversion']

    return Args()

# ============================================================================
# Main training function
# ============================================================================

def train_model():
    """Run the training."""

    try:
        # Setup
        print("Initializing...")
        model_cfg = setup_training()
        params = create_params(model_cfg)
        args = create_args(model_cfg)

        # Create trainer
        print("Creating trainer...")
        trainer = DownstreamTrainer(params, args)

        # Launch
        print("Launching training...")
        trainer.launch()

        # Train
        print("Starting training loop...")
        print("(This will take a while, watch for epoch progress)")
        print()

        trainer.train(
            pretrain=USE_PRETRAIN,
            train_from_checkpoint=False,
            checkpoint_path=None
        )

        print()
        print("="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Log file: {params.log_file_name}")

        return trainer

    except KeyboardInterrupt:
        print()
        print("="*80)
        print("⚠️  Training interrupted by user")
        print("="*80)
        return None

    except Exception as e:
        print()
        print("="*80)
        print(f"❌ ERROR: Training failed!")
        print(f"Error: {str(e)}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Cleanup
        print()
        print("Cleaning up...")
        if 'trainer' in locals():
            trainer.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleanup complete")

# ============================================================================
# Run training (if executed as script)
# ============================================================================

if __name__ == "__main__":
    print("Starting training from script...")
    print("(If running in notebook, you can also call train_model() directly)")
    print()

    trainer = train_model()

    if trainer is not None:
        print()
        print("Training object available as 'trainer' variable")
        print("You can use it for further analysis or inference")
