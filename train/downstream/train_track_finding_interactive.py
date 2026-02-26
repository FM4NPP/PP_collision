#!/usr/bin/env python3
"""
Interactive downstream tracking training script.
For testing and debugging - runs on single GPU with smaller batch size.

Usage:
    # Run from anywhere in the project:
    cd /path/to/FM4NPP
    python train/downstream/train_track_finding_interactive.py --model longformer

    # Or from train/downstream/:
    cd train/downstream
    python train_track_finding_interactive.py --model longformer

    # Run Linformer with custom settings
    python train_track_finding_interactive.py --model linformer --batch_size 16 --events 10000

    # Run Mamba without pretrain
    python train_track_finding_interactive.py --model mamba --no-pretrain
"""
import os
import sys
import argparse
import gc

import torch

# Get script directory to construct correct paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

# ensure fm4npp modules can be found
sys.path.insert(0, PROJECT_ROOT)

from fm4npp.utils import YParams
from track_finding_trainer import DownstreamTrainer

def main():
    parser = argparse.ArgumentParser(description="Interactive downstream tracking training script")

    # Model selection
    parser.add_argument("--model", type=str, required=True,
                       choices=['longformer', 'linformer', 'mamba'],
                       help="Model type: longformer, linformer, or mamba")

    # Default config path relative to script location
    default_config = os.path.join(PROJECT_ROOT, 'scripts/configs/mamba_tracking.yaml')

    # Paths
    parser.add_argument("--yaml_config", default=default_config,
                       type=str, help="Path to YAML config file")
    parser.add_argument("--root_dir", default='/pscratch/sd/d/dpark1/NPFN/DOWNSTREAM_TEST/',
                       type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs',
                       type=str, help="Global dir to store logging only")

    # Training parameters
    parser.add_argument("--run_num", default='interactive0', type=str,
                       help="Run number/identifier")
    parser.add_argument("--events", default=10000, type=int,
                       help="Number of training events (default: 10000 for quick testing)")
    parser.add_argument("--batch_size", default=16, type=int,
                       help="Training batch size (default: 16 for single GPU)")
    parser.add_argument("--epochs", default=50, type=int,
                       help="Maximum epochs (default: 50)")

    # Model options
    parser.add_argument("--usepretrain", dest="usepretrain", action="store_true",
                       help="Use pretrained model (default)")
    parser.add_argument("--no-pretrain", dest="usepretrain", action="store_false",
                       help="Disable pretrained model")
    parser.set_defaults(usepretrain=True)

    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with more verbose output")

    args = parser.parse_args()

    # Model-specific configurations
    model_configs = {
        'longformer': {
            'config': 'longformer_5m_downstream',
            'mambaversion': 'longformer',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/longformer_5m/try1/training_checkpoints/ckpt_best.tar',
            'description': 'Longformer 5M (256d, 10 layers, window=256)'
        },
        'linformer': {
            'config': 'linformer_5m_downstream',
            'mambaversion': 'linformer',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/linformer_5m/try1/training_checkpoints/ckpt_best.tar',
            'description': 'Linformer 5M (128d, 6 layers, proj_dim=128)'
        },
        'mamba': {
            'config': 'mamba_5m_downstream',
            'mambaversion': 'mamba1',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/mamba_5m/try0/training_checkpoints/ckpt_best.tar',
            'description': 'Mamba 5M (256d, 12 layers, d_state=16)'
        }
    }

    model_cfg = model_configs[args.model]

    # Add config to args namespace (required by DownstreamTrainer)
    args.config = model_cfg['config']
    args.mambaversion = model_cfg['mambaversion']
    args.eventnumber = args.events
    args.train_batch_size = args.batch_size

    # Print configuration
    print("="*80)
    print("INTERACTIVE DOWNSTREAM TRAINING")
    print("="*80)
    print(f"Model: {model_cfg['description']}")
    print(f"Config: {model_cfg['config']}")
    print(f"Training events: {args.events}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Pretrained: {'Yes' if args.usepretrain else 'No (train from scratch)'}")
    if args.usepretrain:
        print(f"Checkpoint: {model_cfg['checkpoint']}")
    print(f"Output dir: {args.root_dir}")
    print(f"Run ID: {args.run_num}")
    print("="*80)
    print()

    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), model_cfg['config'])

    # Override with interactive settings
    params.continue_from_best = True
    params.batch_size = args.batch_size
    params.limit_data = True
    params.limit_size = args.events
    params.valid_batch_size = 1
    params.max_epochs = args.epochs
    params.mambaversion = model_cfg['mambaversion']

    # Set checkpoint path
    if args.usepretrain:
        params.pretrained_ckpt = model_cfg['checkpoint']

    # Log file name
    base_name = f"{args.model}_5m_downstream_{args.events}events_{args.run_num}"
    if args.usepretrain:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"

    # Loss weights
    params.loss_matched_ce_weight = 0.5
    params.loss_unmatched_ce_weight = 0.1
    params.loss_dice_weight = 1
    params.loss_focal_weight = 30
    params.num_embedder_layers = 0

    # Interactive mode settings - always show detailed output
    params.log_to_screen = True

    # Debug settings
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Parameters:")
        print(f"  embed_dim: {params.embed_dim}")
        print(f"  num_layers: {params.num_layers_backbone}")
        print(f"  klen: {params.klen}")
        print(f"  dropout: {params.dropout}")
        print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU available!")
        print("This script requires a GPU to run.")
        return

    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Launch and train
    try:
        print("Initializing trainer...")
        trainer = DownstreamTrainer(params, args)

        print("Launching training...")
        trainer.launch()

        print("Starting training loop...")
        checkpoint_path = None
        trainer.train(
            pretrain=args.usepretrain,
            train_from_checkpoint=False,
            checkpoint_path=checkpoint_path
        )

        print()
        print("="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"Results saved to: {args.root_dir}")
        log_path = os.path.join(params.checkpoint_dir, params.log_file_name)
        print(f"Log file: {log_path}")
        print(f"Checkpoint dir: {params.checkpoint_dir}")

    except KeyboardInterrupt:
        print()
        print("="*80)
        print("⚠️  Training interrupted by user")
        print("="*80)

    except Exception as e:
        print()
        print("="*80)
        print(f"❌ ERROR: Training failed!")
        print(f"Error: {str(e)}")
        print("="*80)
        if args.debug:
            import traceback
            traceback.print_exc()
        raise

    finally:
        # Cleanup
        print()
        print("Cleaning up...")
        if 'trainer' in locals():
            trainer.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
