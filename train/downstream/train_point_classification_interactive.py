#!/usr/bin/env python3
"""
Interactive downstream PID classification and noise tagging training script.
For testing and debugging - runs on single GPU with smaller batch size.

Usage:
    # Run from anywhere in the project:
    cd /path/to/FM4NPP
    python train/downstream/train_point_classification_interactive.py --model longformer --task pid

    # Or from train/downstream/:
    cd train/downstream
    python train_point_classification_interactive.py --model longformer --task pid

    # Run noise tagging with Linformer
    python train_point_classification_interactive.py --model linformer --task noise --batch_size 16

    # Run Mamba without pretrain
    python train_point_classification_interactive.py --model mamba --task pid --no-pretrain
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
from point_classification_trainer import DownstreamTrainer

def main():
    parser = argparse.ArgumentParser(description="Interactive downstream point classification training script")

    # Model selection
    parser.add_argument("--model", type=str, required=True,
                       choices=['longformer', 'linformer', 'mamba'],
                       help="Model type: longformer, linformer, or mamba")

    # Task selection
    parser.add_argument("--task", type=str, required=True,
                       choices=['pid', 'noise'],
                       help="Task type: pid (particle ID classification) or noise (noise tagging)")

    # Default config paths relative to script location
    default_pid_config = os.path.join(PROJECT_ROOT, 'scripts/configs/mamba_pointclass.yaml')
    default_noise_config = os.path.join(PROJECT_ROOT, 'scripts/configs/mamba_noiseid.yaml')

    # Paths
    parser.add_argument("--yaml_config_pid", default=default_pid_config,
                       type=str, help="Path to YAML config file for PID task")
    parser.add_argument("--yaml_config_noise", default=default_noise_config,
                       type=str, help="Path to YAML config file for noise task")
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
    parser.add_argument("--epochs", default=100, type=int,
                       help="Maximum training steps (default: 100)")

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

    # Task-specific configurations
    task_configs = {
        'pid': {
            'yaml': args.yaml_config_pid,
            'description': 'Particle ID Classification (5 classes)',
            'num_classes': 5
        },
        'noise': {
            'yaml': args.yaml_config_noise,
            'description': 'Noise Tagging (2 classes: noise/valid)',
            'num_classes': 2
        }
    }

    # Model-specific configurations
    model_configs = {
        'longformer': {
            'config_suffix': '5m',
            'mambaversion': 'longformer',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/longformer_5m/try1/training_checkpoints/ckpt_best.tar',
            'description': 'Longformer 5M (256d, 10 layers, window=256)'
        },
        'linformer': {
            'config_suffix': '5m',
            'mambaversion': 'linformer',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/linformer_5m/try1/training_checkpoints/ckpt_best.tar',
            'description': 'Linformer 5M (128d, 6 layers, proj_dim=128)'
        },
        'mamba': {
            'config_suffix': '5m',
            'mambaversion': 'mamba1',
            'checkpoint': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/mamba_5m/try0/training_checkpoints/ckpt_best.tar',
            'description': 'Mamba 5M (256d, 12 layers, d_state=16)'
        }
    }

    task_cfg = task_configs[args.task]
    model_cfg = model_configs[args.model]

    # Build config name
    task_suffix = 'pid' if args.task == 'pid' else 'noise'
    config_name = f"{args.model}_{model_cfg['config_suffix']}_{task_suffix}"

    # Add required args attributes
    args.config = config_name
    args.mambaversion = model_cfg['mambaversion']
    args.eventnumber = args.events
    args.train_batch_size = args.batch_size

    # Print configuration
    print("="*80)
    print("INTERACTIVE DOWNSTREAM POINT CLASSIFICATION TRAINING")
    print("="*80)
    print(f"Model: {model_cfg['description']}")
    print(f"Task: {task_cfg['description']}")
    print(f"Config: {config_name}")
    print(f"Training events: {args.events}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max training steps: {args.epochs}")
    print(f"Pretrained: {'Yes' if args.usepretrain else 'No (train from scratch)'}")
    if args.usepretrain:
        print(f"Checkpoint: {model_cfg['checkpoint']}")
    print(f"Output dir: {args.root_dir}")
    print(f"Run ID: {args.run_num}")
    print("="*80)
    print()

    # Initialize parameters
    yaml_config = task_cfg['yaml']
    params = YParams(os.path.abspath(yaml_config), config_name)

    # Override with interactive settings
    params.continue_from_best = True
    params.batch_size = args.batch_size
    params.limit_data = True
    params.limit_size = args.events
    params.valid_batch_size = 1
    params.total_steps = args.epochs
    params.mambaversion = model_cfg['mambaversion']
    params.num_output_classes = task_cfg['num_classes']
    params.task = 'pid' if args.task == 'pid' else 'nid'

    # Set checkpoint path
    if args.usepretrain:
        params.pretrained_ckpt = model_cfg['checkpoint']

    # Log file name
    base_name = f"{args.model}_5m_{args.task}_{args.events}events_{args.run_num}"
    if args.usepretrain:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"

    # Interactive mode settings - always show detailed output
    params.log_to_screen = True
    params.num_embedder_layers = 0

    # Debug settings
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Parameters:")
        print(f"  embed_dim: {params.embed_dim}")
        print(f"  num_layers: {params.num_layers_backbone}")
        print(f"  klen: {params.klen}")
        print(f"  dropout: {params.dropout}")
        print(f"  task: {params.task}")
        print(f"  num_output_classes: {params.num_output_classes}")
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
