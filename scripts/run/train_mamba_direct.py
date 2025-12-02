#!/usr/bin/env python3
"""
Direct Training Script (No SLURM)
For single-node or multi-GPU training without job scheduler

Usage:
    # Pretraining
    python scripts/run/train_mamba_direct.py \
        --mode pretrain \
        --config mamba_5m \
        --run_num run0

    # Downstream (track finding)
    python scripts/run/train_mamba_direct.py \
        --mode downstream \
        --config mamba_5m_downstream \
        --run_num run0
"""
import os
import sys
import argparse
import subprocess

def setup_environment():
    """Setup CUDA and distributed training environment."""
    # Set environment variables
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')

    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, project_root)

    return project_root

def run_pretrain(config, run_num, num_gpus=1):
    """Run pretraining."""
    project_root = setup_environment()
    config_file = os.path.join(project_root, 'scripts/configs/mamba_pretrain.yaml')

    cmd = [
        sys.executable,
        '-m', 'torch.distributed.launch',
        f'--nproc_per_node={num_gpus}',
        '-m', 'train.pretrain.nppmamba.train_multi_gpu_mamba1',
        f'--yaml_config={config_file}',
        f'--config={config}',
        f'--run_num={run_num}'
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_downstream(config, run_num, num_gpus=1):
    """Run downstream training."""
    project_root = setup_environment()
    config_file = os.path.join(project_root, 'scripts/configs/mamba_tracking.yaml')
    training_script = os.path.join(project_root, 'train/downstream/track_finding_trainer.py')

    cmd = [
        sys.executable,
        '-m', 'torch.distributed.launch',
        f'--nproc_per_node={num_gpus}',
        training_script,
        f'--yaml_config={config_file}',
        f'--config={config}',
        f'--run_num={run_num}'
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Direct Training Script (No SLURM)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['pretrain', 'downstream'],
                       help='Training mode')
    parser.add_argument('--config', type=str, required=True,
                       help='Config name (e.g., mamba_5m, mamba_5m_downstream)')
    parser.add_argument('--run_num', type=str, default='run0',
                       help='Run identifier')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use')

    args = parser.parse_args()

    print("="*80)
    print(f"FM4NPP Training - Direct Execution")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Run: {args.run_num}")
    print(f"GPUs: {args.num_gpus}")
    print("="*80)
    print()

    try:
        if args.mode == 'pretrain':
            run_pretrain(args.config, args.run_num, args.num_gpus)
        else:
            run_downstream(args.config, args.run_num, args.num_gpus)

        print()
        print("="*80)
        print("✅ Training completed successfully!")
        print("="*80)

    except Exception as e:
        print()
        print("="*80)
        print(f"❌ Training failed: {e}")
        print("="*80)
        sys.exit(1)

if __name__ == '__main__':
    main()
