#!/usr/bin/env python3
"""
Example Usage of FM4NPP Models

This script demonstrates how to:
1. Load pretrained Mamba models
2. Create downstream models
3. Run inference on particle data
"""

import torch
import numpy as np
from fm4npp.models.mambagpt import Mamba1GPT, Mamba2GPT
from train.downstream.model import MambaAttentionHead

def example_pretrain_inference():
    """Example: Load pretrained model and run inference."""
    print("="*80)
    print("Example 1: Pretrained Model Inference")
    print("="*80)

    # Model configuration
    embed_dim = 256
    num_layers = 12
    d_state = 16
    klen = 30

    # Create Mamba model
    model = Mamba1GPT(
        embed_dim=embed_dim,
        num_layers=num_layers,
        d_state=d_state,
        d_conv=4,
        expand=2,
        klen=klen,
        dropout=0.1,
        embed_method='additive',
        pe_method='nerf'
    )

    # Load pretrained checkpoint (update path)
    checkpoint_path = '/path/to/pretrain/checkpoint.tar'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    except:
        print(f"⚠ Checkpoint not found, using randomly initialized model")

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"✓ Model on device: {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create dummy input data
    batch_size = 2
    num_points = 100
    input_dim = 30  # 30D features per point

    # Random point cloud data
    x = torch.randn(batch_size, input_dim, num_points).to(device)

    # Run inference
    with torch.no_grad():
        output = model(x)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print()

def example_downstream_inference():
    """Example: Load downstream model and run track finding."""
    print("="*80)
    print("Example 2: Downstream Track Finding")
    print("="*80)

    # Backbone configuration
    embed_dim = 256
    num_layers = 12
    d_state = 16
    klen = 30

    # Create backbone
    backbone = Mamba1GPT(
        embed_dim=embed_dim,
        num_layers=num_layers,
        d_state=d_state,
        d_conv=4,
        expand=2,
        klen=klen,
        dropout=0.1,
        embed_method='additive',
        pe_method='nerf'
    )

    # Load pretrained weights
    checkpoint_path = '/path/to/pretrain/checkpoint.tar'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        backbone.load_state_dict(checkpoint['model_state'])
        print(f"✓ Loaded pretrained backbone")
    except:
        print(f"⚠ Using randomly initialized backbone")

    # Create downstream head
    input_dim = 30
    downstream_head = MambaAttentionHead(
        embed_dim=embed_dim,
        input_dim=input_dim,
        num_heads=4,
        num_layers_decoder=2,
        num_layers_encoder=2,
        dropout=0.1
    )

    # Move to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbone.to(device)
    downstream_head = downstream_head.to(device)

    backbone.eval()
    downstream_head.eval()

    print(f"✓ Backbone parameters: {sum(p.numel() for p in backbone.parameters()) / 1e6:.2f}M")
    print(f"✓ Head parameters: {sum(p.numel() for p in downstream_head.parameters()) / 1e6:.2f}M")

    # Create dummy input
    batch_size = 2
    num_points = 100
    x = torch.randn(batch_size, input_dim, num_points).to(device)

    # Run inference
    with torch.no_grad():
        # Extract features from backbone
        features = backbone(x)

        # Run downstream head
        track_predictions = downstream_head(features)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Predictions shape: {track_predictions.shape}")
    print()

def example_mamba2():
    """Example: Using Mamba2 model."""
    print("="*80)
    print("Example 3: Mamba2 Model")
    print("="*80)

    # Create Mamba2 model
    model = Mamba2GPT(
        embed_dim=256,
        num_layers=12,
        d_state=128,
        headdim=64,
        ngroups=1,
        klen=30,
        dropout=0.1,
        embed_method='additive',
        pe_method='nerf'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"✓ Mamba2 model created")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Run inference
    batch_size = 2
    num_points = 100
    x = torch.randn(batch_size, 30, num_points).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"✓ Output shape: {output.shape}")
    print()

def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("FM4NPP Example Usage")
    print("="*80 + "\n")

    try:
        example_pretrain_inference()
        example_downstream_inference()
        example_mamba2()

        print("="*80)
        print("✅ All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Update checkpoint paths in this script before running.")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
