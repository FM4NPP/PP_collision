#!/usr/bin/env python3
"""
Example Usage of FM4NPP Models

Rewritten against the API the code actually exposes. The previous version could not
run at all: it imported a class named Mamba2GPT (the class is MambaGPT), passed
embed_method='additive' (the assert allows only 'concat' or 'add'), passed
num_layers_decoder/num_layers_encoder to MambaAttentionHead (which accepts neither),
fed tensors shaped (B, 30, N) to an embedder that wants (B, N, 4), and called
backbone(x) when downstream features come from backbone(x, return_z=True). Every
failure was swallowed by a blanket try/except that printed a generic error, which
made broken code look like a path-configuration problem.

Run:
    pip install -r requirements.txt
    python example_usage.py --checkpoint /path/to/pp_nerf_m1_k30.ckpt
"""
import argparse
import os
import sys

import torch

from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT
# NOTE: the track-finding trainer imports its head from trackinghead.py, NOT model.py.
# train/downstream/model.py holds a same-named class used by the multitask/joint heads.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train', 'downstream'))
from trackinghead import MambaAttentionHead

# The released checkpoints. NOTE the naming trap: this repo's "m1" is the PAPER's "m3".
#   repo m1 = width  256 =  5.3M params = paper m3   (pp_nerf_m1_k30.ckpt)
#   repo m3 = width  512 =   21M params = paper m4   (pp_nerf_m3_k30.ckpt)
#   repo m4 = width 1024 =   84M params = paper m5   (pp_nerf_m4_k30.ckpt)
#   repo m5 = width 1536 =  188M params = paper m6   (pp_nerf_m5_k30.ckpt)
CONFIGS = {
    'm1': dict(embed_dim=256,  d_state=16),
    'm3': dict(embed_dim=512,  d_state=32),
    'm4': dict(embed_dim=1024, d_state=64),
    'm5': dict(embed_dim=1536, d_state=96),
}


def build_backbone(embed_dim, d_state, num_layers=12, klen=30, mambaversion='mamba2'):
    """The released checkpoints are Mamba2 (MambaGPT), not Mamba1.

    You can verify this from the checkpoint itself: Mamba2 blocks carry lin_B/lin_C
    and a per-head A_log, which mamba_ssm's Mamba1 block does not have.
    """
    cls = MambaGPT if mambaversion == 'mamba2' else Mamba1GPT
    return cls(
        embed_dim=embed_dim, num_layers=num_layers,
        d_state=d_state, d_conv=4, expand=2, klen=klen,
        dropout=0.0,               # backbone dropout stays 0 for the released weights
        embed_method='add',        # 'add' or 'concat' -- the backbone was trained with 'add'
        pe_method='nerf',
    )


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model_state']
    # checkpoints were saved from a DistributedDataParallel wrapper
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True), None
    return ckpt.get('iters'), ckpt.get('epoch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='e.g. pp_nerf_m1_k30.ckpt')
    ap.add_argument('--size', default='m1', choices=sorted(CONFIGS))
    args = ap.parse_args()

    cfg = CONFIGS[args.size]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}   backbone: {args.size} (width {cfg["embed_dim"]})')

    backbone = build_backbone(**cfg)
    if args.checkpoint:
        iters, epoch = load_checkpoint(backbone, args.checkpoint)
        print(f'loaded {args.checkpoint} (iters={iters}, epoch={epoch})')
    else:
        print('no --checkpoint given: using randomly initialised weights')
    backbone = backbone.to(device).eval()
    print(f'backbone params: {sum(p.numel() for p in backbone.parameters())/1e6:.2f}M')

    # The downstream head consumes the backbone's PER-LAYER hidden states, so it needs
    # num_feature_layers == the backbone's layer count.
    head = MambaAttentionHead(
        input_dim=cfg['embed_dim'],
        num_layers=0,
        num_embedder_layers=0,
        d_state=64, d_conv=4, expand=2,
        num_feature_layers=12,
        num_prototypes=150,
        dropout=0.1,
        embed_method='concat',     # the downstream head uses 'concat' (see the trainer)
    ).to(device).eval()
    print(f'head params:     {sum(p.numel() for p in head.parameters())/1e6:.2f}M')

    # Input is (B, N, 4) = (E, x, y, z) per spacepoint, already normalised by the dataset.
    # Not 30 features -- README/SETUP.md were wrong about that; the published data is 4D.
    B, N = 2, 512
    x = torch.randn(B, N, 4, device=device)
    padding_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    with torch.no_grad():
        # return_z=True yields the per-layer features the head expects
        _, per_layer, _ = backbone(x, return_z=True)
        feature = torch.stack(per_layer)                 # (L, B, N, D)
        out = head(x, feature, pretrain=True, padding_mask=padding_mask)

    print(f'input:        {tuple(x.shape)}')
    print(f'per-layer:    {tuple(feature.shape)}')
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f'  {k:20s} {tuple(v.shape)}')
        elif isinstance(v, list):
            print(f'  {k:20s} list of {len(v)}')
    print('\nclass_probs is (B, n_prototypes, 2) = track vs no-object;')
    print('mask_probs  is (B, N, n_prototypes)  = per-point assignment to each track query.')
    print('\nOK')


if __name__ == '__main__':
    main()
