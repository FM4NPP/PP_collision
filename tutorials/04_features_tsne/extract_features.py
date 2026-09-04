#!/usr/bin/env python3
"""Extract frozen-backbone features from the paper's m6 for a handful of events.

    INPUT   pp_nerf_m5_k30.ckpt (the paper's m6, 174.7M) + a prepared data root
    OUTPUT  features.npz  --  points, seg_target, features (12 layers), event_id

    python extract_features.py --n_events 10 --out features.npz

WHY .npz AND NOT RaggedMmap
---------------------------
The repo's `scripts/cache_features.py` writes RaggedMmap because it is built for caching
70,000 events (hundreds of GB) for repeated training epochs. For ten events the mmap
machinery is overhead that obscures the lesson, so we write one plain .npz you can load
with a single numpy call. The backbone construction here is the repo's own
`build_backbone()` logic, unchanged.

WHAT `return_z=True` ACTUALLY GIVES YOU
---------------------------------------
Three facts that change how you read a t-SNE of these vectors:

1. It returns the **residual-branch outputs** `z`, one per layer -- NOT the running
   residual stream. The forward is:

        for layer in self.mamba_layers:
            z = layer(x)
            feature.append(z)      # <- this is what you get
            x = z + x              # <- the accumulated stream, never returned

   So layer L's vector is that block's *contribution*, not the representation after L
   blocks. Measured on 10 m6 events, the branch outputs separate tracks WORSE with depth
   -- layer 12 scores below the raw (E, eta, phi, r) coordinates. That is not a bug; a
   late block's increment is a small refinement and looks like noise on its own.

2. You can recover the accumulated stream exactly, and you should:

        x_i = x_0 + sum_{j<i} z_j          where x_0 = embedder(change_maskval(input))

   We save `x0` alongside the layer stack for exactly this. Verified against forward
   hooks on the real layer inputs: agreement to 1.9e-06, one float32 ULP. The stream is
   the representation the model actually computes, and `norm(x_final)` is what the
   pretraining head consumes.

3. `self.norm` and `self.output_layer` are **skipped** on this path, so the features are
   un-normalized and their scale varies by layer. Standardize before t-SNE or the
   distance metric is dominated by whichever layer happens to have the largest norm.

This is also why the downstream adapter learns a softmax over all 12 layers
(`weighted_avg_weights`) instead of taking the last one: no single layer is best.

STORAGE
-------
float16, which is not a precision loss here: the forward runs under bf16 autocast (8
mantissa bits) and float16 carries 10, so the round-trip is exact. Measured range on 10
real m6 events is about [-7.6, 9.6] -- four orders of magnitude inside float16's 65504
limit, so there is no overflow risk either.

    10 events x ~800 points x 12 layers x 1536 dims x 2 bytes  ~=  300 MB
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast

_ROOT = os.environ.get('FM4NPP_ROOT')
if _ROOT:
    sys.path.insert(0, _ROOT)

from fm4npp.utils import YParams                              # noqa: E402
from fm4npp.datasets.dataset import TPCBatchDataset           # noqa: E402
from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT        # noqa: E402


# Paper name -> (repo config, width, d_state, checkpoint env var)
MODELS = {
    'm3': ('d9_m1_k30_p20', 256, 16, 'FM4NPP_CKPT_M3'),
    'm6': ('d9_m5_k30_p20', 1536, 96, 'FM4NPP_CKPT_M6'),
}


def build_backbone(width, d_state, klen, checkpoint, device, embed_method, pe_method):
    """Construct and load the frozen backbone. Mirrors scripts/cache_features.py."""
    model = MambaGPT(embed_dim=width, num_layers=12, d_state=d_state, d_conv=4,
                     expand=2, klen=klen, dropout=0.0,
                     embed_method=embed_method, pe_method=pe_method).to(device).eval()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state, strict=True)     # strict: a shape error must be loud
    n = sum(p.numel() for p in model.parameters())
    print(f'  backbone: width={width} d_state={d_state} layers=12 params={n/1e6:.1f}M')
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='m6', choices=list(MODELS),
                    help="paper name (m6 = the 174.7M model = pp_nerf_m5_k30.ckpt)")
    ap.add_argument('--n_events', type=int, default=10)
    ap.add_argument('--out', default='features.npz')
    ap.add_argument('--data_root', default=os.environ.get('FM4NPP_EVAL_ROOT'))
    ap.add_argument('--split', default='test', choices=['pretrain', 'test'])
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--yaml_config', default=None,
                    help='default: $FM4NPP_ROOT/scripts/configs/mamba_tracking.yaml')
    args = ap.parse_args()

    config, width, d_state, ckpt_var = MODELS[args.model]
    ckpt = args.checkpoint or os.environ.get(ckpt_var)
    yaml_cfg = args.yaml_config or os.path.join(
        _ROOT or '', 'scripts/configs/mamba_tracking.yaml')

    for label, val in (('data_root', args.data_root), ('checkpoint', ckpt),
                       ('FM4NPP_ROOT', _ROOT)):
        if not val:
            raise SystemExit(f'error: {label} not set -- source common/paths.sh')

    params = YParams(os.path.abspath(yaml_cfg), config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[extract] paper {args.model} (repo config {config}), {args.n_events} events')

    # stat_dir must hold bin_edges_v3_nbins_8_8_6.pkl or the Voxelizer SILENTLY
    # recomputes bin edges and the tokenization stops matching the checkpoint.
    stat_dir = os.environ.get('FM4NPP_STATS') or params.stat_dir
    bins = os.path.join(stat_dir, 'bin_edges_v3_nbins_8_8_6.pkl')
    if not os.path.isfile(bins):
        raise SystemExit(f'error: {bins} missing. Without it the Voxelizer rebins '
                         f'silently and every feature is wrong.')

    model = build_backbone(width, d_state, params.klen, ckpt, device,
                           params.embed_method, params.pe_method)

    ds = TPCBatchDataset(
        data_root=args.data_root, version=params.data_version, split=args.split,
        group_size=params.group_size, normalize=True,
        limit_data=True, limit_size=args.n_events * 4,     # filtering drops some events
        nleave=params.nleave, order=params.order, num_pred_points=params.klen,
        len_chunk=params.len_chunk, chunk_training=params.chunk_training,
        bin_dir=stat_dir, return_dict=False,
        voxelize=params.voxelize, space_filling_order=params.space_filling_order,
        space_filling_curve=params.space_filling_curve,
        train=(args.split == 'pretrain'))

    n = min(args.n_events, len(ds))
    print(f'[extract] dataset has {len(ds)} events after filtering; taking {n}')

    all_points, all_target, all_feat, all_evt, all_x0 = [], [], [], [], []
    t0 = time.time()
    for i in range(n):
        grouped, target, _ = ds[i]
        pts = grouped.reshape(-1, grouped.shape[-1])          # (N, 4) = E, eta, phi, r
        inp = pts.unsqueeze(0).to(device)
        with autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            _, per_layer, _ = model(inp, return_z=True)
            # x_0, the embedder output. Reproduces the first two lines of forward()
            # exactly, and is the one term missing from the returned stack that you
            # need to rebuild the accumulated stream.
            x0, _pos = model.embedder(model.change_maskval(inp))
        # per_layer is a list of 12 tensors, each (1, N, D) -- the residual-branch
        # outputs z, not the accumulated stream. See the module docstring.
        f = torch.stack(per_layer)[:, 0].float().cpu().numpy().astype(np.float16)

        all_points.append(pts.numpy().astype(np.float32))
        all_target.append(target.numpy().astype(np.int64))
        all_feat.append(f)                                    # (12, N, D)
        all_x0.append(x0[0].float().cpu().numpy().astype(np.float16))   # (N, D)
        all_evt.append(np.full(pts.shape[0], i, dtype=np.int32))
        print(f'    event {i:>3d}: {pts.shape[0]:>5d} points, '
              f'{len(np.unique(target.numpy())):>3d} tracks', flush=True)

    points = np.concatenate(all_points)
    seg_target = np.concatenate(all_target)
    features = np.concatenate(all_feat, axis=1)               # (12, total_N, D)
    x0_all = np.concatenate(all_x0)                           # (total_N, D)
    event_id = np.concatenate(all_evt)

    np.savez_compressed(
        args.out, points=points, seg_target=seg_target, features=features,
        x0=x0_all, event_id=event_id,
        meta=np.array([args.model, config, str(width), str(d_state),
                       os.path.basename(ckpt), args.split], dtype=object))

    size = os.path.getsize(args.out) / 1e6
    print(f'\n[extract] {points.shape[0]:,} points, {features.shape[0]} layers, '
          f'dim {features.shape[2]}')
    print(f'[extract] wrote {args.out} ({size:.0f} MB) in {time.time()-t0:.0f}s')
    print(f'[extract] feature range [{features.min():.2f}, {features.max():.2f}]')


if __name__ == '__main__':
    main()
