#!/usr/bin/env python3
"""Precompute and cache frozen-backbone features for downstream adapter training.

The downstream task freezes the foundation model (`torch.no_grad()`, `dropout: 0.0`) and
trains only a small adapter head. That means the backbone recomputes *identical* features
for the same events on every epoch. Measured on a GB10 at batch 32, sequence length ~2000:

    frozen backbone forward   786 ms/batch   ~45% of step time
    dataloading (workers=0)   379 ms/batch   ~22%
    matcher + loss (x3)       261 ms/batch   ~15%
    head fwd/bwd              140 ms/batch    ~8%

Caching the backbone output removes the first item outright, and caching the preprocessed
points removes the second, because the expensive CPU work (polar transform, KNN,
voxelisation) is done once instead of every epoch.

For large backbones this is the difference between feasible and not: the 188M-parameter
model costs one expensive pass instead of one per epoch.

WHAT IS STORED, PER EVENT

    points      (N, 4)      float32   preprocessed/serialised (E, eta, phi, r)
    seg_target  (N,)        int64     track ids
    features    (12, N, D)  float16   per-layer backbone hidden states

All twelve layers are required: the adapter learns a softmax over them
(`MambaAttentionHead.weighted_avg_weights`), so they cannot be collapsed in advance.

float16 is safe and is not a precision loss here — observed feature range on real events is
about [-4.7, 4.0], and float16 carries 10 mantissa bits against the 8 that the bf16 forward
pass itself computes with.

SIZE

    bytes/point = 12 * D * 2.  At 60.29M points (the full 70k labelled train split):
        D=256  (repo m1 / paper m3)    370 GB
        D=1536 (repo m5 / paper m6)    2.2 TB

NUMERICAL EQUIVALENCE -- READ THIS BEFORE COMPARING RUNS

Caching is *causally* exact: Mamba is causal, so trailing padding cannot influence real
tokens, and a per-event forward reproduces the batched-with-padding forward over the real
tokens. Verified directly.

It is *not* bitwise identical. Different tensor shapes select different GEMM tilings, so
bf16 reduction order differs and results move by about one ULP (~3e-3 relative). Training
from cache will therefore track, but not exactly reproduce, an on-the-fly run at the same
seed. Compare the two the way you would compare two seeds, not to eight decimals.

The cache is only valid while the backbone is frozen. It is invalidated by a different
checkpoint, different preprocessing, or any config change feeding either. It is
fundamentally incompatible with LoRA or backbone fine-tuning.

USAGE

    python scripts/cache_features.py \
        --yaml_config scripts/configs/mamba_tracking_full0.yaml \
        --config d9_m1_k30_p20 \
        --checkpoint /path/to/pp_nerf_m1_k30.ckpt \
        --out /path/to/cache --eventnumber 10000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'train', 'downstream'))

from mmap_ninja import RaggedMmap  # noqa: E402

from fm4npp.utils import YParams  # noqa: E402
from fm4npp.datasets.dataset import TPCBatchDataset  # noqa: E402
from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT  # noqa: E402


def build_backbone(params, checkpoint, device):
    cls = Mamba1GPT if getattr(params, 'mambaversion', 'mamba2') == 'mamba1' else MambaGPT
    model = cls(embed_dim=params.embed_dim,
                num_layers=params.num_layers_backbone,
                d_state=params.d_state, d_conv=4, expand=2,
                klen=params.klen, dropout=0.0,
                embed_method=params.embed_method,
                pe_method=params.pe_method).to(device).eval()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ck['model_state'].items()}
    missing, unexpected = model.load_state_dict(state, strict=True), None
    n = sum(p.numel() for p in model.parameters())
    print(f'  backbone: {cls.__name__} width={params.embed_dim} '
          f'd_state={params.d_state} layers={params.num_layers_backbone} '
          f'params={n/1e6:.1f}M', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml_config', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, help='cache directory to create')
    ap.add_argument('--eventnumber', type=int, default=10000)
    ap.add_argument('--split', default='pretrain', choices=['pretrain', 'test'])
    ap.add_argument('--data_root', default=None, help='override data_root from the config')
    ap.add_argument('--batch_size', type=int, default=1,
                    help='events per backbone forward; 1 keeps the cache padding-free')
    ap.add_argument('--write_every', type=int, default=500)
    args = ap.parse_args()

    params = YParams(args.yaml_config, args.config)
    params.limit_data = True
    params.limit_size = args.eventnumber
    root = args.data_root or (params.data_root if args.split == 'pretrain'
                              else params.data_root_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[cache] config={args.config} split={args.split} events={args.eventnumber}')
    model = build_backbone(params, args.checkpoint, device)

    ds = TPCBatchDataset(
        data_root=root, version=params.data_version, split=args.split,
        group_size=params.group_size, normalize=True,
        limit_data=True, limit_size=args.eventnumber,
        nleave=params.nleave, order=params.order, num_pred_points=params.klen,
        len_chunk=params.len_chunk, chunk_training=params.chunk_training,
        bin_dir=params.stat_dir, return_dict=False,
        voxelize=params.voxelize, space_filling_order=params.space_filling_order,
        space_filling_curve=params.space_filling_curve,
        train=(args.split == 'pretrain'))
    n_events = len(ds)
    print(f'[cache] {n_events} events after filtering')

    os.makedirs(args.out, exist_ok=True)

    def gen(kind):
        """Yield one array per event. Runs the backbone once per event for 'features'."""
        for i in range(n_events):
            grouped, target, _ = ds[i]
            pts = grouped.reshape(-1, grouped.shape[-1])
            if kind == 'points':
                yield pts.numpy().astype(np.float32)
            elif kind == 'seg_target':
                yield target.numpy().astype(np.int64)
            else:
                with autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                    _, per_layer, _ = model(pts.unsqueeze(0).to(device), return_z=True)
                # (L, 1, N, D) -> (L, N, D)
                yield torch.stack(per_layer)[:, 0].float().cpu().numpy().astype(np.float16)

    t0 = time.time()
    for kind in ('points', 'seg_target', 'features'):
        d = os.path.join(args.out, kind)
        t1 = time.time()
        RaggedMmap.from_generator(out_dir=d, sample_generator=gen(kind),
                                  batch_size=args.write_every, verbose=False)
        print(f'  wrote {kind:11s} in {time.time()-t1:6.0f}s', flush=True)

    meta = dict(config=args.config, split=args.split, n_events=n_events,
                embed_dim=int(params.embed_dim), num_layers=int(params.num_layers_backbone),
                d_state=int(params.d_state), klen=int(params.klen),
                checkpoint=os.path.basename(args.checkpoint),
                feature_dtype='float16', data_root=root)
    with open(os.path.join(args.out, 'cache_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(args.out) for f in fs)
    print(f'[cache] done in {time.time()-t0:.0f}s, {size/1e9:.1f} GB -> {args.out}')


if __name__ == '__main__':
    main()
