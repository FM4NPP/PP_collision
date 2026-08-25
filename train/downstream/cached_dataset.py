"""Dataset that serves precomputed frozen-backbone features.

Pairs with `scripts/cache_features.py`. When the foundation model is frozen — which is the
case for every downstream task in this repository — its output for a given event never
changes, so recomputing it each epoch is wasted work. This reads the cache instead.

The cached loader yields the same 3-tuple arity as the live one, `(points, target, X)`,
where the third slot carries per-layer features rather than k-nearest neighbours. The
trainer selects between them on `self.feature_cache`; nothing else changes.

Padding: features for padded positions are zero-filled. Those positions are excluded by
`padding_mask` in the attention and by `mask` in the loss, so they cannot affect the result.
The live path leaves backbone-computed values there instead — one more reason cached and
live runs agree closely but not bitwise.
"""
import json
import os

import torch
import torch.nn.functional as F
from mmap_ninja import RaggedMmap
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Serves (points, seg_target, features) triples from a cache directory.

    Args:
        cache_dir: directory written by scripts/cache_features.py
        limit_size: optionally use only the first N events
    """

    def __init__(self, cache_dir, limit_size=None):
        self.cache_dir = cache_dir
        meta_path = os.path.join(cache_dir, 'cache_meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f'{meta_path} not found — is {cache_dir} a feature cache?')
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.points = RaggedMmap(os.path.join(cache_dir, 'points'))
        self.seg_target = RaggedMmap(os.path.join(cache_dir, 'seg_target'))
        self.features = RaggedMmap(os.path.join(cache_dir, 'features'))

        n = len(self.points)
        if len(self.seg_target) != n or len(self.features) != n:
            raise ValueError(
                f'cache is inconsistent: points={n} '
                f'seg_target={len(self.seg_target)} features={len(self.features)}')
        self.n = n if limit_size is None else min(n, int(limit_size))

        print(f'[cache] {cache_dir}: {self.n}/{n} events, '
              f'width={self.meta["embed_dim"]} layers={self.meta["num_layers"]} '
              f'ckpt={self.meta["checkpoint"]}')

    @property
    def embed_dim(self):
        return int(self.meta['embed_dim'])

    @property
    def num_layers(self):
        return int(self.meta['num_layers'])

    @property
    def combine_layers(self):
        """True if the cache stores the layer-combined tensor rather than the stack."""
        return bool(self.meta.get('combine_layers', False))

    @property
    def layer_weights(self):
        """Softmax weights baked into a combined cache, or None."""
        return self.meta.get('layer_weights')

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        pts = torch.from_numpy(self.points[i].copy())                  # (N, 4) float32
        tgt = torch.from_numpy(self.seg_target[i].copy())              # (N,)   int64
        # Stay in float16. Inflating to float32 here costs 2x host memory, and on unified
        # -memory parts (GB10) host and device draw on the same pool -- a batch of 32 at
        # width 1536 is ~7.5 GB in fp32 versus ~3.8 GB in fp16.
        fea = torch.from_numpy(self.features[i].copy())                # (L, N, D) float16
        return pts, tgt, fea


class CachedCollator:
    """Pads a batch of cached events.

    Mirrors MyCollator's padding of points and targets (pad_val -100, which the model turns
    into 0 via change_maskval and which the loss masks on), and stacks features into the
    (L, B, N, D) layout the head expects from `torch.stack(pre_embed)`.
    """

    def __init__(self, pad_val=-100):
        self.pad_val = pad_val

    def __call__(self, batch):
        longest = max(p.size(0) for p, _, _ in batch)
        pts, tgts, feas = [], [], []
        for p, t, f in batch:
            n = p.size(0)
            pts.append(F.pad(p, (0, 0, 0, longest - n), value=self.pad_val))
            tgts.append(F.pad(t, (0, longest - n), value=self.pad_val))
            # (L, N, D) -> pad along N only; zero, since these positions are masked out
            feas.append(F.pad(f.float(), (0, 0, 0, longest - n), value=0.0).half())
        points = torch.stack(pts)                     # (B, N, 4)
        targets = torch.stack(tgts)                   # (B, N)
        features = torch.stack(feas).permute(1, 0, 2, 3).contiguous()  # (L, B, N, D)
        return points, targets, features


def get_cached_data_loader(cache_dir, batch_size, shuffle, num_workers=0,
                           limit_size=None, drop_last=False):
    ds = CachedFeatureDataset(cache_dir, limit_size=limit_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(batch_size), shuffle=shuffle,
        num_workers=num_workers, collate_fn=CachedCollator(),
        drop_last=drop_last, pin_memory=False)  # feature batches are large; pinning
                                                # doubles the footprint for no gain here
    return loader, ds
