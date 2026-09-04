#!/usr/bin/env python3
"""
Convert the published TPCpp-10M Zenodo release into the RaggedMmap layout that
fm4npp.datasets.dataset.TPCBatchDataset actually opens.

    https://doi.org/10.5281/zenodo.16970029

This script does not exist in the public repo, and SETUP.md's "preprocessing
example" cannot be run (it calls an undefined load_raw_events() and names the
output directories features_train/, which the code never opens).

INPUT  (Zenodo)                       OUTPUT (RaggedMmap dirs under --out)
  <in>/spacepoints.npz  data,size       features_<split>/     (n_i, 4)  float32
  <in>/track_ids.npz    data            seg_target_<split>/   (n_i,)    int64
  <in>/pid_labels.npz   data            pid_target_<split>/   (n_i,)    int64
  <in>/noise_tags.npz   data            reg_target_<split>/   (n_i, 8)  float32

Two conventions worth knowing before you touch this:

1. SPLIT NAMES. get_data_loader() hardcodes split='pretrain' for the *training*
   loader and split='test' for the eval loader. So labeled TRAINING data must be
   written with --split pretrain. The name is historical; it is not the
   unlabeled pretraining corpus.

2. pid_target HOLDS RAW PDG CODES, NOT CLASS INDICES. downstream_util.get_pidlabel()
   maps PDG -> class itself (211->1 pion, 321->2 kaon, 2212->3 proton, 11->4
   electron, everything else -> 0). Zenodo ships pid_labels that are ALREADY those
   class indices, so feeding them through unchanged collapses every point to
   class 0. We invert the map here so get_pidlabel() round-trips exactly.

reg_target IS SYNTHESIZED -- READ THIS BEFORE USING IT FOR PHYSICS.
   The real reg_target is (n, 8) = px, py, pz, vtx_x, vtx_y, vtx_z, q, e, and
   downstream_util.get_trackinfo_noiselabel() derives noise_labels, track_info and
   valid_tracks from it. Zenodo publishes none of those columns -- only a boolean
   noise_tag. We therefore synthesize the minimum that makes the derived
   quantities correct where the data supports it:
       * px is set so that pt = |px| straddles the 0.06 noise threshold in the
         direction given by the real noise_tags  ->  noise_labels are FAITHFUL.
       * q = 1 and vtx = 0                        ->  valid_tracks = 1 everywhere.
       * pz, py = 0                               ->  track_info is a PLACEHOLDER.
   Track finding and noise tagging never read track_info (the track-finding
   trainer builds targets from masks/labels only). Any task that regresses track
   kinematics needs the real reg_target from the collaboration.
"""
import argparse
import glob
import os
import numpy as np
from mmap_ninja import RaggedMmap

# Inverse of downstream_util.get_pidlabel()'s PDG -> class mapping.
CLASS_TO_PDG = {0: 0, 1: 211, 2: 321, 3: 2212, 4: 11}

NOISE_PT_THRESHOLD = 0.06   # must match get_trackinfo_noiselabel's default
PT_SIGNAL, PT_NOISE = 1.0, 1e-3


def _shards(in_dir, stem):
    """Zenodo shards labeled/train as <stem>_000.npz ... and ships labeled/test and
    labeled/validation as a single <stem>.npz. Handle both."""
    single = os.path.join(in_dir, f'{stem}.npz')
    if os.path.exists(single):
        return [single]
    found = sorted(glob.glob(os.path.join(in_dir, f'{stem}_[0-9]*.npz')))
    if not found:
        raise FileNotFoundError(f'no {stem}.npz or {stem}_NNN.npz in {in_dir}')
    return found


def _concat(in_dir, stem, key='data'):
    parts = []
    for f in _shards(in_dir, stem):
        with np.load(f) as h:
            parts.append(h[key])
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def load_split(in_dir):
    # sizes must be concatenated in the SAME shard order as the point arrays, or every
    # event boundary after the first shard is wrong.
    sp_parts, size_parts = [], []
    for f in _shards(in_dir, 'spacepoints'):
        with np.load(f) as h:
            sp_parts.append(h['data'])
            size_parts.append(h['size'])
    spacepoints = np.concatenate(sp_parts) if len(sp_parts) > 1 else sp_parts[0]
    sizes = np.concatenate(size_parts) if len(size_parts) > 1 else size_parts[0]
    track_ids = _concat(in_dir, 'track_ids')
    pid_labels = _concat(in_dir, 'pid_labels')
    noise_tags = _concat(in_dir, 'noise_tags')
    n = int(sizes.sum())
    for name, arr in [('track_ids', track_ids), ('pid_labels', pid_labels),
                      ('noise_tags', noise_tags)]:
        if len(arr) != n:
            raise ValueError(f'{name} has {len(arr)} rows, spacepoints has {n}')
    if spacepoints.shape[1] != 4:
        raise ValueError(f'expected 4 columns (E, x, y, z), got {spacepoints.shape[1]}')
    return spacepoints, sizes, track_ids, pid_labels, noise_tags


def build_reg_target(noise_tags_ev):
    """(n, 8) = px, py, pz, vtx_x, vtx_y, vtx_z, q, e -- see module docstring."""
    reg = np.zeros((len(noise_tags_ev), 8), dtype=np.float32)
    reg[:, 0] = np.where(noise_tags_ev, PT_NOISE, PT_SIGNAL)   # px -> pt
    reg[:, 6] = 1.0                                            # q
    return reg


def event_slices(sizes, lo, hi):
    ends = np.cumsum(sizes)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts[lo:hi], ends[lo:hi]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_dir', required=True, help="Zenodo labeled/<split> directory")
    ap.add_argument('--out', required=True, help="output data_root")
    ap.add_argument('--split', required=True, choices=['pretrain', 'test'],
                    help="'pretrain' for the TRAINING loader, 'test' for eval (see docstring)")
    ap.add_argument('--start', type=int, default=0, help="first event index (inclusive)")
    ap.add_argument('--end', type=int, default=None, help="last event index (exclusive)")
    ap.add_argument('--batch_size', type=int, default=1000)
    args = ap.parse_args()

    spacepoints, sizes, track_ids, pid_labels, noise_tags = load_split(args.in_dir)
    end = len(sizes) if args.end is None else args.end
    bounds = event_slices(sizes, args.start, end)
    os.makedirs(args.out, exist_ok=True)

    unknown = set(np.unique(pid_labels).tolist()) - set(CLASS_TO_PDG)
    if unknown:
        raise ValueError(f'pid_labels contains classes with no PDG inverse: {sorted(unknown)}')
    lut = np.zeros(max(CLASS_TO_PDG) + 1, dtype=np.int64)
    for cls, pdg in CLASS_TO_PDG.items():
        lut[cls] = pdg

    print(f'[prepare_data] {len(bounds)} events (index {args.start}..{end}) '
          f'-> {args.out} as split={args.split!r}')

    specs = {
        'features':   lambda s, e: spacepoints[s:e].astype(np.float32),
        'seg_target': lambda s, e: track_ids[s:e].astype(np.int64),
        'pid_target': lambda s, e: lut[pid_labels[s:e]],
        'reg_target': lambda s, e: build_reg_target(noise_tags[s:e]),
    }
    for name, fn in specs.items():
        out_dir = os.path.join(args.out, f'{name}_{args.split}')
        RaggedMmap.from_generator(
            out_dir=out_dir,
            sample_generator=(fn(s, e) for s, e in bounds),
            batch_size=args.batch_size,
            verbose=False,
        )
        print(f'  wrote {out_dir}')

    # No mid_target: Zenodo publishes no mother-particle IDs. TPCBatchDataset opens
    # it in a try/except and runs without it; the weak-decay task cannot.
    print('[prepare_data] done. Note: mid_target not written (absent from Zenodo).')


if __name__ == '__main__':
    main()
