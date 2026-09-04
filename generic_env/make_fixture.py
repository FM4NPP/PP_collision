#!/usr/bin/env python3
"""Build the committed 20-event fixture, and record where it came from.

This runs once, by us, against a full converted dataset. It is committed so the fixture is
auditable rather than a blob of unexplained bytes -- but you do not need to run it, and on a
fresh clone you cannot: it needs a full RaggedMmap root as input.

WHY THE FIXTURE IS SHIPPED RATHER THAN FETCHED

`scripts/fetch_labeled_data.py` already avoids the 118.5 GB archive by pulling only the
labeled splits (~1 GB) over HTTP range requests. But it has no granularity below one member,
and the smallest useful member is `labeled/test/spacepoints.npz` at 78 MB. There is no way to
ask Zenodo for twenty events. So the fixture is derived once from data already converted and
committed at ~1 MB.

THE LAYOUT IS NOT ARBITRARY

`get_data_loader()` builds the training set from `data_root` with `split='pretrain'` and the
evaluation set from `data_root_test` with `split='test'`. The split is a SUFFIX on each array
directory, not a parent directory, so one root holds both:

    fixture/
        features_pretrain/  seg_target_pretrain/  pid_target_pretrain/  reg_target_pretrain/
        features_test/      seg_target_test/      pid_target_test/      reg_target_test/

`pretrain` here does not mean the unlabeled pretraining corpus -- it is the loader's name for
"the training split". Getting this wrong produces a dataset that loads and trains on nothing.

Usage:
    python make_fixture.py --src ~/fm4npp_repro/data/mmap70/val100 --n 20
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from mmap_ninja import RaggedMmap

HERE = os.path.dirname(os.path.abspath(__file__))
ARRAYS = ('features', 'seg_target', 'pid_target', 'reg_target')

# Thresholds from fm4npp/datasets/dataset.py filter_data(). An event above either of these is
# dropped at load, which on a 20-event fixture would be a silent 5% data loss.
MAX_POINTS = 3200
MAX_TRACKS = 150


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='a converted RaggedMmap root whose arrays carry the _test suffix')
    ap.add_argument('--src_suffix', default='test')
    ap.add_argument('--n', type=int, default=20, help='events for the training split')
    ap.add_argument('--n_eval', type=int, default=None,
                    help='events for the eval split (default: same as --n, offset after it)')
    ap.add_argument('--out', default=os.path.join(HERE, 'fixture'))
    args = ap.parse_args()
    n_eval = args.n_eval if args.n_eval is not None else args.n

    src = {a: RaggedMmap(os.path.join(args.src, f'{a}_{args.src_suffix}')) for a in ARRAYS}
    total = len(src['features'])
    need = args.n + n_eval
    if total < need:
        sys.exit(f'{args.src} holds {total} events; need {need}')

    # Reject anything filter_data() would drop, so the fixture's event count is what it says.
    kept = []
    for i in range(total):
        pts = src['features'][i]
        tracks = len(np.unique(src['seg_target'][i]))
        if pts.shape[0] <= MAX_POINTS and tracks <= MAX_TRACKS:
            kept.append(i)
        if len(kept) == need:
            break
    if len(kept) < need:
        sys.exit(f'only {len(kept)} of {total} events survive filtering; need {need}')

    split_idx = {'pretrain': kept[:args.n], 'test': kept[args.n:need]}
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    manifest = {'source': os.path.abspath(args.src), 'arrays': {}, 'splits': {}}
    for split, idxs in split_idx.items():
        npts = int(sum(src['features'][i].shape[0] for i in idxs))
        ntrk = [int(len(np.unique(src['seg_target'][i]))) for i in idxs]
        manifest['splits'][split] = dict(
            events=len(idxs), source_indices=idxs, points=npts,
            max_points=int(max(src['features'][i].shape[0] for i in idxs)),
            max_tracks=int(max(ntrk)))
        for a in ARRAYS:
            d = os.path.join(args.out, f'{a}_{split}')
            RaggedMmap.from_generator(out_dir=d,
                                      sample_generator=(src[a][i] for i in idxs),
                                      batch_size=len(idxs), verbose=False)
            manifest['arrays'][f'{a}_{split}'] = dict(
                events=len(RaggedMmap(d)),
                data_sha256=sha256_file(os.path.join(d, 'data.ninja')))
        print(f'  {split:9s} {len(idxs):3d} events  {npts:7,d} points  '
              f'max {manifest["splits"][split]["max_points"]} pts / '
              f'{manifest["splits"][split]["max_tracks"]} tracks')

    # The v3 bin edges travel WITH the data. If this pickle is absent, voxelizer.py does not
    # fail -- it recomputes bin edges from whatever it is handed and writes a new file, giving
    # a different tokenization from the one the checkpoints were pretrained with, silently.
    # Three pickles are read from stat_dir on this path: the v3 bin edges by the voxelizer
    # (every dataset construction), and the two loss pickles by train() (unconditionally,
    # just before the epoch loop). Since the configs set stat_dir to the fixture directory,
    # all three travel with the data.
    manifest['stats'] = {}
    for pkl in ('bin_edges_v3_nbins_8_8_6.pkl', 'loss_bin_pp.pkl', 'loss_weight_pp.pkl'):
        shutil.copy(os.path.join(os.path.dirname(HERE), 'stats', pkl),
                    os.path.join(args.out, pkl))
        manifest['stats'][pkl] = sha256_file(os.path.join(args.out, pkl))

    with open(os.path.join(args.out, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(args.out) for f in fs)
    print(f'\n  wrote {args.out}  ({size/1e6:.2f} MB apparent)')


if __name__ == '__main__':
    main()
