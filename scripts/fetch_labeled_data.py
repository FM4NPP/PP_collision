#!/usr/bin/env python3
"""Download only the labeled splits of TPCpp-10M from Zenodo.

The release is published as one 118.5 GB zip, but the downstream tasks only need the
labeled splits, which total under 1 GB:

    labeled/train/       ~820 MB   70k events (sharded across 7 files)
    labeled/validation/  ~152 MB   13k events
    labeled/test/         ~82 MB    7k events

Zenodo serves HTTP range requests, so we can read the zip's central directory and pull
just those members instead of the whole archive -- about 120x less data. The unlabeled
pretraining corpus (100 x ~1.2 GB) is only needed if you intend to pretrain from
scratch; skip it to reproduce the downstream results.

    pip install remotezip
    python scripts/fetch_labeled_data.py --out ./TPCpp-10M

Then convert to the memory-mapped layout the training code reads:

    python scripts/prepare_data.py --in_dir ./TPCpp-10M/labeled/train \
                                   --out ./mmap_train --split pretrain
    python scripts/prepare_data.py --in_dir ./TPCpp-10M/labeled/test \
                                   --out ./mmap_test  --split test
"""
import argparse
import os
import time

from remotezip import RemoteZip

URL = "https://zenodo.org/records/16970029/files/TPCpp-10M.zip?download=1"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='./TPCpp-10M', help='destination directory')
    ap.add_argument('--splits', default='train,validation,test',
                    help='comma-separated labeled splits to fetch')
    ap.add_argument('--url', default=URL)
    args = ap.parse_args()

    wanted = tuple(f'labeled/{s.strip()}/' for s in args.splits.split(',') if s.strip())
    os.makedirs(args.out, exist_ok=True)

    with RemoteZip(args.url) as z:
        targets = sorted(n for n in z.namelist()
                         if n.startswith(wanted) and not n.endswith('/'))
        if not targets:
            raise SystemExit(f'no members matched {wanted}')
        total = sum(z.getinfo(n).file_size for n in targets)
        print(f'fetching {len(targets)} members, {total/1e9:.2f} GB '
              f'(the full archive is 118.53 GB)')
        done = 0
        for n in targets:
            info = z.getinfo(n)
            dest = os.path.join(args.out, n)
            if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                done += info.file_size
                print(f'  have {n}')
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            t0 = time.time()
            z.extract(n, path=args.out)
            done += info.file_size
            print(f'  {n}  {info.file_size/1e6:.1f} MB in {time.time()-t0:.0f}s '
                  f'[{done/total*100:.0f}%]')
    print('done')


if __name__ == '__main__':
    main()
