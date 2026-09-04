#!/usr/bin/env python3
"""Pooled (test-set-wide) ARI from the per-point CSV that `inference()` writes.

The trainer reports ARI as a *mean over events*: it scores each event separately and
averages. The paper reports a single ARI "computed over the entire test set". Those are
different numbers, and the difference is not a detail -- pooling makes the task harder,
because a global clustering must also keep tracks from different events apart.

To pool, track ids have to be made globally unique first. Event 3's track 1 and event 8's
track 1 are unrelated objects; if they keep the same label, a pooled score credits the
model for "clustering" them together. Offsetting per event by a running maximum fixes it.

Noise points (`seg_target == 0`) are kept, matching what the per-event metric does -- the
trainer passes the raw label vector to adjusted_rand_score without filtering.

Usage:
    python pooled_ari.py per_point.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def offset_by_event(df, col):
    """Relabel `col` so ids are unique across events, preserving within-event grouping."""
    out = np.empty(len(df), dtype=np.int64)
    base = 0
    for _, idx in sorted(df.groupby('batch_idx', sort=True).indices.items()):
        v = df[col].values[idx]
        # factorize within the event to a contiguous 0..k-1, then shift past all ids used
        _, local = np.unique(v, return_inverse=True)
        out[idx] = local + base
        base += int(local.max()) + 1
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'per_point.csv'
    df = pd.read_csv(path, usecols=['batch_idx', 'seg_target', 'pred_assignment'])
    print(f'{len(df):,} points over {df["batch_idx"].nunique():,} events')

    per_ev = np.asarray([
        adjusted_rand_score(g['seg_target'].values, g['pred_assignment'].values)
        for _, g in df.groupby('batch_idx', sort=True)])

    truth = offset_by_event(df, 'seg_target')
    pred = offset_by_event(df, 'pred_assignment')
    pooled = adjusted_rand_score(truth, pred)

    print(f'  per-event mean ARI : {per_ev.mean():.4f}  '
          f'(median {np.median(per_ev):.4f}, min {per_ev.min():.4f}, '
          f'max {per_ev.max():.4f}, sd {per_ev.std():.4f})')
    print(f'  pooled ARI         : {pooled:.4f}')
    print(f'  distinct clusters  : truth {truth.max()+1:,}  pred {pred.max()+1:,}')


if __name__ == '__main__':
    main()
