#!/usr/bin/env python3
"""Turn a per-point CSV into the numbers and plots worth looking at.

    python score_report.py per_point.csv --out figures/

Prints the four-way grid -- {option 1, option 2} x {per-event, pooled} -- so the two
choices that decide comparability with the paper are visible side by side rather than
buried in a single scalar. Then plots the per-event ARI distribution and the predicted
versus true cluster counts, which is where over-segmentation shows up as structure rather
than as a number.

The per-event column here reproduces the trainer's own Avg_ARI. That agreement is the
reason to trust the pooled column printed next to it.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def offset_per_event(events, labels):
    """Relabel so ids are unique across events, preserving within-event grouping.

    Without this, event 3's "track 1" and event 8's "track 1" are scored as one cluster,
    which silently credits the model for grouping points that have nothing to do with
    each other.
    """
    out = np.empty(len(labels), dtype=np.int64)
    base = 0
    for e in np.unique(events):
        m = events == e
        _, local = np.unique(labels[m], return_inverse=True)
        out[m] = local + base
        base += int(local.max()) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--out', default=None, help='directory for figures; omit to skip plots')
    args = ap.parse_args()

    cols = ['batch_idx', 'seg_target', 'pred_assignment']
    have_o1 = 'pred_assignment_o1' in pd.read_csv(args.csv, nrows=0).columns
    if have_o1:
        cols.append('pred_assignment_o1')
    df = pd.read_csv(args.csv, usecols=cols)

    ev = df['batch_idx'].values
    truth = df['seg_target'].values
    n_ev = df['batch_idx'].nunique()
    print(f'{len(df):,} points over {n_ev:,} events\n')

    T = offset_per_event(ev, truth)
    print(f'  {"decoding":26s} {"per-event":>10s} {"pooled":>9s} {"clusters":>10s}')
    per_event_o2 = None
    variants = [('option=2 (mask x class)', 'pred_assignment')]
    if have_o1:
        variants.insert(0, ('option=1 (mask prob)', 'pred_assignment_o1'))
    for name, col in variants:
        pred = df[col].values
        per_ev = np.asarray([adjusted_rand_score(truth[ev == e], pred[ev == e])
                             for e in np.unique(ev)])
        P = offset_per_event(ev, pred)
        print(f'  {name:26s} {per_ev.mean():10.4f} '
              f'{adjusted_rand_score(T, P):9.4f} {P.max() + 1:10,d}')
        if col == 'pred_assignment':
            per_event_o2 = per_ev
    print(f'  {"ground truth":26s} {"":10s} {"":9s} {T.max() + 1:10,d}')

    print(f'\n  per-event spread: median {np.median(per_event_o2):.4f}, '
          f'min {per_event_o2.min():.4f}, max {per_event_o2.max():.4f}, '
          f'sd {per_event_o2.std():.4f}')
    print('  quote the per-event mean against the paper; see this module\'s README.')
    if n_ev < 100:
        print(f'\n  NOTE: {n_ev} events is a small sample. Useful as a setup check '
              '(a broken\n  environment scores ~0.05, not ~0.9), not as a measurement.')

    if not args.out:
        return 0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(per_event_o2, bins=min(40, max(5, n_ev // 3)),
               color='#4C6EF5', edgecolor='white')
    ax[0].axvline(per_event_o2.mean(), color='#E8590C', lw=2,
                  label=f'mean {per_event_o2.mean():.4f}')
    ax[0].set_xlabel('ARI (option 2)'); ax[0].set_ylabel('events')
    ax[0].set_title('per-event ARI'); ax[0].legend()

    tc = np.array([len(np.unique(truth[ev == e])) for e in np.unique(ev)])
    pc = np.array([len(np.unique(df['pred_assignment'].values[ev == e]))
                   for e in np.unique(ev)])
    lim = max(tc.max(), pc.max()) * 1.05
    ax[1].scatter(tc, pc, s=14, alpha=.5, color='#4C6EF5')
    ax[1].plot([0, lim], [0, lim], color='#868E96', ls='--', lw=1)
    ax[1].set_xlabel('true clusters'); ax[1].set_ylabel('predicted clusters')
    ax[1].set_title(f'over-segmentation: {100 * (pc.sum() / tc.sum() - 1):+.1f}%')
    fig.tight_layout()
    p = os.path.join(args.out, 'score_report.png')
    fig.savefig(p, dpi=150)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
