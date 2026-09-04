#!/usr/bin/env python3
"""Plot the training and validation loss from a run's CSV.

    INPUT   $FM4NPP_RUNS/<config>/<run>/config_<config>_run_<run>.csv
    OUTPUT  loss_curve.png

    python plot_loss.py $FM4NPP_RUNS/tutorial_m3/debug0/config_tutorial_m3_run_debug0.csv
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import pandas as pd                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('-o', '--out', default='loss_curve.png')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tr = df[df.split == 'train']
    va = df[df.split == 'val']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(tr.step, tr.loss, lw=1, alpha=.5, color='steelblue', label='train')
    if len(tr) > 20:                       # a rolling mean, since per-step loss is noisy
        axes[0].plot(tr.step, tr.loss.rolling(20, min_periods=1).mean(),
                     lw=2, color='navy', label='train (20-step mean)')
    if len(va):
        axes[0].plot(va.step, va.loss, 'o-', color='crimson', ms=5, label='val')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('MSE loss')
    axes[0].set_title(args.title or os.path.basename(os.path.dirname(args.csv)))
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(tr.step, tr.lr, lw=1.8, color='darkorange')
    axes[1].set_xlabel('step'); axes[1].set_ylabel("lr ('other' group)")
    axes[1].set_title('learning rate schedule')
    axes[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f'wrote {args.out}')

    if len(tr):
        print(f'  steps      : {int(tr.step.max())}')
        print(f'  train loss : {tr.loss.iloc[0]:.4f} -> {tr.loss.iloc[-1]:.4f}')
    if len(va):
        print(f'  val loss   : {va.loss.iloc[0]:.4f} -> {va.loss.iloc[-1]:.4f} '
              f'(best {va.loss.min():.4f})')


if __name__ == '__main__':
    main()
