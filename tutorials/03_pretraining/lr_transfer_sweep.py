#!/usr/bin/env python3
"""The μP payoff experiment: does the optimal learning rate move as the model gets wider?

    INPUT   $FM4NPP_PRETRAIN_ROOT
    OUTPUT  lr_transfer.png + lr_transfer.csv
    CLAIM   with μP the loss-vs-LR minimum stays put across widths; without it, it drifts

This is the experiment that justifies μP existing at all. For each width we train the same
small number of steps at a range of learning rates and record the final training loss. Then
we plot loss vs LR, one curve per width.

    μP ON   -> the curves' minima line up. Tune the LR once at width 128, use it at 1536.
    μP OFF  -> the minima shift with width. Every new size needs its own sweep.

Cost: widths x LRs x 2 arms short runs. The default grid (3 widths, 5 LRs, both arms,
150 steps) is 30 runs, about 25 minutes on one A100 and ~1 h on a smaller card.

    python lr_transfer_sweep.py --widths 128,256,512 --steps 150
    python lr_transfer_sweep.py --widths 128,256 --lrs 1e-5,1e-4,1e-3 --steps 60   # quick

A caveat worth stating up front: FM4NPP ties d_state to width (Nx = Nu/16), while textbook
μP holds the state dimension fixed. This sweep therefore tests the transfer property of
*this* parameterization along *this* ladder -- which is the practically useful question --
not the textbook claim.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '02_mu_parameterization'))
_ROOT = os.environ.get('FM4NPP_ROOT')
if _ROOT:
    sys.path.insert(0, _ROOT)

from fm4npp.models.mambagpt import MambaGPT              # noqa: E402
from fm4npp.datasets.dataset_pretrain import get_data_loader   # noqa: E402
from fm4npp.utils import YParams                         # noqa: E402
import mup                                               # noqa: E402


def run_one(loader, width, d_state, lr, steps, use_mup, klen, device):
    """Train from scratch for `steps` and return the mean loss over the last 20%."""
    torch.manual_seed(0)
    model = MambaGPT(embed_dim=width, num_layers=12, d_state=d_state, d_conv=4,
                     expand=2, klen=klen, dropout=0.0,
                     embed_method='add', pe_method='nerf')
    mup.apply_mup_init(model, Nu=width, Nx=d_state, enabled=use_mup, verbose=False)
    model = model.to(device).train()

    opt = mup.build_mup_optimizer(model, Nu=width, Nx=d_state, base_lr=lr,
                                  enabled=use_mup, verbose=False)
    loss_func = nn.MSELoss(reduction='none')

    losses, it = [], 0
    while it < steps:
        for grouped, _, knearest in loader:
            b, c = grouped.size(0), grouped.size(-1)
            klabel = knearest.reshape(b, -1, klen * 3).to(device)
            x = grouped.reshape(b, -1, c).to(device)

            model.zero_grad()
            pred = model(x)
            kmask = klabel != -100
            loss = (loss_func(pred, klabel) * kmask).sum() / kmask.sum()
            if not torch.isfinite(loss):
                return float('nan')          # diverged; that is a real data point
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            opt.step()

            losses.append(loss.item())
            it += 1
            if it >= steps:
                break

    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return float(np.mean(losses[-max(1, steps // 5):]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yaml_config', default=None,
                    help='rendered tutorial config (default: render tutorial_m3.yaml)')
    ap.add_argument('--config', default='tutorial_m3')
    ap.add_argument('--widths', default='128,256,512')
    ap.add_argument('--lrs', default='1e-5,3e-5,1e-4,3e-4,1e-3')
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--out', default='lr_transfer')
    args = ap.parse_args()

    cfg = args.yaml_config
    if cfg is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'tutorial_m3.rendered.yaml')
        if not os.path.isfile(cfg):
            os.system(f'python {here}/configs/render_config.py '
                      f'{here}/configs/tutorial_m3.yaml -o {cfg}')

    params = YParams(os.path.abspath(cfg), args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    widths = [int(w) for w in args.widths.split(',')]
    lrs = [float(x) for x in args.lrs.split(',')]

    # One loader, reused for every run, so data order is not a confound.
    loader, _, _, _ = get_data_loader(params, False)

    print(f'device={device}  widths={widths}  lrs={lrs}  steps={args.steps}')
    print(f'{len(widths) * len(lrs) * 2} runs total\n')

    rows = []
    t0 = time.time()
    for use_mup in (True, False):
        for w in widths:
            d_state = max(w // 16, 1)          # the ladder's rule: Nx = Nu/16
            for lr in lrs:
                t = time.time()
                final = run_one(loader, w, d_state, lr, args.steps, use_mup,
                                params.klen, device)
                rows.append(dict(mup=use_mup, width=w, d_state=d_state, lr=lr,
                                 loss=final))
                print(f'  mup={"ON " if use_mup else "OFF"} width={w:<5d} '
                      f'lr={lr:<9.1e} loss={final:>9.5f}  ({time.time()-t:.0f}s)',
                      flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f'{args.out}.csv', index=False)
    print(f'\nwrote {args.out}.csv  ({time.time()-t0:.0f}s total)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, on in zip(axes, (True, False)):
        sub = df[df.mup == on]
        for w in widths:
            s = sub[sub.width == w].sort_values('lr')
            ax.plot(s.lr, s.loss, 'o-', lw=1.8, ms=6, label=f'width {w}')
            fin = s.dropna(subset=['loss'])
            if len(fin):
                best = fin.loc[fin.loss.idxmin()]
                ax.axvline(best.lr, ls=':', lw=1, alpha=.55,
                           color=ax.lines[-1].get_color())
        ax.set_xscale('log')
        ax.set_xlabel('learning rate')
        ax.set_title(f'μP {"ON — minima should align" if on else "OFF — minima drift"}')
        ax.grid(alpha=.3, which='both')
        ax.legend()
    axes[0].set_ylabel(f'training loss (mean of last 20% of {args.steps} steps)')
    plt.tight_layout()
    plt.savefig(f'{args.out}.png', dpi=140)
    print(f'wrote {args.out}.png')

    print('\noptimal LR per width:')
    for on in (True, False):
        sub = df[(df.mup == on)].dropna(subset=['loss'])
        best = {int(w): float(sub[sub.width == w].sort_values('loss').lr.iloc[0])
                for w in widths if len(sub[sub.width == w])}
        spread = (max(best.values()) / min(best.values())) if len(best) > 1 else float('nan')
        print(f'  μP {"ON " if on else "OFF"}: {best}   spread = {spread:.1f}x')
    print('\nA smaller spread with μP ON is the transfer property working.')


if __name__ == '__main__':
    main()
