"""μ-parameterization for FM4NPP's Mamba2 backbone — with a switch.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The μP that produced the released FM4NPP checkpoints is **not in the public repository**.
The shipped pretraining script says so in its own docstring ("no μ-transfer"), uses a
single-group AdamW, and initializes width-independently.

What the repo *does* contain is μP-shaped code in the two downstream trainers. This module
is a faithful reconstruction of that rule, lifted from
`train/downstream/track_finding_trainer.py`, made to actually take effect. It is not the
original file, and we do not claim it reproduces the original training exactly.

There is good evidence the original existed and looked like this: the downstream trainer
loads a **four-group** optimizer state from the pretrained checkpoint, while the shipped
pretraining script saves a **one-group** state. PyTorch raises on that mismatch. Something
saved four groups.

THE RULE
--------
With Nu = embed_dim (width) and Nx = d_state, the repo's rule is

    init    std(lin_B) = sqrt(Nx / Nu)              # input-to-state projection
            std(lin_C) = sqrt(1 / (Nu * Nx))        # state-to-output projection

    lr      A_log  :  base_lr * Nu
            lin_B  :  base_lr * Nx / sqrt(Nu)
            lin_C  :  base_lr * sqrt(Nu) / Nx
            rest   :  base_lr

Two honest caveats, both of which a student should know before trusting this:

1. It is *partial* μP. Only `lin_B`, `lin_C`, norms and biases are touched. `in_proj`,
   `out_proj`, `conv1d`, the embedder and the output layer keep PyTorch's default init,
   which is not what μP prescribes.

2. FM4NPP's width ladder scales `d_state` with width (`Nx = Nu/16` exactly, across all
   four released sizes). Textbook μP holds the state/head dimension fixed. Along this
   ladder the B rule reduces to ∝ sqrt(Nu) and the C rule to ∝ 1/sqrt(Nu), so the
   exponents do not mean what the standard derivation says they mean.

WHY YOUR LEARNING RATES MIGHT BE SILENTLY IGNORED
-------------------------------------------------
`CosineAnnealingWarmupRestarts.__init__` calls `init_lr()`, which does:

    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.min_lr
        self.base_lrs.append(self.min_lr)

Every group is flattened to the same scalar. Construct that scheduler after building a
μP optimizer and **all** width-dependent scaling is gone, with no warning. This is what
happens in the shipped downstream trainers. `attach_scheduler()` below fixes it by
restoring the per-group ratios afterwards; `demonstrate_scheduler_nullification()` shows
the bug happening.

USAGE
-----
    from mup import apply_mup_init, build_mup_optimizer, attach_scheduler

    apply_mup_init(model, Nu=params.embed_dim, Nx=params.d_state, enabled=args.mup)
    opt = build_mup_optimizer(model, Nu, Nx, base_lr=params.min_lr, enabled=args.mup)
    sched = attach_scheduler(opt, total_steps=..., max_lr=..., min_lr=..., warmup=...)
"""
import math

import torch
import torch.nn.init as init


# --------------------------------------------------------------------------- init

def apply_mup_init(model, Nu, Nx, enabled=True, verbose=True):
    """Width-dependent initialization, or the standard one when `enabled=False`.

    Mirrors `initialize_mamba2(model, d_state, embed_dim)` in
    train/downstream/track_finding_trainer.py.

    Args:
        model:   the backbone (MambaGPT)
        Nu:      embed_dim (width)
        Nx:      d_state
        enabled: False reproduces the shipped pretraining init — norms to 1, biases to 0,
                 everything else left at PyTorch defaults, no width dependence at all.

    Returns:
        dict of the std values used, for logging/plots.
    """
    stds = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if enabled and 'lin_B' in name:
                std = math.sqrt(Nx / Nu)
                param.normal_(mean=0.0, std=std)
                stds['lin_B'] = std
            elif enabled and 'lin_C' in name:
                std = math.sqrt(1.0 / (Nu * Nx))
                param.normal_(mean=0.0, std=std)
                stds['lin_C'] = std
            elif 'norm.weight' in name:
                init.ones_(param)
            elif 'bias' in name:
                init.zeros_(param)

    if verbose:
        if enabled:
            print(f'[mup] init ON   Nu={Nu} Nx={Nx}  '
                  f'std(lin_B)={stds.get("lin_B", float("nan")):.5f}  '
                  f'std(lin_C)={stds.get("lin_C", float("nan")):.6f}')
        else:
            print(f'[mup] init OFF  Nu={Nu} Nx={Nx}  '
                  f'(PyTorch defaults; no width dependence)')
    return stds


# --------------------------------------------------------------------------- optimizer

def mup_lr_table(Nu, Nx, base_lr):
    """The four learning rates, without building anything. Useful for plots."""
    return {
        'A_log': base_lr * Nu,
        'lin_B': base_lr * Nx / math.sqrt(Nu),
        'lin_C': base_lr * math.sqrt(Nu) / Nx,
        'other': base_lr,
    }


def build_mup_optimizer(model, Nu, Nx, base_lr, enabled=True,
                        weight_decay=0.1, betas=(0.9, 0.95), verbose=True):
    """Four width-scaled parameter groups, or one flat group when `enabled=False`.

    Mirrors the four-group AdamW in track_finding_trainer.py:299-320.
    """
    if not enabled:
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                weight_decay=weight_decay, betas=betas)
        if verbose:
            print(f'[mup] optim OFF  1 group @ lr={base_lr:.3e}')
        return opt

    groups = {'A_log': [], 'lin_B': [], 'lin_C': [], 'other': []}
    for name, p in model.named_parameters():
        if 'A_log' in name:
            groups['A_log'].append(p)
        elif 'lin_B' in name:
            groups['lin_B'].append(p)
        elif 'lin_C' in name:
            groups['lin_C'].append(p)
        else:
            groups['other'].append(p)

    lrs = mup_lr_table(Nu, Nx, base_lr)
    opt = torch.optim.AdamW(
        [{'params': groups[k], 'lr': lrs[k], 'name': k} for k in
         ('A_log', 'lin_B', 'lin_C', 'other')],
        weight_decay=weight_decay, betas=betas)

    if verbose:
        print(f'[mup] optim ON   Nu={Nu} Nx={Nx} base_lr={base_lr:.3e}')
        for k in ('A_log', 'lin_B', 'lin_C', 'other'):
            print(f'         {k:<6s} {len(groups[k]):>4d} tensors  '
                  f'lr={lrs[k]:.3e}  ({lrs[k]/base_lr:>9.4f} x base)')
    return opt


# --------------------------------------------------------------------------- scheduler

def attach_scheduler(optimizer, total_steps, max_lr, min_lr, warmup_steps,
                     preserve_mup=True, verbose=True):
    """Build the cosine scheduler WITHOUT destroying per-group learning rates.

    `CosineAnnealingWarmupRestarts.__init__` calls `init_lr()`, which overwrites every
    group's `lr` with the scalar `min_lr` and sets `base_lrs = [min_lr] * n_groups`.
    Any μP scaling you built is gone at that moment.

    With `preserve_mup=True` we capture each group's LR *before* constructing the
    scheduler, then rewrite `base_lrs` to keep the ratios. The scheduler still drives the
    shape of the schedule; the groups keep their relative scaling.

    Pass `preserve_mup=False` to reproduce the shipped (broken) behaviour exactly.
    """
    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

    before = [g['lr'] for g in optimizer.param_groups]
    # Each group's μP factor, relative to the unscaled base rate. The scheduler will
    # produce ONE scalar schedule; we multiply it back out by these.
    mult = [lr / min_lr for lr in before]

    sched = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=total_steps, cycle_mult=1.0,
        max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=1.0)

    after_init = [g['lr'] for g in optimizer.param_groups]

    if preserve_mup and len(set(mult)) > 1:
        # Leave sched.base_lrs alone -- it must stay [min_lr]*n so that the scalar
        # schedule still runs min_lr -> max_lr -> min_lr. Rewriting it instead makes
        # each group anneal TOWARD max_lr from a scaled start, which inverts the
        # scaling partway through the run.
        def _apply(_o=optimizer, _m=mult):
            scalar = _o.param_groups[0]['lr'] / _m[0] if _o.param_groups[0]['lr'] else 0.0
            for g, m in zip(_o.param_groups, _m):
                g['lr'] = scalar * m

        _orig_step = sched.step

        def step(epoch=None, _s=sched, _o=optimizer, _m=mult):
            # Undo our multipliers so the scheduler sees the scalar it expects,
            # let it advance, then re-apply.
            scalar = _o.param_groups[0]['lr'] / _m[0]
            for g in _o.param_groups:
                g['lr'] = scalar
            _orig_step(epoch)
            base = _o.param_groups[0]['lr']
            for g, m in zip(_o.param_groups, _m):
                g['lr'] = base * m

        sched.step = step
        for g, m in zip(optimizer.param_groups, mult):
            g['lr'] = min_lr * m

    if verbose:
        print(f'[mup] scheduler: before={[f"{x:.3e}" for x in before]}')
        print(f'[mup]            after init_lr={[f"{x:.3e}" for x in after_init]}'
              f'  <-- what the shipped code lives with')
        if preserve_mup and len(set(mult)) > 1:
            now = [g["lr"] for g in optimizer.param_groups]
            print(f'[mup]            restored={[f"{x:.3e}" for x in now]}')
    return sched


# --------------------------------------------------------------------------- demo

def demonstrate_scheduler_nullification(Nu=1536, Nx=96, base_lr=2e-5):
    """Reproduce the bug in ~10 lines, with no model and no data.

    Returns (before, after) lists of per-group learning rates.
    """
    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

    lrs = mup_lr_table(Nu, Nx, base_lr)
    dummy = [torch.nn.Parameter(torch.zeros(1)) for _ in range(4)]
    opt = torch.optim.AdamW(
        [{'params': [dummy[i]], 'lr': lrs[k]}
         for i, k in enumerate(('A_log', 'lin_B', 'lin_C', 'other'))])

    before = [g['lr'] for g in opt.param_groups]
    CosineAnnealingWarmupRestarts(opt, first_cycle_steps=50000, cycle_mult=1.0,
                                  max_lr=2e-4, min_lr=base_lr,
                                  warmup_steps=1000, gamma=1.0)
    after = [g['lr'] for g in opt.param_groups]
    return before, after


if __name__ == '__main__':
    print(__doc__.split('USAGE')[0])
    print('=' * 78)
    print(f'μP learning rates at the four released widths (base_lr = 2e-5)\n')
    print(f'{"paper":<6s}{"Nu":>6s}{"Nx":>5s}   ' +
          ''.join(f'{k:>13s}' for k in ('A_log', 'lin_B', 'lin_C', 'other')))
    for paper, Nu, Nx in (('m3', 256, 16), ('m4', 512, 32),
                          ('m5', 1024, 64), ('m6', 1536, 96)):
        t = mup_lr_table(Nu, Nx, 2e-5)
        print(f'{paper:<6s}{Nu:>6d}{Nx:>5d}   ' +
              ''.join(f'{t[k]:>13.3e}' for k in ('A_log', 'lin_B', 'lin_C', 'other')))

    print('\n' + '=' * 78)
    print('The scheduler bug, at m6:\n')
    before, after = demonstrate_scheduler_nullification()
    print(f'  before scheduler : {[f"{x:.3e}" for x in before]}')
    print(f'  after  scheduler : {[f"{x:.3e}" for x in after]}')
    print(f'\n  {"IDENTICAL — every bit of μP scaling erased" if len(set(after)) == 1 else "ratios survived"}')
