#!/usr/bin/env python3
"""One entry point for the three downstream tasks on the committed fixture.

    python run.py --task tracking --stage train
    python run.py --task pid      --stage eval
    python run.py --task nid      --stage both

Every invocation prints which Mamba2 implementation is in use, because that single fact
decides whether a number means anything. The repository shipped for months with a
pure-PyTorch fallback that computed a different model -- an ungated norm on the wrong tensor,
and an EMA in place of the state-space scan that discarded B and C. It loaded released
checkpoints cleanly with strict=True and scored about 0.09 ARI below the paper, and nothing
anywhere said which path had run. This banner exists so that cannot happen again silently.

This environment defaults to that fallback, now corrected. See README.md for what that means
for the numbers.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

TASKS = {
    'tracking': dict(config='configs/generic_tracking.yaml',
                     train='train_track_finding.py', evaluate='eval_track_finding.py',
                     metric='ARI over predicted vs true track assignments'),
    'pid':      dict(config='configs/generic_pid.yaml',
                     train='train_point_classification.py',
                     evaluate='eval_point_classification.py',
                     metric='per-class precision/recall over 5 particle classes'),
    'nid':      dict(config='configs/generic_nid.yaml',
                     train='train_point_classification.py',
                     evaluate='eval_point_classification.py',
                     metric='precision/recall for signal vs noise'),
}


def kernel_banner():
    """Report which Mamba2 path will run, and how much to trust what comes out of it."""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa
        real = True
    except Exception:
        real = False
    # This process always passes FM4NPP_ALLOW_FALLBACK=1 down to the child when the kernels
    # are missing, so there are only two states to report, not three.
    bar = '─' * 74
    print(bar)
    if real:
        print('  kernels   mamba-ssm present — the fused Triton path will run')
        print('  numbers   comparable to the paper, if the data and training are')
    else:
        print('  kernels   mamba-ssm ABSENT — corrected pure-PyTorch fallback')
        print('  numbers   DEMONSTRATION ONLY. This path has never been compared against')
        print('            the real kernels on any machine. Validate before quoting:')
        print('              python ../scripts/check_kernel_equivalence.py')
    print(bar)
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--stage', default='both', choices=['train', 'eval', 'both'])
    ap.add_argument('--checkpoint', default=os.environ.get('FM4NPP_CKPT', ''),
                    help='pretrained backbone; defaults to $FM4NPP_CKPT')
    ap.add_argument('--config_name', default='d9_m1_k30_p20')
    ap.add_argument('--run_num', default='gen')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    spec = TASKS[args.task]
    real = kernel_banner()
    print(f'  task      {args.task}')
    print(f'  metric    {spec["metric"]}')
    if not args.checkpoint:
        sys.exit('\nNo backbone. Pass --checkpoint, or run: bash get_checkpoint.sh')
    if not os.path.isfile(args.checkpoint):
        sys.exit(f'\nNo such checkpoint: {args.checkpoint}')
    print(f'  backbone  {os.path.basename(args.checkpoint)}')
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)

    env = dict(os.environ)
    env['PYTHONPATH'] = REPO + os.pathsep + os.path.join(REPO, 'train', 'downstream')
    if not real:
        env.setdefault('FM4NPP_ALLOW_FALLBACK', '1')

    common = ['--yaml_config', os.path.join(HERE, spec['config']),
              '--config', args.config_name,
              '--pretrained_ckpt', os.path.abspath(args.checkpoint),
              '--eventnumber', '20', '--run_num', args.run_num]
    rc = 0
    if args.stage in ('train', 'both'):
        cmd = [sys.executable, os.path.join(REPO, 'train', 'downstream', spec['train']),
               *common, '--root_dir', os.path.join(HERE, 'runs') + '/',
               '--train_batch_size', '4', '--seed', str(args.seed)]
        rc |= _run(cmd, HERE, env, args.dry_run)
    if args.stage in ('eval', 'both'):
        cmd = [sys.executable, os.path.join(REPO, 'train', 'downstream', spec['evaluate']),
               *common, '--root_dir', os.path.join(HERE, 'runs') + '/',
               '--checkpoint_dir', os.path.join(HERE, 'runs') + '/',
               # pinned: the test loader uses drop_last=True, so anything above 1 on a
               # 20-event split yields zero batches -> nan -> no checkpoint -> a confusing
               # "Checkpoint loading failed" one step later.
               '--eval_batch_size', '1', '--seed', str(args.seed)]
        rc |= _run(cmd, HERE, env, args.dry_run)

    print('\n  Reminder: 20 events. This shows the pipeline runs; it does not measure '
          'anything.')
    return rc


def _run(cmd, cwd, env, dry):
    print('\n$ ' + ' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd))
    return 0 if dry else subprocess.call(cmd, cwd=cwd, env=env)


if __name__ == '__main__':
    sys.exit(main())
