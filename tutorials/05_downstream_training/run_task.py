#!/usr/bin/env python3
"""Run any of the three downstream tasks on a handful of events.

    python run_task.py --task tracking --mode eval    # score a released head
    python run_task.py --task pid      --mode train   # a few training steps
    python run_task.py --task nid      --mode both

WHAT "5 EVENTS" CAN AND CANNOT SHOW YOU

`--mode eval` loads a head the authors trained on 70,000 events and scores it. Five events
is a small sample of a real measurement, so the number is noisy but *meaningful*: if your
environment is wrong you will see it immediately, which is the whole point.

`--mode train` runs a few optimizer steps from scratch. The loop is real; the resulting
metric is **not**. A tracking adapter has 2.2M parameters and five events hold a few
thousand spacepoints. Treat a number from `--mode train` as proof the code runs, nothing
more. This script prints that caveat next to any number it produces this way.

THE TRAP THAT WILL BITE YOU

`--eventnumber` limits the *training* split only. The test loader reads every event in
`data_root_test`, and it uses `drop_last=True`. So on a 5-event test directory, an
`--eval_batch_size` above 1 yields **zero** batches -- validation loss becomes `nan`, `nan`
never beats the best-so-far, no checkpoint is ever written, and the eval you run afterwards
fails with "Checkpoint loading failed" pointing at a file that was never created. The batch
size is pinned to 1 below and should stay there.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TASKS = {
    'tracking': dict(
        train='train/downstream/train_track_finding.py',
        evaluate='train/downstream/eval_track_finding.py',
        config='scripts/configs/mamba_tracking.yaml',
        metric='ARI (adjusted Rand index) over predicted vs true track assignments',
    ),
    'pid': dict(
        train='train/downstream/train_point_classification.py',
        evaluate='train/downstream/eval_point_classification.py',
        config='scripts/configs/mamba_pointclass.yaml',
        metric='per-class precision/recall over 5 particle classes, plus accuracy',
    ),
    'nid': dict(
        train='train/downstream/train_point_classification.py',
        evaluate='train/downstream/eval_point_classification.py',
        config='scripts/configs/mamba_noiseid.yaml',
        metric='precision/recall for noise vs signal (2 classes)',
    ),
}


def run(cmd, dry):
    print('\n$ ' + ' \\\n    '.join(cmd) + '\n', flush=True)
    if dry:
        return 0
    return subprocess.call(cmd, cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--mode', default='both', choices=['train', 'eval', 'both'])
    ap.add_argument('--config', default='d9_m1_k30_p20',
                    help="repo config name; d9_m1_k30_p20 is the paper's m3 (5.3M)")
    ap.add_argument('--n_events', type=int, default=5)
    ap.add_argument('--work', default=os.environ.get('FM4NPP_WORK', ''),
                    help='defaults to $FM4NPP_WORK from common/paths.sh')
    ap.add_argument('--data_root_test', default=None,
                    help='a directory holding only a few events; see the README')
    ap.add_argument('--checkpoint', default=None,
                    help='released head to score in --mode eval')
    ap.add_argument('--run_num', default='tut')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if not args.work:
        sys.exit('set FM4NPP_WORK (source common/paths.sh) or pass --work')
    spec = TASKS[args.task]
    runs = os.path.join(args.work, 'tutorial_runs')
    evals = os.path.join(args.work, 'tutorial_evals')
    os.makedirs(runs, exist_ok=True)
    os.makedirs(evals, exist_ok=True)

    print(f'task     {args.task}')
    print(f'metric   {spec["metric"]}')
    print(f'config   {spec["config"]}  [{args.config}]')

    rc = 0
    if args.mode in ('train', 'both'):
        cmd = [sys.executable, spec['train'],
               '--yaml_config', spec['config'], '--config', args.config,
               '--root_dir', runs + '/', '--eventnumber', str(args.n_events),
               '--train_batch_size', '2', '--run_num', args.run_num,
               '--seed', str(args.seed)]
        rc |= run(cmd, args.dry_run)
        print('\n  NOTE: any metric above came from %d events. It shows the loop runs.'
              '\n  It is not a measurement -- see the module README.' % args.n_events)

    if args.mode in ('eval', 'both'):
        cmd = [sys.executable, spec['evaluate'],
               '--yaml_config', spec['config'], '--config', args.config,
               '--root_dir', evals + '/', '--eventnumber', str(args.n_events),
               # pinned: see the docstring. Above 1, a tiny test split yields no batches.
               '--eval_batch_size', '1',
               '--run_num', args.run_num, '--seed', str(args.seed)]
        if args.data_root_test:
            cmd += ['--data_root_test', args.data_root_test]
        if args.checkpoint:
            cmd += ['--checkpoint', args.checkpoint] if args.task == 'tracking' else \
                   ['--checkpoint_dir', os.path.dirname(args.checkpoint)]
        else:
            cmd += ['--checkpoint_dir', runs + '/']
        rc |= run(cmd, args.dry_run)

    return rc


if __name__ == '__main__':
    sys.exit(main())
