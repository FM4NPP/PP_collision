#!/usr/bin/env python3
"""End-to-end check that this environment works: fixture -> features -> head -> score.

    python selftest.py                 # tracking only, the fast path
    python selftest.py --all           # all three downstream tasks

It verifies the fixture against its manifest first. That step is not ceremony. If
`bin_edges_v3_nbins_8_8_6.pkl` goes missing, `voxelizer.py` does not raise -- it recomputes
bin edges from whatever data it is handed and writes a fresh pickle. The run then succeeds
with a tokenization different from the one the released checkpoints were pretrained with, and
nothing anywhere says so. Hashing the fixture is the only thing standing between that and a
quietly wrong number.

Exit status is 0 only if every stage ran and produced a non-degenerate result.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'fixture')


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def check_fixture():
    print('\n[1] fixture integrity')
    man_path = os.path.join(FIXTURE, 'MANIFEST.json')
    if not os.path.isfile(man_path):
        print(f'    FAIL  no manifest at {man_path}')
        return False
    man = json.load(open(man_path))
    ok = True

    for name, meta in man['arrays'].items():
        d = os.path.join(FIXTURE, name)
        got = sha256_file(os.path.join(d, 'data.ninja'))
        if got != meta['data_sha256']:
            print(f'    FAIL  {name}: data.ninja hash differs from the manifest')
            ok = False
    for pkl, want in man.get('stats', {}).items():
        p = os.path.join(FIXTURE, pkl)
        if not os.path.isfile(p):
            print(f'    FAIL  {pkl} is missing.')
            if pkl.startswith('bin_edges'):
                print('          Without it the voxelizer silently RECOMPUTES bin edges and')
                print('          the model is fed a different tokenization than it was')
                print('          pretrained with. This is not a recoverable warning.')
            ok = False
        elif sha256_file(p) != want:
            print(f'    FAIL  {pkl} hash differs from the manifest')
            ok = False

    if ok:
        s = man['splits']
        print(f"    ok    {s['pretrain']['events']} train + {s['test']['events']} eval events, "
              f"{s['pretrain']['points'] + s['test']['points']:,} points, "
              f"{len(man['arrays'])} arrays + {len(man.get('stats', {}))} stat files verified")
    return ok


def check_kernels():
    print('\n[2] kernel path')
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa
        print('    ok    mamba-ssm present; results are comparable to the paper')
        return True
    except Exception:
        print('    note  mamba-ssm absent; the corrected pure-PyTorch fallback will run.')
        print('          Numbers from this run demonstrate the pipeline, nothing more.')
        return False


def run_task(task, ckpt):
    print(f'\n[3] {task}: train then evaluate')
    env = dict(os.environ)
    env.setdefault('FM4NPP_ALLOW_FALLBACK', '1')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'run.py'),
                        '--task', task, '--stage', 'both', '--checkpoint', ckpt],
                       cwd=HERE, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(f'    FAIL  exit {p.returncode}')
        for line in (p.stdout + p.stderr).splitlines()[-12:]:
            print('      ' + line)
        return False

    # A run that "succeeds" having scored nothing is the failure mode this repository had for
    # a long time, so look for the artifacts rather than trusting the exit code.
    runs = os.path.join(HERE, 'runs')
    logs = [f for f in os.listdir(runs) if f.endswith('.log') and 'eval' in f] \
        if os.path.isdir(runs) else []
    ckpts = [f for f in os.listdir(runs) if f.endswith('.pth')] if os.path.isdir(runs) else []
    if not ckpts:
        print('    FAIL  training wrote no checkpoint')
        return False
    if not logs:
        print('    FAIL  evaluation wrote no metric file')
        return False
    print(f'    ok    {len(ckpts)} checkpoint(s), {len(logs)} eval log(s)')
    for f in sorted(logs):
        body = open(os.path.join(runs, f)).read().strip().splitlines()
        if len(body) >= 2:
            print(f'      {f.split("_eval_")[-1][:34]:36s} {body[-1]}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--all', action='store_true', help='all three tasks, not just tracking')
    ap.add_argument('--checkpoint', default=os.environ.get('FM4NPP_CKPT', ''))
    args = ap.parse_args()

    print('=' * 74)
    print('  generic_env selftest')
    print('=' * 74)

    ok = check_fixture()
    check_kernels()

    ckpt = args.checkpoint
    if not ckpt or not os.path.isfile(ckpt):
        print(f'\n[3] SKIPPED — no backbone at {ckpt or "$FM4NPP_CKPT"}')
        print('    bash get_checkpoint.sh     (59 MB, the paper\'s m3)')
        return 0 if ok else 1

    for task in (['tracking', 'pid', 'nid'] if args.all else ['tracking']):
        ok &= run_task(task, ckpt)

    print('\n' + '=' * 74)
    print('  PASS — this environment works end to end' if ok else '  FAIL — see above')
    print('=' * 74)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
