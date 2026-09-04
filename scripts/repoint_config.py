#!/usr/bin/env python3
"""Rewrite PP_collision's tracking config to point at your own paths.

The published config carries the paths of the machines the work was done on: NERSC CFS for
the pretraining corpus and statistics, a BNL mount (`/mldata/sli/...`) for the labeled
splits, and one user's scratch for outputs. The BNL paths do not exist on Perlmutter, and
the failure they produce is a bare FileNotFoundError several seconds into a job, which is a
slow way to learn you had a typo.

This rewrites the four keys that matter, in place, and prints what it changed.

    python repoint_config.py scripts/configs/mamba_tracking.yaml --work $SCRATCH/fm4npp

Pass --dry-run to see the substitutions without writing.
"""
import argparse
import os
import re
import shutil
import sys

# published path -> key it fills, and where it should point under --work
SUBS = [
    ('/global/cfs/cdirs/m4722/NPFM/data/pp_12M_mmap', 'data_root',      '{work}/data/train'),
    ('/mldata/sli/sphenix_fm/pp_100k_mmap-particle_ids/', 'data_root_train/test',
                                                          '{work}/data/train/'),
    ('/global/cfs/cdirs/m4722/NPFM/data/stats', 'stat_dir',
                                                '{work}/PP_collision/stats'),
    ('/pscratch/sd/d/dpark1/NPFN/test', 'checkpoint_dir, downstream_dir', '{work}/runs'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='path to mamba_tracking.yaml')
    ap.add_argument('--work', required=True,
                    help='your working root, e.g. $SCRATCH/fm4npp')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    work = args.work.rstrip('/')
    if not os.path.isfile(args.config):
        sys.exit(f'no such config: {args.config}')

    src = open(args.config).read()
    out = src
    for old, key, new_t in SUBS:
        new = new_t.format(work=work)
        n = out.count(old)
        print(f'  {key:28s} {n} occurrence(s)')
        print(f'      {old}\n   -> {new}')
        out = out.replace(old, new)

    if out == src:
        print('\nnothing changed -- already repointed, or an unexpected config version.')
        return
    if args.dry_run:
        print('\n--dry-run: not written')
        return

    shutil.copy2(args.config, args.config + '.orig')
    open(args.config, 'w').write(out)
    print(f'\nwritten; original saved as {args.config}.orig')

    # Surface anything still pointing off your filesystem, so it fails now rather than in
    # the queue twenty minutes from now.
    leftovers = sorted(set(re.findall(r'(?:/global/cfs|/mldata|/pscratch)\S*', out)))
    if leftovers:
        print('\nstill referencing paths outside --work (check these exist for you):')
        for p in leftovers:
            print(f'  {p}   {"ok" if os.path.exists(p) else "MISSING"}')


if __name__ == '__main__':
    main()
