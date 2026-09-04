#!/usr/bin/env python3
"""Expand __ENV_VAR__ placeholders in a tutorial YAML into a runnable config.

YParams reads plain YAML with no environment-variable expansion, and the config needs
absolute paths that differ per user. Rather than have you hand-edit a file (and forget,
and silently train against the wrong stats directory), we template it.

    python configs/render_config.py configs/tutorial_m3.yaml -o /tmp/m3.rendered.yaml

Placeholders are `__NAME__` for any exported environment variable. Missing variables are
an error, not an empty string -- an empty stat_dir is exactly the silent-corruption case
described in common/fm4npp_naming.md section 5.
"""
import argparse
import os
import re
import sys

PLACEHOLDER = re.compile(r'__([A-Z][A-Z0-9_]*)__')


def render(text):
    missing = set()

    def sub(m):
        name = m.group(1)
        val = os.environ.get(name)
        if not val:
            missing.add(name)
            return m.group(0)
        return val

    out = PLACEHOLDER.sub(sub, text)
    if missing:
        raise SystemExit(
            f'error: unset environment variable(s): {", ".join(sorted(missing))}\n'
            f'       run:  source common/paths.sh')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('template')
    ap.add_argument('-o', '--out', required=True)
    args = ap.parse_args()

    with open(args.template) as f:
        rendered = render(f.read())
    with open(args.out, 'w') as f:
        f.write(rendered)

    print(f'rendered {args.template} -> {args.out}')
    for line in rendered.splitlines():
        if any(k in line for k in ('data_root:', 'stat_dir:')):
            print(f'  {line.strip()}')

    # Fail loudly now rather than 40 minutes into a job.
    import yaml
    cfg = next(iter(yaml.safe_load(rendered).values()))
    for key in ('data_root', 'stat_dir'):
        if not os.path.isdir(cfg[key]):
            print(f'\nWARNING: {key} does not exist: {cfg[key]}', file=sys.stderr)
    bins = os.path.join(cfg['stat_dir'], 'bin_edges_v3_nbins_8_8_6.pkl')
    if not os.path.isfile(bins):
        print(f'\nERROR: {bins} missing. The Voxelizer will SILENTLY recompute bin '
              f'edges and every result will be wrong.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
