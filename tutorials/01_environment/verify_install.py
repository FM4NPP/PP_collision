#!/usr/bin/env python3
"""Module 01 -- verify the environment before you spend GPU hours on it.

    INPUT   the environment built by setup_perlmutter.sh and fetch_data.sh
    OUTPUT  a PASS/FAIL table; exit 0 if everything a tutorial needs is present

Every check here corresponds to a way this stack fails quietly. The one worth reading
twice is BINS: if the bin-edges pickle is missing, the Voxelizer silently recomputes
bin edges from your data and every downstream number is wrong with no error raised.

    source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
    python 01_environment/verify_install.py
"""
import importlib
import os
import sys

RESULTS = []


def check(tag, desc, fn, required=True):
    """Run fn(); record PASS/FAIL/WARN with whatever detail fn returns or raises."""
    try:
        detail = fn()
        RESULTS.append(('PASS', tag, desc, detail or ''))
        return True
    except Exception as e:                                    # noqa: BLE001
        RESULTS.append(('FAIL' if required else 'WARN', tag, desc,
                        f'{type(e).__name__}: {e}'))
        return False


def env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f'{name} unset -- did you `source common/paths.sh`?')
    return v


# --------------------------------------------------------------- python deps
def _torch():
    import torch
    cuda = 'CUDA ok' if torch.cuda.is_available() else 'NO CUDA (cpu only)'
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return f'torch {torch.__version__}, {cuda}, {n} device(s)'


def _imports():
    mods = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'plotly', 'pandas',
            'einops', 'ruamel.yaml', 'mmap_ninja', 'huggingface_hub',
            'cosine_annealing_warmup']
    missing = [m for m in mods if not importlib.util.find_spec(m.split('.')[0])]
    if missing:
        raise RuntimeError(f'missing: {", ".join(missing)}')
    return f'{len(mods)} modules importable'


def _no_mamba_ssm():
    """Absence is CORRECT. Present is fine too, just unnecessary."""
    if importlib.util.find_spec('mamba_ssm'):
        return 'mamba_ssm present (not required; harmless)'
    return 'mamba_ssm absent -- correct, the Mamba2 path is pure PyTorch'


def _tsne():
    from sklearn.manifold import TSNE                          # noqa: F401
    return 'sklearn.manifold.TSNE importable'


# --------------------------------------------------------------- the repo
def _repo():
    root = env('FM4NPP_ROOT')
    if not os.path.isdir(os.path.join(root, 'fm4npp')):
        raise RuntimeError(f'no fm4npp package under {root}')
    sys.path.insert(0, root)
    import fm4npp                                              # noqa: F401
    from fm4npp.models.mambagpt import MambaGPT                # noqa: F401
    return f'fm4npp imports from {root}'


def _branch():
    import subprocess
    root = env('FM4NPP_ROOT')
    b = subprocess.run(['git', '-C', root, 'rev-parse', '--abbrev-ref', 'HEAD'],
                       capture_output=True, text=True).stdout.strip()
    if b != 'downstream-reproducibility':
        raise RuntimeError(f"on '{b}', need 'downstream-reproducibility' "
                           f"(main's downstream code does not run)")
    return b


def _bins():
    stats = env('FM4NPP_STATS')
    need = ['bin_edges_v3_nbins_8_8_6.pkl', 'loss_bin_pp.pkl', 'loss_weight_pp.pkl']
    missing = [f for f in need if not os.path.isfile(os.path.join(stats, f))]
    if missing:
        raise RuntimeError(f'missing from {stats}: {", ".join(missing)} '
                           f'-- the Voxelizer will SILENTLY rebin and every '
                           f'downstream number will be wrong')
    return f'all 3 stat files in {stats}'


# --------------------------------------------------------------- data
def _mmap(root_var, split, label):
    from mmap_ninja import RaggedMmap
    root = env(root_var)
    lens = {}
    for name in ('features', 'seg_target', 'reg_target', 'pid_target'):
        d = os.path.join(root, f'{name}_{split}')
        if not os.path.isdir(d):
            raise RuntimeError(f'missing {d} (all four are opened unconditionally)')
        lens[name] = len(RaggedMmap(d))
    if len(set(lens.values())) != 1:
        raise RuntimeError(f'length mismatch: {lens}')
    return f'{label}: {next(iter(lens.values())):,} events x 4 arrays'


# --------------------------------------------------------------- checkpoints
def _ckpt(var, paper, expect_m, width, d_state):
    import torch
    sys.path.insert(0, env('FM4NPP_ROOT'))
    from fm4npp.models.mambagpt import MambaGPT
    path = env(var)
    if not os.path.isfile(path):
        raise RuntimeError(f'not found: {path}')
    model = MambaGPT(embed_dim=width, num_layers=12, d_state=d_state, d_conv=4,
                     expand=2, klen=30, dropout=0.0,
                     embed_method='add', pe_method='nerf')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ck['model_state'].items()}
    model.load_state_dict(state, strict=True)          # strict: shape mismatch = loud
    n = sum(p.numel() for p in model.parameters())
    # NOTE: these are the counts the CHECKPOINTS actually contain, which are ~7% below
    # the paper's stated figures (m3: 4.92M vs 5.3M; m6: 174.7M vs 188M). The ratio is
    # 0.93 at every size -- a systematic discrepancy, not a download problem.
    if abs(n / 1e6 - expect_m) > max(0.05, expect_m * 0.02):
        raise RuntimeError(f'{n/1e6:.2f}M params, expected ~{expect_m}M')
    return f'paper {paper}: {n/1e6:.2f}M params, strict load ok'


def main():
    print(f'{"":6s}{"CHECK":<10s}{"WHAT":<48s}DETAIL')
    print('-' * 110)

    check('PY', 'python >= 3.9',
          lambda: (_ for _ in ()).throw(RuntimeError(f'{sys.version_info}'))
          if sys.version_info < (3, 9) else f'{sys.version.split()[0]}')
    check('TORCH', 'torch imports and sees a GPU', _torch)
    check('DEPS', 'all runtime imports available', _imports)
    check('MAMBA', 'mamba_ssm not required', _no_mamba_ssm, required=False)
    check('TSNE', 'sklearn TSNE available (module 04)', _tsne)
    check('REPO', 'fm4npp package importable', _repo)
    check('BRANCH', 'on downstream-reproducibility', _branch)
    check('BINS', 'stat pickles present (silent-corruption guard)', _bins)
    check('DATA1', 'pretrain root, train split (module 03)',
          lambda: _mmap('FM4NPP_PRETRAIN_ROOT', 'pretrain', 'train'))
    check('DATA2', 'pretrain root, val split (module 03)',
          lambda: _mmap('FM4NPP_PRETRAIN_ROOT', 'test', 'val'))
    check('DATA3', 'eval root, test split (module 04)',
          lambda: _mmap('FM4NPP_EVAL_ROOT', 'test', 'test'))
    check('CKPT3', 'paper m3 checkpoint loads',
          lambda: _ckpt('FM4NPP_CKPT_M3', 'm3', 4.92, 256, 16))
    check('CKPT6', 'paper m6 checkpoint loads',
          lambda: _ckpt('FM4NPP_CKPT_M6', 'm6', 174.69, 1536, 96))

    for status, tag, desc, detail in RESULTS:
        mark = {'PASS': ' ok ', 'FAIL': 'FAIL', 'WARN': 'warn'}[status]
        print(f'[{mark}] {tag:<10s}{desc:<48s}{detail}')

    fails = [r for r in RESULTS if r[0] == 'FAIL']
    print('-' * 110)
    print(f'{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed')
    if fails:
        print('\nfailed: ' + ', '.join(r[1] for r in fails))
        return 1
    print('\nEnvironment is ready. Continue to 02_mu_parameterization/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
