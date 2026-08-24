#!/usr/bin/env python3
"""Regression checks for the FM4NPP public-repo defects.

Run against a repo tree:  python verify_bugs.py /path/to/repo
Exits non-zero if any check fails. Cheap: no data, no GPU, no training.
"""
import ast
import contextlib
import io
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train', 'downstream'))

results = []


def check(tag, desc, ok, detail=''):
    results.append((tag, desc, ok, detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {tag}: {desc}' + (f'  -- {detail}' if detail else ''))


def src(rel):
    with open(os.path.join(REPO, rel)) as f:
        return f.read()


print(f'Verifying: {os.path.abspath(REPO)}\n')

# --- B1: downstream_dropout must be resolvable for the tracking config ---
check('B1', 'downstream_dropout defined in mamba_tracking.yaml',
      'downstream_dropout' in src('scripts/configs/mamba_tracking.yaml'),
      'track_finding_trainer.py reads self.params.downstream_dropout')

# --- B3: every real MambaAttentionHead construction passes embed_method='concat' ---
tr = src('train/downstream/track_finding_trainer.py')


def head_callsites(source):
    """AST, not substrings: comments and prose must not count as call sites."""
    sites = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == 'MambaAttentionHead':
            kw = {k.arg: k.value for k in node.keywords}
            em = kw.get('embed_method')
            sites.append(em.value if isinstance(em, ast.Constant) else None)
    return sites


sites = head_callsites(tr)
check('B3', "every MambaAttentionHead call site passes embed_method='concat'",
      len(sites) > 0 and all(s == 'concat' for s in sites),
      f'{sites}')

# --- B4: the noise-loss guard must key on something the heads actually emit ---
loss = src('train/downstream/loss.py')
check('B4', 'noise-loss guard keys on "noise_logits", not never-emitted "noise_probs"',
      '"noise_logits" in outputs' in loss and '"noise_probs" in outputs' not in loss)

# --- B5: track-reg loss gated on the model producing it ---
check('B5', 'track-reg loss also requires "track_reg_result" in outputs',
      '"track_reg_result" in outputs and all("track_info"' in loss)

# --- B6: loss_track_reg / loss_pid_ce bound before the num_matched branch ---
fn = next(n for n in ast.walk(ast.parse(loss))
          if isinstance(n, ast.FunctionDef) and n.name == 'compute_point_loss')
pre = []
for stmt in fn.body:
    if isinstance(stmt, ast.If) and 'num_matched' in ast.dump(stmt.test):
        break
    pre.append(stmt)
pre_src = '\n'.join(ast.dump(s) for s in pre)
check('B6', 'loss_track_reg/loss_pid_ce initialised before the num_matched branch',
      'loss_track_reg' in pre_src and 'loss_pid_ce' in pre_src)

# --- B16: mamba_ssm import must be optional ---
mg = src('fm4npp/models/mambagpt.py')
check('B16', 'mamba_ssm import is guarded (the mamba2 path needs no compiled kernels)',
      'try:' in mg.split('from mamba_ssm import Mamba')[0][-40:])

# --- B18: mid_target detection must not trust RaggedMmap to raise ---
check('B18', 'mid_target presence checked via isdir/len, not a bare try/except',
      'os.path.isdir(mid_dir)' in src('fm4npp/datasets/dataset.py'))

# --- B20: inference() must define amp_enabled before using it ---
inf = next(n for n in ast.walk(ast.parse(tr))
           if isinstance(n, ast.FunctionDef) and n.name == 'inference')
uses_amp = 'amp_enabled' in ast.dump(inf)
assigns_amp = any(isinstance(n, ast.Name) and n.id == 'amp_enabled'
                  and isinstance(n.ctx, ast.Store) for n in ast.walk(inf))
check('B20', 'inference() defines amp_enabled before use',
      (not uses_amp) or assigns_amp,
      'used but never assigned -> NameError on every eval run'
      if uses_amp and not assigns_amp else '')

# --- B22: checkpoint-loading failures must not be swallowed into exit 0 ---
_inf_src = ast.get_source_segment(tr, inf) or ''
check('B22', 'eval raises on checkpoint-loading failure instead of returning exit 0',
      'Checkpoint loading failed' not in _inf_src or 'raise' in _inf_src)

# --- B8: seeding present and encoded in artifact names ---
tt = src('train/downstream/train_track_finding.py')
check('B8', 'training seeds RNGs and puts the seed in artifact names',
      'def set_seed' in tt and '--seed' in tt and '_seed{args.seed}' in tt)

# --- B14: example_usage.py refers to names that exist ---
def code_only(source):
    """Strip comments and all docstrings; prose explaining a past bug is not the bug."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)


ex = code_only(src('example_usage.py'))
check('B14a', 'example_usage.py does not import the nonexistent Mamba2GPT',
      'Mamba2GPT' not in ex)
check('B14b', "example_usage.py does not pass embed_method='additive'",
      "'additive'" not in ex)

# --- B12/B13: converter exists and docs state 4D ---
check('B12', 'scripts/prepare_data.py (Zenodo -> RaggedMmap) exists',
      os.path.exists(os.path.join(REPO, 'scripts', 'prepare_data.py')))
check('B13', 'docs state 4D features, not 30D',
      '30D per point' not in src('README.md'))

# --- B15: requirements list the deps imported at module load ---
req = src('requirements.txt')
missing = [m for m in ['einops', 'ruamel', 'scikit-learn', 'plotly', 'cosine_annealing_warmup']
           if m not in req]
check('B15', 'requirements.txt lists all hard imports', not missing,
      f'missing: {missing}' if missing else '')

# --- B23: the per-point CSV export must be reachable, and its analysis shipped ---
ev = src('train/downstream/eval_track_finding.py')
has_flag = 'save_csv' in ev
wired = 'save_csv=args.save_csv' in ev
has_script = os.path.exists(os.path.join(REPO, 'train', 'downstream',
                                         'calculate_tracking_eff_purity.py'))
check('B23', 'per-point CSV export reachable from the CLI, eff/purity analysis present',
      has_flag and wired and has_script,
      f'cli_flag={has_flag} wired={wired} eff_purity_script={has_script}')

# --- B24: per-batch ARI must accumulate, not overwrite ---
check('B24', 'validate loop accumulates ARI over the batch (+=, not =)',
      'adjusted_rand_index += adjusted_rand_score' in tr,
      'with "=" it reports ARI(last event)/batch_size')

# --- B25: the released checkpoints must all have a config to run them ---
cfg_txt = src('scripts/configs/mamba_tracking.yaml')
sizes = [c for c in ('d9_m1_k30_p20', 'd9_m3_k30_p20', 'd9_m4_k30_p20', 'd9_m5_k30_p20')
         if c in cfg_txt]
check('B25', 'every published checkpoint has a matching config', len(sizes) == 4,
      f'{len(sizes)}/4 present: {sizes}')

# --- B26: LR cycle length decoupled from max_epochs ---
check('B26', 'LR cycle length is its own key, not hardwired to max_epochs',
      'first_cycle_steps=first_cycle' in tr,
      'shortening max_epochs silently rewrites the LR schedule')

# --- B27: feature-cache path present (makes the large models tractable) ---
check('B27', 'frozen-backbone feature cache available',
      os.path.exists(os.path.join(REPO, 'scripts', 'cache_features.py'))
      and os.path.exists(os.path.join(REPO, 'train', 'downstream', 'cached_dataset.py'))
      and 'feature_cache' in tr)

# --- B2: the heads in model.py must be constructable at all ---
try:
    with contextlib.redirect_stdout(io.StringIO()):
        import model as _m
        _m.MambaAttentionHead(input_dim=256)
        _m.MambaHead(input_dim=256)
    check('B2', 'model.py heads construct (no undefined Embedder)', True)
except Exception as e:  # noqa: BLE001
    check('B2', 'model.py heads construct (no undefined Embedder)', False,
          f'{type(e).__name__}: {e}')

# --- Architectural: the LIVE track-finding head must build EmbedderConcat ---
try:
    import torch  # noqa: F401
    from trackinghead import MambaAttentionHead
    # use the embed_method THIS REPO's trainer actually passes (None -> class default)
    em = sites[0] if sites else None
    kwargs = dict(input_dim=256, num_layers=0, num_embedder_layers=0,
                  d_state=64, d_conv=4, expand=2, num_feature_layers=12,
                  num_prototypes=150, dropout=0.1)
    if em is not None:
        kwargs['embed_method'] = em
    with contextlib.redirect_stdout(io.StringIO()):
        h = MambaAttentionHead(**kwargs)
    total = sum(p.numel() for p in h.parameters())
    check('ARCH', "head built as this repo's trainer builds it has the published param count",
          total == 2285646,
          f'{total} via embed_method={em!r} (published 2285646 needs concat; add yields 2203918)')
except Exception as e:  # noqa: BLE001
    check('ARCH', 'track-finding head parameter count', False, f'{type(e).__name__}: {e}')

failed = [r for r in results if not r[2]]
print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
if failed:
    print('FAILED: ' + ', '.join(r[0] for r in failed))
sys.exit(1 if failed else 0)
