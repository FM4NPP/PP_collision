# Naming, and other places the code disagrees with the paper

Read this once. Every one of these has cost someone a day.

## 1. Model sizes: repo `m1` ≠ paper `m1`

| paper | width | d_state | params (paper) | params (checkpoint) | repo config | HF checkpoint |
|---|---|---|---|---|---|---|
| m1 | 64 | — | 0.34M | not released | `d9_m64_k30_p20` | not released |
| m2 | 128 | — | 1.3M | not released | `d9_m128_k30_p20` | not released |
| m3 | 256 | 16 | 5.3M | **4.92M** | `d9_m1_k30_p20` | `pp_nerf_m1_k30.ckpt` |
| m4 | 512 | 32 | 21M | — | `d9_m3_k30_p20` | `pp_nerf_m3_k30.ckpt` |
| m5 | 1024 | 64 | 84M | — | `d9_m4_k30_p20` | `pp_nerf_m4_k30.ckpt` |
| m6 | 1536 | 96 | 188M | **174.69M** | `d9_m5_k30_p20` | `pp_nerf_m5_k30.ckpt` |

The two params columns disagree by a consistent ~7% (ratio 0.929 at both sizes we can
check). Verified by loading each checkpoint with `strict=True` and counting. Quote the
checkpoint column for anything reproducible.

The checkpoint filename follows the **repo** convention. So the paper's m6 — the headline
188M model — is the file called `pp_nerf_m5_k30.ckpt`.

Every released checkpoint is a **Mamba2** backbone, 12 layers, `klen=30`. Verified by loading
each with `strict=True`.

Note `d_state = width / 16` exactly across the ladder. This matters for μP (module 02):
textbook μP holds the state/head dimension fixed while width grows. Here it does not, so the
scaling exponents mean something different than the standard derivation gives.

## 2. `split='pretrain'` means the *labeled training* split

Not the unlabeled pretraining corpus. It is a historical name.

`get_data_loader()` hardcodes `split='pretrain'` for the train loader and `split='test'` for
validation. You cannot change this from config.

## 3. The split is a suffix, not a directory

`prepare_data.py --out $DIR --split pretrain` writes:

```
$DIR/features_pretrain/     $DIR/seg_target_pretrain/
$DIR/reg_target_pretrain/   $DIR/pid_target_pretrain/
```

There is no `$DIR/pretrain/`. To get a data root usable for pretraining you run the script
**twice into the same `--out`**, once per split. Any `train/`, `val/`, `test/` directory
level you see in our examples is a name we chose, not something the code knows about.

## 4. All four arrays must exist even when unused

`TPCBatchDataset.__init__` opens `features_*`, `seg_target_*`, `reg_target_*` and
`pid_target_*` unconditionally. Pretraining never reads `reg_target` or `pid_target`, but
their directories must be present or construction throws.

## 5. `stat_dir` must point at the repo's `stats/`

If `bin_edges_v3_nbins_8_8_6.pkl` is not found, the `Voxelizer` **silently recomputes** the
bin edges from whatever data you handed it. No error, no warning. The tokenization then no
longer matches what the pretrained checkpoints were trained on, and every downstream number
is quietly wrong.

The file ships in the repo at `stats/`. The default in `mamba_pretrain.yaml` is a dead
personal path (`/global/homes/d/dpark1/...`). Always set `stat_dir` explicitly.

## 6. `mamba-ssm` is not required

`README.md` and `SETUP.md` in the official repo tell you to install it.
`requirements.txt` correctly has it commented out. The requirements file is right.

The released checkpoints are Mamba2 and run on `fm4npp/models/mamba2.py`, which is pure
PyTorch with every fused-kernel import `try/except`-guarded. `mamba-ssm` is reachable only
from `Mamba1GPT`, which no released checkpoint uses.

The one exception: the *shipped pretraining script* hardcodes `Mamba1GPT`, so it does need
it. Module 03 switches to `MambaGPT`, which removes the requirement and also matches the
released architecture.

## 7. Dead config keys

Present in the YAMLs, read by nothing: `base_dim`, `init_std`, `portion`, `n_data_idx`,
`data_version`, `max_gt_classes` (pretrain path), `checkpoint_dir` (pretrain path — the
`--root_dir` CLI argument is what actually controls output location), `headdim`, `ngroups`.

`d_conv` and `expand` are in the pretrain YAML but the model is constructed without them, so
they fall back to class defaults (4 and 2). Those happen to match — change the YAML and
nothing happens.

## 8. Two functions named `initialize_mamba2`

Different signatures, different bodies, same file. `initialize_mamba2(model, d_state,
embed_dim)` is the width-dependent backbone init; `initialize_mamba2(model, num_layers,
num_residuals=1)` is the depth-dependent head init defined ~450 lines later, shadowing the
first. Module 02 covers both.
