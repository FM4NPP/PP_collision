# Running this repository on Perlmutter

The downstream track-finding task end to end on NERSC Perlmutter, from an empty `$SCRATCH`
to a number you can compare against the paper.

Everything here targets the **`downstream-reproducibility`** branch. The published `main`
does not run — it fails at import before reaching a GPU.

Two facts to carry through the whole tutorial, because both cost people days:

- **The checkpoint names do not match the paper.** `pp_nerf_m1_k30.ckpt` is the paper's
  **m3**; `pp_nerf_m5_k30.ckpt` is the paper's **m6**. Download "m1" expecting the paper's
  m1 and you get a model 16x larger.
- **You DO need `mamba-ssm`.** An earlier version of this document said the opposite. It
  was wrong, and it cost roughly 0.09 ARI. Without the compiled kernels
  `fm4npp/models/mamba2.py` falls back to a pure-PyTorch path that used to compute a
  *different function* -- an ungated norm on the wrong tensor, and an EMA in place of the
  SSD scan that threw away `B` and `C` entirely. Released checkpoints loaded into it
  cleanly and scored about 0.86 where the paper reports 0.9448. Both errors are fixed and
  the fallback now tracks the kernels, but it is orders of magnitude slower and refuses to
  run unless you set `FM4NPP_ALLOW_FALLBACK=1`. Install the kernels:

  ```bash
  bash tutorials/01_environment/install_kernels.sh   # fetches prebuilt wheels
  python scripts/check_kernel_equivalence.py
  ```

  Do not run `pip install mamba-ssm causal-conv1d` directly. PyPI ships source only, so
  that compiles for 20-60 minutes, fails without `--no-build-isolation` because setup.py
  imports torch, and on the current release drags in `tilelang` and `quack-kernels`. The
  script above reads your torch/CUDA/Python/ABI and fetches the matching prebuilt wheel
  from the upstream GitHub releases instead.

---

## 0. Prerequisites

A NERSC account with access to a project that has GPU hours. The repo's configs reference
project `m4722`, which is where this work originated.

```bash
ssh perlmutter.nersc.gov

export PROJ=m4722
export WORK=$SCRATCH/fm4npp
mkdir -p $WORK && cd $WORK
```

**Check your account suffix before submitting anything.** NERSC GPU allocations usually
carry a `_g` suffix, and a wrong `-A` is rejected at submit time:

```bash
sacctmgr -n show assoc user=$USER format=account%20 | sort -u
```

If that prints `m4722_g`, use `m4722_g` in every `-A` flag below.

## 1. Clone

```bash
cd $WORK
git clone -b downstream-reproducibility https://github.com/FM4NPP/PP_collision.git
cd PP_collision
git log --oneline -3
```

You want to see the `B28` commits at the top. Without them, caching features for the
188M-parameter model needs **2.17 TB** instead of 185 GB.

## 2. Environment

Layer a venv on the PyTorch module rather than building a conda environment. This inherits
a CUDA-correct torch and takes about two minutes:

```bash
module load pytorch/2.6.0          # `module avail pytorch` for what's current
python -m venv --system-site-packages $WORK/venv
source $WORK/venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists `mamba-ssm`, `causal-conv1d` and `triton` as required. They
compile, so install them on a login node, not in a batch job.
Leave them that way.

## 3. Verify before spending an allocation

```bash
python scripts/verify_repro.py .
```

**Expect `22/22 checks passed`.**

| symptom | cause |
|---|---|
| `20/22`, two failures naming `torch` | running system python — re-activate the venv |
| fewer than 20 | you're on `main`, not the branch |

This script is static analysis plus two constructor checks; it needs no GPU and takes a
second. Run it before every job submission until you trust the tree.

## 4. Checkpoints

```bash
mkdir -p $WORK/assets && cd $WORK/assets
huggingface-cli download FM4NPP/PP_collision pp_nerf_m1_k30.ckpt --local-dir .
huggingface-cli download FM4NPP/PP_collision pp_nerf_m5_k30.ckpt --local-dir .
```

`pp_nerf_m1_k30.ckpt` is 59 MB (paper m3, 5.3M params) — start here.
`pp_nerf_m5_k30.ckpt` is 2.1 GB (paper m6, 175M params) — the headline model.

## 5. Data

The configs ship pointing at `/mldata/sli/...`, which is a **BNL path that does not exist
on Perlmutter**. Check CFS first:

```bash
ls /global/cfs/cdirs/m4722/NPFM/data/
```

If a labeled mmap directory is already there, use it. Otherwise pull from Zenodo. The
labeled splits are **under 1 GB** of the 118.5 GB archive, fetched by HTTP range request:

```bash
cd $WORK
python PP_collision/scripts/fetch_labeled_data.py --out $WORK/TPCpp-10M     # ~12 min

python PP_collision/scripts/prepare_data.py \
    --in_dir $WORK/TPCpp-10M/labeled/train --out $WORK/data/train --split pretrain
python PP_collision/scripts/prepare_data.py \
    --in_dir $WORK/TPCpp-10M/labeled/val   --out $WORK/data/val   --split test
python PP_collision/scripts/prepare_data.py \
    --in_dir $WORK/TPCpp-10M/labeled/test  --out $WORK/data/test  --split test
```

**`--split pretrain` for the *training* data is not a typo.** The training dataloader is
hardcoded to that suffix. Passing `test` there produces an empty dataset and no error
message.

## 6. Point the config at your paths

```bash
cd $WORK/PP_collision
python scripts/repoint_config.py scripts/configs/mamba_tracking.yaml --work $WORK
```

Or edit by hand — the four keys that matter are `data_root`, `data_root_train`,
`data_root_test`, `stat_dir`.

`stat_dir` should point at the repo's own `stats/` directory. Those three `.pkl` files are
loaded unconditionally by the trainer and exist nowhere else public; the branch ships them.

## 7. Smoke test on an interactive node

Never debug in the batch queue.

```bash
salloc -A $PROJ -C gpu -q interactive -t 01:00:00 -N 1 --gpus-per-node=1

cd $WORK/PP_collision && source $WORK/venv/bin/activate
python train/downstream/train_track_finding.py \
    --yaml_config scripts/configs/mamba_tracking.yaml --config d9_m1_k30_p20 \
    --root_dir $WORK/runs/ --eventnumber 50 --train_batch_size 4 \
    --run_num smoke --seed 42
```

Read these four lines in order:

| line | what it confirms |
|---|---|
| `✅ Mamba v2 Model Initialized` | backbone built |
| `Nparams: 5292…` | checkpoint loaded with `strict=True` |
| `Total parameters in down_model: 2285646` | head built as `EmbedderConcat` |
| `[sched] cosine cycle=200 epochs, warmup=20` | LR cycle decoupled from run length |

**If `down_model` reads 2,203,918**, the head was built as `EmbedderAdd` and no real
checkpoint will ever load into it. That was the original bug; seeing the wrong number here
means something re-broke.

**If the `[sched]` line is missing**, you are not on the branch. Stop — a short run will
silently compress the entire LR cycle into however many epochs you asked for, spend half
of it in warmup, and anneal to the floor while the loss is still falling.

## 8. Train

```bash
sbatch scripts/run/perlmutter/m1_70k.sh
```

For the 188M model, cache the frozen backbone first:

```bash
sbatch scripts/run/perlmutter/m6_cache.sh    # ~185 GB with --combine_layers
sbatch scripts/run/perlmutter/m6_70k.sh
```

**`--combine_layers` is not optional at width 1536.** With it the cache is 185 GB; without
it, 2.17 TB. Run `myquota` first either way, and remember `$SCRATCH` is purged on a rolling
window — the cache is expensive to rebuild but it is not a backup.

The backbone is frozen for every downstream task in this repo, so its output for a given
event never changes. Caching removes roughly 45% of step time (the backbone forward) plus
another 22% (CPU preprocessing).

## 9. Evaluate and score

```bash
python train/downstream/eval_track_finding.py \
    --yaml_config scripts/configs/mamba_tracking.yaml --config d9_m1_k30_p20 \
    --run_num m1_70k --seed 42 --eventnumber 70000 --eval_batch_size 1 \
    --root_dir $WORK/evals/ --save_csv \
    --csv_output_path $WORK/evals/m1_per_point.csv

python train/downstream/pooled_ari.py $WORK/evals/m1_per_point.csv
```

For an m6 checkpoint trained from a combined cache, add `--combine_weights` with the vector
from `$WORK/cache/m6_70k/cache_meta.json`. Without it the head is rebuilt at the wrong shape
and the load fails with `size mismatch for weighted_avg_weights: [1] vs [12]`.

### Which number to quote

`pooled_ari.py` prints two. **Quote the pooled one.**

The paper states: *"All metrics are computed over the entire test set rather than averaged
per event."* The trainer does the opposite — it scores each event separately and averages.
Pooling is the harder measure, because a single global clustering must also keep tracks
from *different* events apart, and the two differ by about 0.055.

The script's per-event column reproduces the trainer's own `Avg_ARI` to five decimals,
which is what lets you trust the pooled column beside it.

## 10. What you should get

Measured on a single GB10, 6,943 held-out test events, 5,793,045 points:

| model | pooled ARI | per-event ARI | predicted clusters |
|---|---|---|---|
| m1 (paper m3), 70k | **0.8176** | 0.8758 | 112,484 |
| m6 (paper m6), 70k | **0.8591** | 0.9075 | 108,808 |
| ground truth | | | 107,003 |
| **paper m6, Table 2** | **0.9448** | | |

Two things to know about that last row.

**The 0.086 gap is not a metric artifact.** Aggregation (0.053–0.058), decoding choice,
frozen layer-mixing (0.8508 vs 0.8480), training duration, and code divergence were each
measured and ruled out. The two surviving explanations are dataset provenance and
checkpoint provenance — the released backbone is 174.8M parameters against the paper's
stated 188M.

**If Perlmutter reproduces materially higher than 0.8591, that is the interesting result.**
It would mean the gap is environmental or data-related, and CFS `m4722` may hold the
original labeled events that the Zenodo release doesn't match. Perlmutter is where the
original training ran, so it is the cheapest available test of that hypothesis.

## Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError: fm4npp` | run from the repo root, or `export PYTHONPATH=$PWD` |
| `size mismatch for embedder.proj: [1,64] vs [1,256]` | head built as `EmbedderAdd`; you're on `main` |
| `size mismatch for weighted_avg_weights: [1] vs [12]` | combined-cache checkpoint; pass `--combine_weights` |
| `NameError: amp_enabled` during eval | on `main`; the branch fixes it |
| `downstream_dropout` AttributeError | on `main` |
| empty dataset, no error | `--split test` used for training data; use `pretrain` |
| dataloader hangs at 0% CPU mid-validation | set `num_data_workers: 0` |
| `No space left` during caching | `--combine_layers` missing at width 1536 |

## Provenance

Numbers here come from a replication audit on 2x NVIDIA GB10. The metric conventions
(pooled, `option=2` decoding) were established empirically rather than taken from the
paper's text — see the audit report for how.
