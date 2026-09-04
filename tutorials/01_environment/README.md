# Module 01 — Environment

| | |
|---|---|
| **Input** | a Perlmutter account (or any machine with one CUDA GPU) |
| **Algorithm** | `uv` virtualenv · range-request fetch from Zenodo · `.npz` → RaggedMmap |
| **Output** | a working env, ~1 GB of labeled data, two checkpoints |
| **Check** | `verify_install.py` — 13 checks, PASS/FAIL table |

Non-trivial for two reasons: the dataset is published as a **118.5 GB** archive when the
tutorials need under 1 GB of it, and one missing pickle causes **silent** numerical
corruption rather than an error.

## Run it

All three steps on a **login node** — they need network, and compute nodes have none.

```bash
git clone https://github.com/colpark/FM4NPP_tutorials_090426
cd FM4NPP_tutorials_090426

bash 01_environment/setup_perlmutter.sh     # ~10 min (torch is ~2.5 GB)
source common/paths.sh
source $FM4NPP_WORK/.venv/bin/activate
bash 01_environment/fetch_data.sh           # ~15 min (1 GB data + 2.2 GB checkpoints)
python 01_environment/verify_install.py     # ~1 min
```

Every new shell afterwards needs only:

```bash
source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
```

## What gets built

```
$SCRATCH/fm4npp_tutorial/
├── .venv/                     python 3.11 + torch + fm4npp deps
├── PP_collision/              the official repo, branch downstream-reproducibility
├── zenodo/labeled/{train,validation,test}/   raw .npz
├── data/
│   ├── pretrain_root/         features_pretrain/ + features_test/   → module 03
│   └── eval_root/             features_test/                        → module 04
└── checkpoints/
    ├── pp_nerf_m1_k30.ckpt    paper m3, 5.3M, 59 MB
    └── pp_nerf_m5_k30.ckpt    paper m6, 175M, 2.1 GB
```

## Four things worth understanding

**You do need `mamba-ssm`, and this module used to say otherwise.** The reasoning looked
sound: every released checkpoint is Mamba2, `fm4npp/models/mamba2.py` guards every
fused-kernel import, and the build takes 20+ minutes and can fail. All true. What it missed
is that the guards fall through to an implementation that was not the same model — an
ungated RMSNorm applied to the SSM input, and an EMA scan that discarded `B` and `C`, so the
state was rank 1 where Mamba2's is rank `d_state`.

Nothing detected it. `load_state_dict(strict=True)` passed, because no shape changes. An
adapter trained on top learned to read the corrupted features and reached a believable
score. The only symptom was a final number about 0.09 ARI below the paper — which reads as
a research result, not a bug. It took loading a head trained elsewhere, which scored 0.054
against 0.948 in its own log, to find it.

Both errors are fixed and the fallback now tracks the kernels. It is still far slower and
still unvalidated on any given machine, so it refuses to run unless you ask:

```bash
pip install mamba-ssm causal-conv1d          # login node; these compile
python scripts/check_kernel_equivalence.py   # PASS, or SKIP if the kernels are absent
```

If they cannot be built on your machine, `export FM4NPP_ALLOW_FALLBACK=1` and read the
warning it prints. The lesson generalises past this repo: a numerical fallback that cannot
be compared against the thing it replaces is a silent-failure generator.

**You don't need the 118.5 GB archive.** `fetch_labeled_data.py` reads the zip's central
directory over HTTP range requests and pulls only the labeled members — 0.97 GB, about
12 minutes, versus roughly 26 hours for the whole thing. The rest is the unlabeled
pretraining corpus, needed only if you intend to pretrain from scratch for real.

**`prepare_data.py` runs twice into the same directory.** Pretraining wants one `data_root`
containing *both* splits, because `get_data_loader()` hardcodes `split='pretrain'` for
training and `split='test'` for validation. The split is a **suffix on each array directory**
(`features_pretrain/`, `features_test/`), not a subdirectory. Confusingly, `pretrain` here
means the labeled *training* split, not the unlabeled corpus.

**`stat_dir` must point at the repo's `stats/`.** This is the one that will cost you a week
if you get it wrong. If `bin_edges_v3_nbins_8_8_6.pkl` is missing, the `Voxelizer` does not
raise — it **recomputes bin edges from your data**. The tokenization then no longer matches
what the checkpoints were trained on, everything still runs, and every number is wrong. The
`VERIFY: BINS` check exists solely to catch this.

## Limits of the fetched data

`fetch_data.sh` prepares 20,000 training and 2,000 validation events — enough for every
tutorial here. Full reproduction uses 70,000. Raise `--end` in `fetch_data.sh` if you want
more; the Zenodo train split holds 70k.

`reg_target` is **synthesized**. Zenodo publishes a boolean noise tag, not track kinematics,
so `prepare_data.py` fabricates a plausible `reg_target` to keep the loaders working. Noise
labels are faithful. Track-kinematics regression and efficiency/purity metrics are **not
reproducible** from public data.

## If something fails

| symptom | cause |
|---|---|
| `module load python` fails | NERSC module names drift; `module avail python` and substitute |
| `git+https://...` install hangs | you're on a compute node — go back to a login node |
| `BINS` fails | `FM4NPP_STATS` isn't the repo's `stats/`; re-`source common/paths.sh` |
| `BRANCH` fails | you cloned `main`; its downstream code does not run |
| `CKPT6` fails on shapes | you downloaded the wrong file — `pp_nerf_m5` is the paper's m6 |
| `No module named 'plotly'` | it is a *hard* import in the data path, not optional |
