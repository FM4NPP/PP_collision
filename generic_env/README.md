# generic_env — the downstream tasks, without the rest of the repository

Released checkpoint in, trained head and a score out. No pretraining, no cluster, no large
download, no CUDA build.

```bash
uv sync                      # ~12 packages, all wheels, no compilation
source env.sh
bash get_checkpoint.sh       # 59 MB backbone from HuggingFace
python selftest.py           # ~70 seconds, end to end
```

If that prints `PASS`, everything below works.

| | |
|---|---|
| **Data** | 40 events committed in `fixture/` (1.5 MB). Nothing to download. |
| **Backbone** | `pp_nerf_m1_k30.ckpt`, 59 MB — the repo's `m1`, which is the **paper's m3** |
| **Tasks** | track finding, particle ID, noise ID |
| **Runtime** | ~70 s for tracking; a few minutes for all three |
| **Hardware** | any machine with PyTorch; a GPU is optional |

```bash
python run.py --task tracking --stage both
python run.py --task pid --stage train
python run.py --task nid --stage eval
python selftest.py --all          # all three, with fixture verification
```

## Read this before quoting any number

**This environment runs a fallback implementation by default, and that path has never been
compared against the real kernels on any machine.**

`fm4npp/models/mamba2.py` uses the fused Triton kernels from `mamba-ssm` when they are
installed, and its own pure-PyTorch implementation when they are not. Those two were not
equivalent. The fallback applied an ungated RMSNorm to the wrong tensor and replaced the
state-space scan with an exponential moving average that discarded `B` and `C` — so the state
was rank 1 where Mamba2's is rank `d_state`. Released checkpoints loaded into it cleanly with
`strict=True`, scored about 0.09 ARI below the paper, and nothing said which path had run. A
head the authors had trained scored **0.054** here against **0.948** in its own log.

Both errors are fixed. But "fixed on a machine that cannot run the alternative" is not the
same as verified, and this environment deliberately opts into that path so the install stays
simple. Every `run.py` invocation prints which implementation it used. Treat what comes out
as evidence the pipeline works, not as a measurement.

To upgrade, on x86 + CUDA:

```bash
bash ../tutorials/01_environment/install_kernels.sh   # prebuilt wheels, no compile
python ../scripts/check_kernel_equivalence.py         # PASS means the two agree
```

Do **not** run `pip install mamba-ssm causal-conv1d`. Neither package has a wheel on PyPI, so
that compiles CUDA extensions for 20–60 minutes and fails without `--no-build-isolation`
because `setup.py` imports torch before torch exists.

## Twenty events is not a measurement either

The fixture is sized to make the code path testable, not to produce a result. Tracking on 40
events after 4 epochs returns an ARI around 0.1; the same model trained properly returns
0.95. A number from here tells you the pipeline runs.

For real data, `../scripts/fetch_labeled_data.py` pulls the labeled splits over HTTP range
requests — about 1 GB and twelve minutes, rather than the 118.5 GB the full archive would be.
The fixture cannot be re-fetched that way: the fetcher has no granularity below one member,
and the smallest useful member is 78 MB. It was derived once from converted data and
committed.

## What the layout is protecting you from

Three failure modes in this codebase are silent, and this folder is arranged so they cannot
happen rather than documented so you can avoid them.

**Paths.** Every path in `configs/` is relative to this directory, so nothing needs
repointing. `../scripts/repoint_config.py` rewrites four absolute literals, but it misses
`stat_dir` in the point-classification configs, and its leftover-detector greps only for
`/global/cfs|/mldata|/pscratch` — so a surviving `/home/shuhang` path is never reported and
the tool prints "written" as though it had succeeded.

**The backbone path.** All four entry points carry a `model2ckpt` table of the original
author's cluster paths, assigned *after* the config is read, so it silently overrode whatever
the config said and could not be changed from the command line. `run.py` passes
`--pretrained_ckpt` explicitly.

**The bin edges.** If `bin_edges_v3_nbins_8_8_6.pkl` is absent, `voxelizer.py` does not fail —
it recomputes bin edges from whatever data it is handed, writes a new pickle, and proceeds
with a tokenization different from the one the checkpoints were pretrained with. The pickle
travels inside `fixture/` and `selftest.py` verifies its hash before anything runs.

And one that is merely confusing rather than silent: **the checkpoint names do not match the
paper.** The repository's configs and the paper both run `m1`–`m6` and they do not agree.
`pp_nerf_m1_k30.ckpt` is the paper's **m3** (width 256, 5.3M). Downloading "m1" expecting the
paper's m1 gets you a model 16× larger.

## Files

```
env.sh              PYTHONPATH, backbone location, FM4NPP_ALLOW_FALLBACK=1
pyproject.toml      the traced dependency set, with notes on what was excluded and why
get_checkpoint.sh   fetch the 59 MB backbone
run.py              one entry point for the three tasks; prints the kernel banner
selftest.py         fixture verification + end-to-end run
make_fixture.py     how fixture/ was built, for audit; you do not need to run it
configs/            relative-path configs sized for the fixture
fixture/            40 events, the three stat pickles, and MANIFEST.json
```

`pyproject.toml` lists 12 packages against the repository's larger `requirements.txt`, having
traced what the downstream entry points actually import. Notably absent: **pyyaml** — the
config loader uses `ruamel.yaml`, and nothing on this path does `import yaml`.
