# FM4NPP Tutorials

Hands-on modules for the FM4NPP foundation model for nuclear and particle physics —
pretraining, μ-parameterization, and what the learned representations actually contain.

Written for physics PhD students and postdocs. You need to know Python and roughly what a
transformer does. You do not need to know Mamba, state-space models, or μP.

Everything runs on **NERSC Perlmutter**, and every module also runs on any single CUDA GPU.

| module | you give it | it runs | you get out |
|---|---|---|---|
| [01 Environment](01_environment/) | a Perlmutter account | `uv` venv, Zenodo fetch, HF download | a working env, ~1 GB of data, 2 checkpoints |
| [02 μ-parameterization](02_mu_parameterization/) | the FM4NPP source | a guided code tour + live experiments | where μP is, why it currently does nothing, a working on/off switch |
| [03 Pretraining](03_pretraining/) | the prepared dataset | 200 steps of m3 pretraining on the debug queue | `ckpt.tar`, a loss curve, an LR-transfer plot |
| [04 Features & t-SNE](04_features_tsne/) | the m6 checkpoint + 10 events | frozen-backbone forward pass | `features.npz`, t-SNE colored by true track |
| [05 Downstream training](05_downstream_training/) | a released backbone + ~5 events | the three task heads, trained and scored | a checkpoint per task, and real numbers from released heads |
| [06 Downstream evaluation](06_downstream_eval/) | a per-point CSV | ARI four ways, efficiency/purity inputs | the number that is comparable to the paper, and why |

Each module is standalone and states its inputs and outputs at the top. Later modules use
earlier outputs but never earlier *state* — you can start at 04 if you have a checkpoint.

## Start here

```bash
git clone -b downstream-reproducibility https://github.com/FM4NPP/PP_collision
cd PP_collision/tutorials
cat 01_environment/README.md
```

These modules live inside the code repository they teach, so the commands in them run
against the checkout you already have, and a fix to the code and a fix to its explanation
land in the same commit.

## ⚠️ Model naming: the repo's `m1` is NOT the paper's `m1`

This trips up everyone, including us. The code and the paper use the same letters for
different models. **Throughout these tutorials we use the paper's names**, with the repo
config in parentheses.

| paper | width | d_state | params (paper) | params (checkpoint) | repo config | HuggingFace checkpoint |
|---|---|---|---|---|---|---|
| m1 | 64 | — | 0.34M | *not released* | `d9_m64_k30_p20` | *not released* |
| m2 | 128 | — | 1.3M | *not released* | `d9_m128_k30_p20` | *not released* |
| **m3** | **256** | **16** | 5.3M | **4.92M** | `d9_m1_k30_p20` | `pp_nerf_m1_k30.ckpt` |
| m4 | 512 | 32 | 21M | — | `d9_m3_k30_p20` | `pp_nerf_m3_k30.ckpt` |
| m5 | 1024 | 64 | 84M | — | `d9_m4_k30_p20` | `pp_nerf_m4_k30.ckpt` |
| **m6** | **1536** | **96** | 188M | **174.69M** | `d9_m5_k30_p20` | `pp_nerf_m5_k30.ckpt` |

Download "m1" from HuggingFace expecting the paper's m1 and you get a model **16× larger**
than you wanted. See [`common/fm4npp_naming.md`](common/fm4npp_naming.md).

**The checkpoints hold ~7% fewer parameters than the paper states**, at every size we can
check: m3 is 4.92M against 5.3M, m6 is 174.69M against 188M — a ratio of 0.929 and 0.929.

This is now resolved, and it is a table artifact rather than a missing layer. Both stated
figures are what you get by counting **13** layers instead of 12: m3 is 4.92M at 12 layers
and 5.33M at 13 against a stated 5.3M; m6 is 174.78M and 189.33M against a stated 188M.
Both within 0.7%. That the released files have 12 layers is confirmed independently by the
authors' own downstream heads, whose `weighted_avg_weights` is a 12-vector — a 13-layer
backbone would have handed them 13. So the checkpoints are complete and Table 1 counts one
layer too many. Quote the checkpoint figures.

Of a checkpoint's parameters, a small number are **frozen**: the NeRF embedder's Fourier
projection (`embedder.embed.projection`, 63×256) is a fixed random matrix that never
receives a gradient. For m3 that is 16,384 of 4,923,386 — so 4,907,002 are trainable.
Module 04 shows this random projection is doing more representational work than you would
guess.

## What these tutorials are built on

- **Code**: [`FM4NPP/PP_collision`](https://github.com/FM4NPP/PP_collision), branch
  `downstream-reproducibility`. That branch carries fixes without which the downstream code
  does not run; module 01 clones it specifically.
- **Data**: the Zenodo TPC pp-collision release. The archive is 118.5 GB; these tutorials
  need **under 1 GB** of it, fetched by HTTP range request.
- **Checkpoints**: the `FM4NPP/PP_collision` HuggingFace repo.

## Honest notes

We would rather you hit a documented limitation than an undocumented surprise.

- **The μP that produced the released checkpoints is not in the public repo.** The shipped
  pretraining script says so in its own docstring. What remains is μP-shaped code in the
  *downstream* trainers that is inert for two independent reasons. Module 02 shows you both,
  executably, and supplies a working reconstruction.
- **The Perlmutter scripts are untested.** They were written from NERSC's documented
  conventions and the path fossils left in the repo. They have not been run on Perlmutter.
  If `module load` names have drifted, that is the first thing to check — please open an
  issue with the fix.
- **`reg_target` is synthesized.** Zenodo ships a boolean noise tag, not track kinematics.
  `prepare_data.py` fabricates a plausible `reg_target` so the loaders work. Track finding
  and noise tagging are faithful; **track-kinematics regression is not**, and efficiency /
  purity numbers cannot be reproduced from public data.
- **Pretraining as shipped is single-GPU.** `init_process_group` is never called, so SLURM's
  `--ntasks-per-node=4` is silently ignored. Module 03 fixes this in its vendored copy.
