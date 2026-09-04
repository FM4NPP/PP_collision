# Module 03 — Pretraining on Perlmutter

| | |
|---|---|
| **Input** | `$FM4NPP_PRETRAIN_ROOT` from module 01 |
| **Algorithm** | masked k-nearest-neighbour regression, Mamba2 backbone, 12 layers |
| **Output** | `ckpt.tar` + a per-step loss CSV |
| **Visualization** | loss curve; optimal-LR-vs-width transfer plot |

You will run a real pretraining job — small, but the same code path as the full one. Ten
minutes on the debug queue, no allocation burned waiting in `regular`.

## Run it

```bash
source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
cd 03_pretraining

# 1. Put your project account in the sbatch script (the only field left blank)
#    Find it with:  iris
$EDITOR perlmutter_debug.sbatch        # line 15:  #SBATCH -A <your_account>

# 2. Check everything builds, on the login node, no GPU time
python configs/render_config.py configs/tutorial_m3.yaml -o $SCRATCH/m3.yaml
python pretrain_m3.py --yaml_config $SCRATCH/m3.yaml --config tutorial_m3 --dry_run

# 3. Submit
sbatch perlmutter_debug.sbatch

# 4. Plot
python plot_loss.py $FM4NPP_RUNS/tutorial_m3/debug0/config_tutorial_m3_run_debug0.csv
```

`--dry_run` builds the model, dataset and optimizer and then exits. Use it every time
before submitting — a config typo costs you a queue wait otherwise.

## What the model is doing

The pretraining task is **self-supervised**: from each spacepoint's representation, predict
the 3D positions of its `klen=30` nearest neighbours. Loss is masked MSE over those
30 × 3 = 90 numbers per point.

No labels involved. The model learns local detector geometry — which hits belong to the same
trajectory — because that is what makes neighbour positions predictable. That is precisely
the structure the downstream track-finding head then exploits.

Paper m3: width 256, `d_state` 16, 12 Mamba2 layers. The released checkpoint holds
**4,923,386** parameters (4,907,002 trainable — the NeRF Fourier projection is a fixed
random matrix). The paper states 5.3M; see the top-level README on that discrepancy.
Our config reproduces the checkpoint's architecture exactly, verified by a `strict=True`
load.

## Six changes from the repo's script

`pretrain_m3.py` is vendored from `train/pretrain/nppmamba/train_multi_gpu_mamba1.py` and
each change is tagged `# [TUT-n]` inline, so `diff` tells the whole story.

| tag | change | why it matters |
|---|---|---|
| **TUT-1** | `Mamba1GPT` → `MambaGPT` | The original builds **Mamba1**, needing compiled `mamba_ssm` — and every released checkpoint is **Mamba2**. One line removes a 20-minute CUDA build *and* makes your output comparable to `pp_nerf_m1_k30.ckpt`. |
| **TUT-2** | call `init_process_group` | The original never does, so `dist.is_initialized()` is always False: no DDP, no `DistributedSampler`, all four ranks read the same data and write the same checkpoint. You get 1 GPU of 4. |
| **TUT-3** | `--mup / --no-mup` | Module 02's switch. |
| **TUT-4** | save/restore scheduler state | The original resumes with the scheduler at step 0 — redoing warmup, then following the wrong cosine phase. `best_loss` isn't restored either, so the first post-resume validation always overwrites `ckpt_best.tar`. |
| **TUT-5** | fix `os.makedirs` | `training_checkpoints/` was created only inside `if not os.path.isdir(exp_dir)`. Existing run dir + missing subdir = first save crashes. |
| **TUT-6** | `--max_steps`, `--dry_run` | Stop cleanly inside 10 minutes; validate a config without GPU time. |

## Why the SLURM script looks like this

```bash
#SBATCH -C gpu        # the repo's scripts OMIT this — without it you don't get GPUs
#SBATCH -q debug      # 10 min cap, short wait
#SBATCH -N 1 --ntasks-per-node=4 --gpus-per-node=4
```

and the `srun` block is **live**, not commented out:

```bash
srun -l bash -c '
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
  python pretrain_m3.py ...'
```

Those three variables are what let `init_process_group` find its peers. In the official
repo they live inside a commented-out block, with a bare `$cmd` running underneath — which
is why that script quietly uses one GPU.

**Time budget for 10 minutes:** dataset scan ≈1 min (the loader loops over every event to
read its length before step 0), then ~4 min for 200 steps at width 256 on 4×A100. The config
uses `limit_size: 2000` to keep the scan short. Raising it is the first thing to change for a
longer run.

## The μP payoff experiment

```bash
python lr_transfer_sweep.py --widths 128,256,512 --steps 150
```

30 short runs (~25 min on an A100) producing `lr_transfer.png`: training loss vs learning
rate, one curve per width, μP on and off side by side.

**What to look for.** With μP the minima of the three curves sit at roughly the same LR — so
you tune once at width 128 and reuse it at 1536. Without μP they drift, and every model size
needs its own sweep. The script prints the optimal LR per width and the spread between them;
a smaller spread in the μP arm is the transfer property working.

One honest caveat: FM4NPP ties `d_state` to width (`Nx = Nu/16`), while textbook μP holds
the state dimension fixed. This measures transfer for *this* parameterization along *this*
ladder — the practically useful question — not the textbook claim.

## Expectations, plainly

200 steps produces a **checkpoint, not a useful model**. Real pretraining is 50,000 steps on
12M events over 24–48 h on 4×A100. What you should see is loss falling from ~1.0 to
~0.3–0.5 and a `ckpt.tar` on disk. If loss is flat or NaN, something is wrong — check
`stat_dir` first.

**These Perlmutter scripts are untested.** They were written from NERSC's documented
conventions and the path fossils left in the repo. If `module load python` fails, run
`module avail python`, substitute, and please open an issue with the working name.

## Troubleshooting

| symptom | cause |
|---|---|
| `sbatch: error: Invalid account` | `-A` is still blank |
| job runs but uses 1 GPU | `srun` line edited out, or `WORLD_SIZE` unset |
| `FileNotFoundError: loss_bin_pp.pkl` | `stat_dir` wrong — re-`source common/paths.sh` |
| loss is NaN from step 1 | usually `max_lr` too high with `--no-mup` at large width |
| `No such file: features_test` | you ran `prepare_data.py` once; it needs to run twice (both splits, same `--out`) |
| job hangs at startup | dataset scan on too many events — lower `limit_size` |
