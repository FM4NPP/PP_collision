# Module 05 — The three downstream tasks

| | |
|---|---|
| **Input** | a released backbone + ~5 events from module 01 |
| **Algorithm** | frozen backbone, small trainable adapter, three different heads |
| **Output** | a checkpoint per task, and a score from the authors' released heads |
| **Visualization** | none here — module 06 does the scoring and the plots |

The foundation model is frozen for all three tasks. What changes between them is a head of
a few hundred thousand to a couple of million parameters, and what it is asked to predict.

| task | head | params (width 256) | predicts |
|---|---|---|---|
| track finding | `MambaAttentionHead` | 2,203,918 | which spacepoints belong to the same track |
| particle ID | `AttentionHead` | 399,249 | one of 5 particle classes, per point |
| noise ID | `AttentionHead` | 398,478 | signal vs noise, per point |

## Run it

```bash
source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
cd tutorials/05_downstream_training

# score a head the authors trained on 70k events — this is a real number
python run_task.py --task tracking --mode eval \
    --checkpoint $FM4NPP_WORK/heads/d9_m1_k30_p20_nerf_tracking_head_d70000_2_seed42_checkpoint.pth

# run the training loop yourself — this is not a real number
python run_task.py --task pid --mode train --n_events 5
python run_task.py --task nid --mode train --n_events 5
```

`--dry_run` prints the commands without executing them, which is the quickest way to see
what the wrapper is actually doing.

## Why two modes, and why only one of them means anything

`--mode eval` on a released head is a **measurement**. Five events is a small sample, so
expect noise, but if your environment is broken the number collapses rather than wobbles.
That makes it a good check: on the paper's m6 head, a correct setup scores ~0.95 ARI and a
setup with the wrong Mamba kernels scores **0.054**. There is no ambiguous middle.

`--mode train` is a **demonstration**. Five events hold a few thousand spacepoints; the
tracking adapter alone has 2.2M parameters. The loss will move and the loop will complete,
and the resulting metric means nothing at all. The script prints that warning next to any
number it produces this way, because a plausible-looking number from an under-trained model
is exactly the kind of thing that gets quoted later by accident.

## The trap to avoid

`--eventnumber` limits the **training** split only. The test loader reads every event in
`data_root_test`, and it constructs with `drop_last=True`.

So on a 5-event test directory, `--eval_batch_size 8` yields **zero batches**. Then:

```
validation loss = np.mean([])  ->  nan
nan < best_loss - min_delta    ->  False, forever
                               ->  no checkpoint is ever written
                               ->  eval fails: "Checkpoint loading failed"
```

The error surfaces at the *evaluation* step, names a checkpoint path, and says nothing
about batch sizes — so it reads as a missing file rather than an empty loader. `run_task.py`
pins `--eval_batch_size 1` for this reason. If you invoke the training scripts directly,
pin it yourself.

Two smaller ones in the same family:

- `--eventnumber`, `--run_num` and `--seed` are all baked into the checkpoint filename.
  Train and evaluate with identical values or eval looks for a file that does not exist.
- The training set is built from `data_root`, **not** `data_root_train` — that key is dead.
  In `mamba_tracking.yaml`, `data_root` points at the *unlabeled pretraining* corpus, so a
  config that looks correctly repointed can still train on the wrong data. Run
  `scripts/repoint_config.py` rather than editing by hand.

## What PID and NID needed before they would run at all

Both tasks were unrunnable from this repository until [B30–B33], and the reasons are worth
knowing because they are the same class of problem twice over:

- Their configs were believed lost, and reconstructions were shipped in their place. The
  originals were in the authors' tree the whole time.
- `mid_target` was read unconditionally, but the dataset omits it and `prepare_data.py`
  never writes it — a `KeyError` on any dataset you can actually build.
- The evaluation script overrode the test-data path and the checkpoint directory with the
  authors' own machine paths, with no way to override either.
- The head could not be reconstructed. The public rewrite added FFN blocks and swapped the
  embedder, building 677,905 parameters where the released head is 399,249 — and the
  embedder class it needed had been commented out. Neither difference touches a name or a
  shape that `load_state_dict` inspects.

The last one is the general lesson of this whole repository: **structural checks cannot see
a semantic divergence.** Names matched, shapes matched, and the model was still wrong.
