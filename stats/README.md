# Statistics files

These are loaded **unconditionally** by the downstream code. Without them nothing runs:

| file | loaded by | purpose |
|---|---|---|
| `bin_edges_v3_nbins_8_8_6.pkl` | `fm4npp/datasets/voxelizer.py:43` | voxelizer bin edges (`bin_version='v3'`, `n_bins=(8,8,6)`) |
| `loss_bin_pp.pkl` | `train/downstream/track_finding_trainer.py` | loss binning |
| `loss_weight_pp.pkl` | `train/downstream/track_finding_trainer.py` | loss weights |

Point `stat_dir` in your config at this directory.

The other files (`bin_edges_v1*.pkl`, `bin_edges_v2*.pkl`, `exp_dict_v1.pkl`) are earlier
binning variants, included for the ablations that reference them. They are not used by the
default configuration.

## Why this matters

The voxelizer falls back to *recomputing* bin edges from whatever dataset you hand it if the
file is absent (`voxelizer.py:46-49`). That silently produces a different tokenization from
the one the released checkpoints were pretrained with — no error, just wrong results. The two
loss pickles have no fallback at all and raise `FileNotFoundError`.

These files were previously excluded from the repository by the blanket `*.pkl` rule in
`.gitignore`; that rule now carries an explicit exception for this directory.

## Note

These are Python pickles and execute code on load. They originate from the FM4NPP
collaboration's own preprocessing.

## Not included

`save_means_gross.pkl` (79 MB) is part of the same statistics bundle but is not referenced
anywhere in this codebase, so it is omitted rather than committed as a large binary.
