# Module 06 — Scoring, and which number to quote

| | |
|---|---|
| **Input** | a trained head from module 05, or a released one |
| **Algorithm** | inference over held-out events → per-point CSV → ARI |
| **Output** | `per_point.csv`, and the two ARI numbers it yields |
| **Visualization** | per-event ARI distribution; predicted vs true cluster counts |

Getting a number out is easy. Getting the number that is comparable to the paper takes two
decisions, and both have a wrong answer that looks reasonable.

## Run it

```bash
source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
cd tutorials/06_downstream_eval

python ../../train/downstream/eval_track_finding.py \
    --yaml_config ../../scripts/configs/mamba_tracking.yaml --config d9_m1_k30_p20 \
    --run_num tut --seed 42 --eventnumber 5 --eval_batch_size 1 \
    --root_dir $FM4NPP_WORK/tutorial_evals/ --save_csv \
    --csv_output_path $FM4NPP_WORK/tutorial_evals/per_point.csv

python ../../train/downstream/pooled_ari.py $FM4NPP_WORK/tutorial_evals/per_point.csv
python score_report.py $FM4NPP_WORK/tutorial_evals/per_point.csv --out figures/
```

For a head trained from a `--combine_layers` cache, add `--combine_weights` with the vector
from that cache's `cache_meta.json`. Without it the head is rebuilt at the wrong shape and
the load fails with `size mismatch for weighted_avg_weights: [1] vs [12]`.

## Decision 1 — which decoding

`assign_points_to_masks` offers two ways to turn mask logits into a clustering, and they do
not differ by a little:

| decoding | per-event ARI | clusters found | true clusters |
|---|---|---|---|
| `option=1` — mask probability alone | 0.6598 | 88,584 | 30,500 |
| `option=2` — mask × class probability | **0.9465** | 31,022 | 30,500 |

Measured on 2,000 held-out events with the paper's m6. Option 1 over-segments by 2.9×,
which is a property of scoring on mask probability with no class gate, not a property of
the model. **Use option 2.** The evaluation script already does.

## Decision 2 — per-event mean, or pooled

The trainer scores each event separately and averages. You can also pool: relabel every
track so ids are unique across events, then compute one ARI over all points at once.
Pooling is harder, because a single global clustering must also keep tracks from *different*
events apart.

| | m6, 2,000 events |
|---|---|
| per-event mean | **0.9465** |
| pooled | 0.9213 |
| paper, Table 2 | 0.9448 |

**Quote the per-event mean.** The paper's prose says *"All metrics are computed over the
entire test set rather than averaged per event"*, which reads like an instruction to pool —
and an earlier version of this tutorial said so. The authors' own training logs settle it
the other way: the mean over their ten seeds of the per-event ARI at the best-validation
epoch is **0.9442**, against the 0.9448 in Table 2. Pooled is 0.023 away. That sentence
most likely describes the efficiency and purity metrics.

When prose and logs disagree, believe the logs.

`pooled_ari.py` prints both. Its per-event column reproduces the trainer's own `Avg_ARI` to
five decimals, which is what lets you trust the pooled column printed beside it.

## What a correct setup looks like

Scoring the authors' released m6 head on held-out events:

```
2,000 events, 1,649,781 points
  option=2   per-event 0.9465 | pooled 0.9213 | 31,022 clusters (30,500 true)
```

against 0.9448 in the paper. Over-segmentation is 1.7%.

If you get something in the neighbourhood of **0.05**, your Mamba kernels are wrong — see
module 01 and run `scripts/check_kernel_equivalence.py`. That failure is unmistakable: the
classifier collapses to about 2 clusters per event where the truth is 15. There is no
gradual degradation between 0.05 and 0.95, which is worth knowing because it means this
check is binary and therefore useful.

## Reading the per-point CSV yourself

One row per spacepoint, with `batch_idx` identifying the event:

| column | meaning |
|---|---|
| `batch_idx` | event index within the run |
| `seg_target` | true track id, 0 = noise |
| `pred_assignment` | predicted track id, option 2 |
| `E, x, y, z` | the spacepoint |
| `confidence` | max mask probability for that point |

That is enough to compute efficiency and purity, plot the per-event ARI distribution, or
find the events the model fails on. `score_report.py` does the first two;
`calculate_tracking_eff_purity.py` in `train/downstream/` does efficiency and purity, though
it needs real `reg_target` values, which the Zenodo release does not carry.
