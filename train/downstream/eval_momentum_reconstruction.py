#!/usr/bin/env python3
"""
OBSOLETE (2026-11-02): momentum is to be calculated in ACTS, not in this package. Kept, not
deleted, so the momentum task can be picked up again. Running it raises a FutureWarning.

Evaluate momentum reconstruction (MomentumHead) after a trained track-finding head, the way
eval_track_finding.py evaluates that head. Run from train/downstream:

    python eval_momentum_reconstruction.py --root_dir out/ [--checkpoint <head.pth>] [--no-csv]

Backbone: --pretrained_ckpt, else pretrained_ckpt in --config. Track head: --checkpoint, else
{track_config}_nerf_tracking_head_d{eventnumber}_{run_num}_seed{seed}[_nopretrain]_checkpoint.pth
in --checkpoint_dir (default: the config's checkpoint_dir). Field: --bz, else YAML momentum_bz,
else 1.4.

Outputs in --root_dir, named after {track_config}_eval_momentum_head_d...log: the .log metrics,
_per_point_data.csv (unless --no-csv), _metrics.json, _results.npz (per fitted track) and
_momentum_head.pth (for eval_vertex_reconstruction.py --momentum_ckpt).
"""
import os
import sys
import warnings
import argparse
import gc
import json

import numpy as np
import torch

# make sure the FM4NPP modules can be imported
sys.path.append('../..')

from fm4npp.utils import YParams
from momentum_reconstruction_trainer import DownstreamTrainer

DEFAULT_YAML = "../../scripts/configs/mamba_tracking.yaml"
DEFAULT_CONFIG = "mamba_5m_downstream"
DEFAULT_TRACK_CONFIG = "d9_m1_k30_p20"    # train_track_finding.py's name for the same backbone


def main():
    warnings.warn("eval_momentum_reconstruction.py is obsolete: momentum is to be calculated in ACTS (kept for a possible "
                  "return, not maintained)", FutureWarning, stacklevel=2)
    parser = argparse.ArgumentParser(
        description="Evaluation script for the momentum reconstruction downstream task"
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        default=DEFAULT_YAML,
        help=f"Path to the YAML config file (default: {DEFAULT_YAML})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Config name inside the YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--track_config",
        type=str,
        default=DEFAULT_TRACK_CONFIG,
        help="--config the track head was trained with; only names the derived checkpoint "
             f"(default: {DEFAULT_TRACK_CONFIG})",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Frozen backbone checkpoint (default: pretrained_ckpt from the config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to the trained track-head checkpoint (optional; overrides default)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Where track-head checkpoints live (default: checkpoint_dir from the config, "
             "where train_track_finding.py writes them)",
    )
    parser.add_argument(
        "--run_num",
        type=str,
        default="0",
        help="Run number / seed identifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used during track-head training (selects the checkpoint)",
    )
    parser.add_argument(
        "--no_seed_in_ckpt",
        action="store_true",
        default=False,
        help="Use a checkpoint name without the _seed<N> suffix (pre-B8 checkpoints)",
    )
    parser.add_argument(
        "--save_csv",
        dest="save_csv",
        action="store_true",
        default=True,
        help="Save the per-point CSV (default: enabled), followed by the fitted momentum of each "
             "hit's track (pred_* columns)",
    )
    parser.add_argument(
        "--no-csv",
        dest="save_csv",
        action="store_false",
        help="Disable per-point CSV output",
    )
    parser.add_argument(
        "--csv_output_path",
        type=str,
        default=None,
        help="Path for the per-point CSV (default: derived from the log file name)",
    )
    parser.add_argument(
        "--data_root_test",
        type=str,
        default=None,
        help="Evaluation data root (default: data_root_test from the config)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory to store evaluation outputs",
    )
    parser.add_argument(
        "--eventnumber",
        type=int,
        default=70000,
        help="Number of events (samples) to evaluate; also names the derived checkpoint",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--usepretrain",
        dest="usepretrain",
        action="store_true",
        help="enable using the pretrained model (default)",
    )
    parser.add_argument(
        "--no-pretrain",
        dest="usepretrain",
        action="store_false",
        help="disable pretrained model (track head trained with --no-pretrain)",
    )
    parser.set_defaults(usepretrain=True)
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument(
        "--max_eval_events",
        type=int,
        default=20000,
        help="Evaluate at most this many events",
    )
    parser.add_argument(
        "--combine_weights",
        type=str,
        default=None,
        help="Comma-separated POST-softmax layer weights. Required for a track head that was "
             "trained against a --combine_layers cache (built with num_feature_layers=1). Read "
             "the vector from the cache's cache_meta.json.",
    )
    parser.add_argument(
        "--bz",
        type=float,
        default=None,
        help="Solenoid field, Tesla (default: YAML momentum_bz, else 1.4)",
    )
    args = parser.parse_args()

    # Prepare hyperparameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    elif not getattr(params, "pretrained_ckpt", None):
        sys.exit(f"config {args.config!r} has no pretrained_ckpt; add one to the YAML or pass --pretrained_ckpt")
    params.limit_data = True
    params.limit_size = args.eventnumber
    params.batch_size = args.eval_batch_size
    params.valid_batch_size = args.eval_batch_size
    # [FIX B8] mirror the seed suffix used by train_track_finding.py
    seed_suffix = "" if args.no_seed_in_ckpt else f"_seed{args.seed}"
    pretrain_suffix = "" if args.usepretrain else "_nopretrain"
    checkpoint_name = (f"{args.track_config}_nerf_tracking_head_d{params.limit_size}_{args.run_num}"
                       f"{seed_suffix}{pretrain_suffix}_checkpoint.pth")
    params.log_file_name = (f"{args.track_config}_eval_momentum_head_d{params.limit_size}_{args.run_num}"
                            f"{seed_suffix}{pretrain_suffix}.log")
    if args.data_root_test:
        params.data_root_test = args.data_root_test
    checkpoint_path = (args.checkpoint if args.checkpoint
                       else os.path.join(args.checkpoint_dir or params.checkpoint_dir, checkpoint_name))
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"track-head checkpoint not found: {checkpoint_path}\n"
                 f"(pass --checkpoint, or check --track_config/--run_num/--seed/--eventnumber "
                 f"against the training run)")

    params.num_embedder_layers = 0
    params.return_reg_test = True        # val loader then yields reg as a 4th element
    params.return_dict = False           # the trainer unpacks tuples, like track_finding_trainer
    if args.bz is not None:
        params.momentum_bz = args.bz

    # Ensure output directory exists
    log_dir = args.root_dir
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, params.log_file_name)
    print(f"[momentum] backbone   : {params.pretrained_ckpt}")
    print(f"[momentum] track head : {checkpoint_path}")

    # Launch and run inference
    trainer = DownstreamTrainer(params, args)
    trainer.max_eval_events = args.max_eval_events
    if args.combine_weights:
        w = [float(x) for x in args.combine_weights.split(',')]
        if len(w) != int(params.num_layers_backbone):
            raise ValueError(f'--combine_weights has {len(w)} entries, model has '
                             f'{params.num_layers_backbone} layers')
        if abs(sum(w) - 1.0) > 1e-3:
            raise ValueError(f'--combine_weights sums to {sum(w):.6f}, expected 1.0 -- '
                             'pass the POST-softmax vector from cache_meta.json')
        trainer.cache_combined = True          # so the head is built with L=1
        trainer.eval_combine_weights = w       # so the backbone stack is collapsed the same way
        print(f'[eval] layer-combined head: {[round(x,4) for x in w]}')
    trainer.launch()
    result = trainer.inference(
        checkpoint_path=checkpoint_path,
        pretrain=args.usepretrain,
        logfile=logfile,
        save_csv=args.save_csv,
        csv_output_path=args.csv_output_path,
    )
    if trainer.world_rank == 0:
        save_results(result, trainer, params, args, checkpoint_path, logfile)
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()


def save_results(result, trainer, params, args, checkpoint_path, logfile):
    """
    Write _results.npz, _metrics.json and _momentum_head.pth next to logfile; append the metrics to
    the performance log.
    """
    npz_path = logfile.replace(".log", "_results.npz")
    np.savez(npz_path, **result["tracks"])

    ckpt_path = logfile.replace(".log", "_momentum_head.pth")
    trainer.momentum_head.save_checkpoint(ckpt_path, params=vars(params))

    summary = {
        "config": args.config,
        "yaml_config": os.path.abspath(args.yaml_config),
        "pretrained_ckpt": params.pretrained_ckpt,
        "track_ckpt": os.path.abspath(checkpoint_path),
        "momentum_config": trainer.momentum_head.config(),
        "max_eval_events": args.max_eval_events,
        "metrics": result["metrics"],
    }
    json_path = logfile.replace(".log", "_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    trainer.log_infile("[momentum] " + json.dumps(result["metrics"]))
    print(f"[momentum] wrote {logfile}\n[momentum] wrote {npz_path}\n[momentum] wrote {json_path}\n"
          f"[momentum] wrote {ckpt_path}")


if __name__ == "__main__":
    main()
