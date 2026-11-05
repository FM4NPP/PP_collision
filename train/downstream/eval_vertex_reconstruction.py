#!/usr/bin/env python3
"""
Evaluate vertex reconstruction (VertexHead) after a trained track-finding head, the way
eval_track_finding.py evaluates that head. Run from train/downstream:

    python eval_vertex_reconstruction.py --root_dir out/ [--checkpoint <head.pth>] [--no-csv]
    python eval_vertex_reconstruction.py --run_config runs/<run>.yaml [any option to override it]

A run file (runs/) holds any of the options below as YAML keys (option names without --), so a model and
site setup is one file; options on the command line override it.

Backbone: --pretrained_ckpt, else pretrained_ckpt in --config. Track head: --checkpoint, else
{track_config}_nerf_tracking_head_d{eventnumber}_{run_num}_seed{seed}[_nopretrain]_checkpoint.pth
in --checkpoint_dir (default: the config's checkpoint_dir). Momentum: --momentum_ckpt, else built
from --bz (else YAML vertex_bz, else 1.4). VertexHead: --detector (else YAML vertex_detector) gives a
detector file (scripts/acts/detectors/<name>/<name>.yaml) with the field, ACTS inputs and fit settings.

Outputs in --root_dir, named after {track_config}_eval_vertex_head_d...log: the .log metrics,
_per_point_data.csv (unless --no-csv), _metrics.json and _results.npz (per-event vertices). --no_output writes
none of these (a test run: metrics and timing are printed only).
"""
import os
import sys
import argparse
import gc
import json

import numpy as np
import torch

# make sure the FM4NPP modules can be imported
sys.path.append('../..')

from fm4npp.utils import YParams
from vertex_reconstruction_trainer import DownstreamTrainer

DEFAULT_YAML = "../../scripts/configs/mamba_tracking.yaml"
DEFAULT_CONFIG = "mamba_5m_downstream"
DEFAULT_TRACK_CONFIG = "d9_m1_k30_p20"    # train_track_finding.py's name for the same backbone


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for the vertex reconstruction downstream task"
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
        "--momentum_ckpt",
        type=str,
        default=None,
        help="MomentumHead checkpoint for the PV fit's track parameters (default: one built from --bz); "
             "experimental, as the momentum task (ACTS computes momentum)",
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
        help="Save the per-point CSV (default: enabled); vtx_x/y/z hold the reconstructed vertex, "
             "test_* columns (truth vertex, residual) are for testing only",
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
        default=None,
        help="Root directory to store evaluation outputs (required, here or in --run_config)",
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
        help="Solenoid field, Tesla (default: YAML vertex_bz, else 1.4); ignored with --momentum_ckpt",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Detector file for VertexHead.from_detector (scripts/acts/detectors/<name>/<name>.yaml): field, "
             "ACTS inputs and fit settings (default: YAML vertex_detector, else the built-in sPHENIX defaults)",
    )
    parser.add_argument(
        "--no_output",
        action="store_true",
        help="Test run: write no files (no CSV, .log, _results.npz, _metrics.json, experiment or global log "
             "directories); the metrics and timing are only printed. --root_dir is then not needed",
    )
    parser.add_argument(
        "--vertex_chunk_events",
        type=int,
        default=None,
        help="Events per VertexHead call (default: YAML vertex_chunk_events, else 32; 1 = per loader batch); "
             "the backbone and track head still run per loader batch",
    )
    parser.add_argument(
        "--vertex_device",
        type=str,
        default=None,
        help="Device for VertexHead, e.g. cpu (default: YAML vertex_device, else the backbone's device)",
    )
    parser.add_argument(
        "--run_config",
        type=str,
        default=None,
        help="YAML file of this script's options (keys = option names without --, e.g. runs/example.yaml); "
             "command-line options override it, relative paths are relative to the file",
    )
    # a run file sets the defaults; anything given on the command line still wins
    run_config = parser.parse_known_args()[0].run_config
    if run_config:
        import yaml
        with open(run_config) as f:
            run = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(run) - known)
        if unknown:
            sys.exit(f"{run_config}: unknown option(s) {unknown}; valid: {sorted(known - {'help', 'run_config'})}")
        base = os.path.dirname(os.path.abspath(run_config))
        path_keys = {"yaml_config", "pretrained_ckpt", "checkpoint", "checkpoint_dir", "momentum_ckpt",
                     "csv_output_path", "data_root_test", "root_dir", "detector", "global_log_dir"}
        # each value converted as the command line would be (e.g. run_num: 2 -> "2")
        types = {a.dest: a.type for a in parser._actions if a.type is not None}
        run = {k: types[k](v) if k in types and v is not None else v for k, v in run.items()}
        parser.set_defaults(**{k: os.path.join(base, os.path.expanduser(v)) if k in path_keys and isinstance(v, str) else v
                               for k, v in run.items()})
    args = parser.parse_args()
    if args.no_output:
        args.save_csv = False
        args.root_dir = args.root_dir or "."
    if not args.root_dir:
        parser.error("--root_dir is required (on the command line or in --run_config), unless --no_output")

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
    params.log_file_name = (f"{args.track_config}_eval_vertex_head_d{params.limit_size}_{args.run_num}"
                            f"{seed_suffix}{pretrain_suffix}.log")
    if args.data_root_test:
        params.data_root_test = args.data_root_test
    checkpoint_path = (args.checkpoint if args.checkpoint
                       else os.path.join(args.checkpoint_dir or params.checkpoint_dir, checkpoint_name))
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"track-head checkpoint not found: {checkpoint_path}\n"
                 f"(pass --checkpoint, or check --track_config/--run_num/--seed/--eventnumber "
                 f"against the training run)")
    if args.momentum_ckpt and not os.path.isfile(args.momentum_ckpt):
        sys.exit(f"momentum checkpoint not found: {args.momentum_ckpt}")

    params.num_embedder_layers = 0
    params.return_reg_test = True        # val loader then yields reg as a 4th element
    params.return_dict = False           # the trainer unpacks tuples, like track_finding_trainer
    if args.bz is not None:
        params.vertex_bz = args.bz
    if args.vertex_chunk_events is not None:
        params.vertex_chunk_events = args.vertex_chunk_events
    if args.vertex_device is not None:
        params.vertex_device = args.vertex_device
    if args.detector is not None:
        params.vertex_detector = os.path.abspath(args.detector)

    # Ensure output directory exists
    params.no_output = args.no_output
    log_dir = args.root_dir
    if not args.no_output:
        os.makedirs(log_dir, exist_ok=True)
    logfile = None if args.no_output else os.path.join(log_dir, params.log_file_name)
    print(f"[vertex] backbone   : {params.pretrained_ckpt}")
    print(f"[vertex] track head : {checkpoint_path}")

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
        momentum_ckpt=args.momentum_ckpt,
    )
    if trainer.world_rank == 0 and not args.no_output:
        save_results(result, trainer, params, args, checkpoint_path, logfile)
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()


def save_results(result, trainer, params, args, checkpoint_path, logfile):
    """
    Write _results.npz and _metrics.json next to logfile and append the metrics to the performance
    log.
    """
    npz_path = logfile.replace(".log", "_results.npz")
    np.savez(npz_path,
             vertex_estimate=result["vertex_estimate"].float().numpy(),
             vertex_target=result["vertex_target"].float().numpy(),
             fit_is_valid=result["fit_is_valid"].numpy())

    summary = {
        "config": args.config,
        "yaml_config": os.path.abspath(args.yaml_config),
        "pretrained_ckpt": params.pretrained_ckpt,
        "track_ckpt": os.path.abspath(checkpoint_path),
        "momentum_ckpt": os.path.abspath(args.momentum_ckpt) if args.momentum_ckpt else None,
        "track_fit_config": trainer.vertex_head.track_fit_config(),
        "max_eval_events": args.max_eval_events,
        "metrics": result["metrics"],
    }
    json_path = logfile.replace(".log", "_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    trainer.log_infile("[vertex] " + json.dumps(result["metrics"]))
    print(f"[vertex] wrote {logfile}\n[vertex] wrote {npz_path}\n[vertex] wrote {json_path}")


if __name__ == "__main__":
    main()
