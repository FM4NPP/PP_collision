#!/usr/bin/env python3
"""
Evaluate VertexHead after a trained track-finding head. Run from train/downstream:

    python eval_vertex_reconstruction.py --run_config ../../scripts/configs/vertex_<run>.yaml [options override the file]
    python eval_vertex_reconstruction.py --root_dir out/ --checkpoint <head.pth> [--no-csv]

Backbone: --pretrained_ckpt (else the config's). Track head: --checkpoint, else found by name in
--checkpoint_dir. VertexHead: --detector (detector file), --bz, --vertex_device, --vertex_chunk_events.
Outputs in <root_dir>/<backbone checkpoint>/<track-head checkpoint>/e<events>/: data/vertex_eval_<N>_per_point_data.csv,
logs/vertex_eval_<N>.log (parameters, metrics) and metrics/vertex_eval_<N>.json (inputs, fit settings, metrics).
--no_output writes nothing.
"""
import os
import sys
import argparse
import gc
import json

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
        help="Path for the per-point CSV (default: next to the run log)",
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
        help="Solenoid field, Tesla (default: YAML vertex_bz, else the detector file's, else 1.4)",
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
        help="Test run: write no files or run folder; the metrics and timing are only printed. "
             "--root_dir is then not needed",
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
        help="YAML file of this script's options (keys = option names without --, e.g. scripts/configs/vertex_m5_gpuserver0.yaml); "
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
        path_keys = {"yaml_config", "pretrained_ckpt", "checkpoint", "checkpoint_dir",
                     "csv_output_path", "data_root_test", "root_dir", "detector"}
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
    if args.data_root_test:
        params.data_root_test = args.data_root_test
    checkpoint_path = (args.checkpoint if args.checkpoint
                       else os.path.join(args.checkpoint_dir or params.checkpoint_dir, checkpoint_name))
    # the run folder: <backbone checkpoint>/<track-head checkpoint>/e<events> (outputs vertex_eval_<N>.*)
    stem = lambda path: os.path.splitext(os.path.basename(path))[0]  # noqa: E731
    head = stem(checkpoint_path).replace("_checkpoint", "").replace("_nerf_tracking_head", "")
    params.run_folder = os.path.join(stem(params.pretrained_ckpt), head, f"e{args.max_eval_events}")
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"track-head checkpoint not found: {checkpoint_path}\n"
                 f"(pass --checkpoint, or check --track_config/--run_num/--seed/--eventnumber "
                 f"against the training run)")

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

    params.no_output = args.no_output
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
        save_csv=args.save_csv,
        csv_output_path=args.csv_output_path,
    )
    if trainer.world_rank == 0 and not args.no_output:
        save_results(result, trainer, params, args, checkpoint_path)
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()


def save_results(result, trainer, params, args, checkpoint_path):
    """
    Write the run's metrics JSON (metrics/): inputs, VertexHead's fit settings and the metrics.
    """
    json_path = trainer.metrics_path
    with open(json_path, "w") as f:
        json.dump({
            "config": args.config,
            "yaml_config": os.path.abspath(args.yaml_config),
            "pretrained_ckpt": params.pretrained_ckpt,
            "track_ckpt": os.path.abspath(checkpoint_path),
            "track_fit_config": trainer.vertex_head.track_fit_config(),
            "max_eval_events": args.max_eval_events,
            "metrics": result["metrics"],
        }, f, indent=2)
    print(f"[vertex] wrote {trainer.logfile}\n[vertex] wrote {json_path}")


if __name__ == "__main__":
    main()
