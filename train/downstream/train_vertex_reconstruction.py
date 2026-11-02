#!/usr/bin/env python3
"""
Vertex reconstruction training (phase 2), laid out like train_track_finding.py. VertexHead has no
trainable parameters yet, so train() raises NotImplementedError; evaluate with
eval_vertex_reconstruction.py. Artifacts: {config}_nerf_vertex_head_d{N}_{run}_seed{seed}.
"""
import os
import sys

# TEMPORARY -- remove when phase 2 gives this script a real train(). Until the downstream
# trainers were aligned with track/point, this file was the vertex *evaluation* script; stop
# anyone still running it that way here, before the slow imports, instead of failing later
# on an unknown flag or train()'s NotImplementedError.
if __name__ == "__main__":
    sys.exit(
        "train_vertex_reconstruction.py no longer evaluates vertices; it is reserved for phase-2\n"
        "training, which does not exist yet. To run vertex reconstruction use:\n\n"
        "    python eval_vertex_reconstruction.py --root_dir <out/> [--checkpoint <head.pth>]\n\n"
        "Flags renamed to match eval_track_finding.py:\n"
        "    --track_ckpt         -> --checkpoint\n"
        "    --track_run_num      -> --run_num  (also names the run directory)\n"
        "    --no-pretrain-track  -> --no-pretrain\n"
        "    --max_events         -> --max_eval_events  (default now 20000)\n"
        "    --save_csv           -> on by default; --no-csv to turn it off\n"
        "Outputs now go to --root_dir, named after the log file\n"
        "({track_config}_eval_vertex_head_d...log, _per_point_data.csv, _metrics.json, _results.npz).\n"
        "See eval_vertex_reconstruction.py --help."
    )

import argparse
import gc
import random
import numpy as np

import torch

# ensure fm4npp modules can be found
sys.path.append('../..')


from fm4npp.utils import YParams
from vertex_reconstruction_trainer import DownstreamTrainer

DEFAULT_YAML = "../../scripts/configs/mamba_tracking.yaml"
DEFAULT_CONFIG = "mamba_5m_downstream"


def set_seed(seed):
    """Set all random seeds for reproducibility, as train_track_finding.py does ([FIX B8])."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    print(f"Random seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Downstream vertex reconstruction training script")
    parser.add_argument("--yaml_config", default=DEFAULT_YAML, type=str, help="Path to YAML config file")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=str, help="Model config name")
    parser.add_argument("--pretrained_ckpt", default=None, type=str,
                        help="Frozen backbone checkpoint (default: pretrained_ckpt from the config)")
    parser.add_argument("--run_num", default='0', type=str, help="Sub run number")
    parser.add_argument("--root_dir", required=True, type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=70000, type=int, help="downstream training event number")
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
        help="disable pretrained model",
    )
    parser.set_defaults(usepretrain=True)
    parser.add_argument("--train_batch_size", default=32, type=int, help="train batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--feature_cache", default=None, type=str,
                        help="Directory of precomputed frozen-backbone features "
                             "(scripts/cache_features.py). Skips the backbone forward.")
    parser.add_argument("--feature_cache_val", default=None, type=str,
                        help="Feature cache for the validation split")
    args = parser.parse_args()

    # seed before anything touches an RNG
    set_seed(args.seed)

    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    elif not getattr(params, "pretrained_ckpt", None):
        sys.exit(f"config {args.config!r} has no pretrained_ckpt; add one to the YAML or pass --pretrained_ckpt")
    params.continue_from_best = True
    params.batch_size = int(args.train_batch_size)
    params.limit_data = True
    params.limit_size = int(args.eventnumber)
    params.valid_batch_size = 1
    # the seed appears in artifact names so runs at different seeds don't overwrite each other
    base_name = f"{args.config}_nerf_vertex_head_d{params.limit_size}_{args.run_num}_seed{args.seed}"
    if args.usepretrain:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"
    params.num_embedder_layers = 0
    params.return_reg_test = True        # loaders then yield reg (truth) as a 4th element
    params.return_dict = False
    params.feature_cache = args.feature_cache
    params.feature_cache_val = args.feature_cache_val

    # Launch and train
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    checkpoint_path = None
    trainer.train(pretrain=args.usepretrain, train_from_checkpoint=False, checkpoint_path=checkpoint_path)

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
