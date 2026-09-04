#!/usr/bin/env python3
"""
Evaluation script for point classification downstream task.
"""
import os
import sys
import argparse
import gc

import torch

# make sure the FM4NPP modules can be imported
sys.path.append('../..')


from fm4npp.utils import YParams
from point_classification_trainer import DownstreamTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for point classification downstream task"
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config name (e.g. d9_m64_k30_p20)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to the trained checkpoint (optional; overrides default)",
    )
    parser.add_argument(
        "--run_num",
        type=str,
        default="0",
        help="Run number / seed identifier",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/shuhang/FM4NPP/downstream_eval/",
        help="Root directory to store evaluation outputs",
    )
    parser.add_argument(
        "--eventnumber",
        type=int,
        default=70000,
        help="Number of events (samples) to evaluate",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument(
        "--data_root_test",
        type=str,
        default=None,
        help="Test-split root. Without this the config value is overwritten by a BNL "
             "mount that exists on no other machine, with no way to override it.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory holding the trained head. Without this it is read from a "
             "hardcoded /home/shuhang path, so a head you trained yourself is never found.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed suffix in the checkpoint name, matching train_point_classification.py. "
             "Omit for checkpoints written before seeds were recorded.",
    )
    args = parser.parse_args()

    # Default mapping from config to checkpoint if not provided via --checkpoint
    model2ckpt = {
        'd9_m5_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k5.ckpt',
        'd9_m1_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k5.ckpt',
        'd9_m3_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k5.ckpt',
        'd9_m4_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k5.ckpt',
        'd9_m4_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k5.ckpt',
        'd9_m5_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k5.ckpt',
        'd9_m1_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k30.ckpt',
        'd9_m3_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k30.ckpt',
        'd9_m4_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k30.ckpt',
        'd9_m5_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k30.ckpt',
        'd9_m64_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k5_debugged.ckpt',
        'd9_m64_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k30.ckpt',
        'd9_m96_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k5.ckpt',
        'd9_m96_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k30.ckpt',
        'd9_m128_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k5.ckpt',   
        'd9_m128_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k30.ckpt',
        'd9_m192_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k5.ckpt',
        'd9_m192_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k30.ckpt',
        'ablate_reference': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_reference.ckpt',
        'ablate_pe_PROJ': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_PROJ.ckpt',
        'ablate_pe_FF': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_FF.ckpt',
        'ablate_pe_CPE': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_CPE.ckpt',
        'ablate_order_RPE': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_RPE.ckpt',
        'ablate_order_REP': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_REP.ckpt',
        'ablate_order_PER': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_PER.ckpt',
        'ablate_embedconcat': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_embedconcat.ckpt',
        'ablate_lossreweight': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_lossreweight.ckpt',
        'ablate_space_filling_hilbert': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_hilbert.ckpt',
        'ablate_space_filling_z': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_z.ckpt',
        'ablate_novoxelize': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_novoxelize.ckpt',
    }

    # Determine which checkpoint to use
    

    # Prepare hyperparameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.limit_data = True
    params.limit_size = args.eventnumber
    params.batch_size = args.eval_batch_size
    params.valid_batch_size = args.eval_batch_size
    params.pretrained_ckpt = model2ckpt[args.config]
    params.log_file_name = f"{args.config}_eval_{params.task}_d{params.limit_size}_{args.run_num}.log"
    params.num_embedder_layers = 0
    # [B31] Both of these used to be unconditional literals pointing at the author's own
    # machine: the test root silently overrode whatever the config said, and the checkpoint
    # directory ignored --root_dir, so a head trained anywhere else could never be scored.
    # Tracking got this treatment in [FIX B21]; the point-classification tasks did not.
    if args.data_root_test:
        params.data_root_test = args.data_root_test
    seed_suffix = "" if args.seed is None else f"_seed{args.seed}"
    checkpoint_name = (f"{args.config}_nerf_{params.task}_d{params.limit_size}"
                       f"_{args.run_num}{seed_suffix}_checkpoint.pth")
    checkpoint_base_dir = args.checkpoint_dir or args.root_dir
    checkpoint_path = os.path.join(checkpoint_base_dir, checkpoint_name)

    # Ensure output directory exists
    log_dir = args.root_dir
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, params.log_file_name)

    # Launch and run inference
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    trainer.inference(
        checkpoint_path=checkpoint_path,
        pretrain=True,                # evaluation uses downstream checkpoint
        logfile=logfile
    )
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
