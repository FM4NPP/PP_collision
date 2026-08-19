#!/usr/bin/env python3
"""
Downstream tracking training script.
"""
import os
import sys
import argparse
import gc
import random
import numpy as np

import torch

# ensure fm4npp modules can be found
sys.path.append('../..')


from fm4npp.utils import YParams
from track_finding_trainer import DownstreamTrainer


def set_seed(seed):
    """[FIX B8] Set all random seeds for reproducibility.

    Without this the downstream run is non-deterministic and the paper's
    multi-seed studies cannot be reproduced.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    print(f"Random seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Downstream tracking training script")
    parser.add_argument("--yaml_config", default='', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='', type=str, help="Model config name")
    parser.add_argument("--run_num", default='0', type=str, help="Sub run number")
    parser.add_argument("--root_dir", default='/home/shuhang/FM4NPP/downstream_log/', type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=70000, type=int, help="downstream training event number")
    #parser.add_argument("--usepretrain", default=True, type=str, help="use pretrain model")
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
    parser.add_argument("--mambaversion", default="mamba2", type=str, help="mambd2/mamba1 for the pretrain model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # [FIX B8] seed before anything touches an RNG
    set_seed(args.seed)

    # Mapping from model name to log file and checkpoint paths
    model2log = {
        'd9_m1_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m1_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m3_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m4_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m4_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m3_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m5_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m5_k5_p20_run_noAMP1_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m1_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m1_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m3_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m4_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m4_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m3_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m5_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m5_k5_p20_run_noAMP1_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
    }
    
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
        # 5M Parameter Models
        'longformer_5m_downstream': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/longformer_5m/try1/training_checkpoints/ckpt_best.tar',
        'linformer_5m_downstream': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/linformer_5m/try0/training_checkpoints/ckpt_best.tar',
        'mamba_5m_downstream': '/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA/mamba_5m/try0/training_checkpoints/ckpt_best.tar',
    }

    # Example overrides for running in a notebook; uncomment to hardcode
    # args.yaml_config = "/home/shuhangli/FMNP/FM4NPP/scripts/configs/mamba.yaml"
    # args.config = "d9_m96_k5_p20"
    # args.run_num = "2"

    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.continue_from_best = True
    params.batch_size = int(args.train_batch_size)
    params.limit_data = True
    params.limit_size = int(args.eventnumber)
    params.valid_batch_size = 1
    params.pretrained_ckpt = model2ckpt[args.config]
    # [FIX B8] the seed must appear in artifact names, otherwise runs at different seeds
    # overwrite each other and eval cannot find the checkpoint it is asked for.
    base_name = f"{args.config}_nerf_tracking_head_d{params.limit_size}_{args.run_num}_seed{args.seed}"
    if args.usepretrain:
        params.log_file_name = base_name + ".log"
    else:
        params.log_file_name = base_name + "_nopretrain.log"
    params.loss_matched_ce_weight = 0.5
    params.loss_unmatched_ce_weight = 0.1
    params.loss_dice_weight = 1
    params.loss_focal_weight = 30
    params.num_embedder_layers = 0
    params.mambaversion = args.mambaversion


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
