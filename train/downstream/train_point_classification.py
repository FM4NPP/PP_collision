#!/usr/bin/env python3
"""
Downstream tracking training script.
"""
import os
import sys
import argparse
import gc

import torch

# ensure your fm4npp modules can be found
sys.path.append('../..')
sys.path.append('/home/shuhang/FM4NPP/FM4NPP')
sys.path.append('/home/shuhang/FM4NPP/FM4NPP/train/downstream')

from fm4npp.utils import YParams
from point_classification_trainer import DownstreamTrainer

def main():
    parser = argparse.ArgumentParser(description="Downstream tracking training script")
    parser.add_argument("--yaml_config", default='', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='', type=str, help="Model config name")
    parser.add_argument("--run_num", default='0', type=str, help="Sub run number")
    parser.add_argument("--root_dir", default='/home/shuhang/FM4NPP/downstream_log/', type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=70000, type=int, help="downstream training event number")
    parser.add_argument("--usepretrain", default=True, type=str, help="use pretrain model")
    parser.add_argument("--train_batch_size", default=32, type=int, help="train batch size")
    args = parser.parse_args()

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
    params.log_file_name = f"{args.config}_nerf_{params.task}_d{params.limit_size}_{args.run_num}.log"
    params.num_embedder_layers = 0

    # Launch and train
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    checkpoint_path = None
    trainer.train(pretrain=True, train_from_checkpoint=False, checkpoint_path=checkpoint_path)

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
