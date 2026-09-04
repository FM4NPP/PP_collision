#!/usr/bin/env python3
"""
Evaluation script for point classification downstream task.
"""
import os
import sys
import argparse
import gc

import torch

# make sure your FM4NPP modules can be imported
sys.path.append('../..')

from fm4npp.utils import YParams
from track_finding_trainer import DownstreamTrainer

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
        "--seed",
        type=int,
        default=42,
        help="[FIX B8] Random seed used during training (selects the checkpoint)",
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
        help="[FIX B23] Save per-point track assignments to CSV (default: enabled). "
             "This is the input to calculate_tracking_eff_purity.py.",
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
        help="[FIX B21] Evaluation data root (default: data_root_test from the config)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="[FIX B21] Where downstream checkpoints live (default: --root_dir)",
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
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument(
        "--max_eval_events",
        type=int,
        default=20000,
        help="Evaluate at most this many events. Useful for a fast environment "
             "check: a correct stack and a broken one differ by ~0.9 ARI, which "
             "is visible in a few hundred events.",
    )
    parser.add_argument(
        "--combine_weights",
        type=str,
        default=None,
        help="Comma-separated POST-softmax layer weights. Required to evaluate a head that "
             "was trained against a --combine_layers cache: such a head was built with "
             "num_feature_layers=1 and cannot consume the live backbone's full stack. Read "
             "the vector from the cache's cache_meta.json.",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path to the pretrained backbone. Overrides the built-in model2ckpt table, "
             "which points at paths that exist only on the original author's cluster.",
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
    # [B35] The model2ckpt dict above hardcodes the original author's cluster paths and is
    # assigned AFTER YParams, so it silently overrides whatever pretrained_ckpt the config
    # says -- and repoint_config.py only rewrites YAML, so this is the one path surface
    # repointing cannot reach. Anyone running outside that filesystem had no way to point at
    # their own backbone.
    if getattr(args, 'pretrained_ckpt', None):
        params.pretrained_ckpt = args.pretrained_ckpt
    else:
        params.pretrained_ckpt = model2ckpt[args.config]
    # [FIX B8] mirror the seed suffix used by train_track_finding.py
    seed_suffix = "" if args.no_seed_in_ckpt else f"_seed{args.seed}"
    checkpoint_base_name = f"{args.config}_nerf_tracking_head_d{params.limit_size}_{args.run_num}{seed_suffix}"
    log_base_name = f"{args.config}_eval_tracking_head_d{params.limit_size}_{args.run_num}{seed_suffix}"
    if args.usepretrain:
        params.log_file_name = log_base_name + ".log"
        checkpoint_name = checkpoint_base_name + "_checkpoint.pth"
    else:
        params.log_file_name = log_base_name + "_nopretrain.log"
        checkpoint_name = checkpoint_base_name + "_nopretrain_checkpoint.pth"
    params.num_embedder_layers = 0
    # [FIX B21] both of these were hardcoded to a personal cluster path, so --root_dir
    # was ignored and no external user could point eval at their own data/checkpoints.
    # data_root_test now comes from the config; the checkpoint dir defaults to --root_dir
    # (which is where train_track_finding.py writes) and is overridable with --checkpoint_dir.
    if args.data_root_test:
        params.data_root_test = args.data_root_test
    checkpoint_base_dir = args.checkpoint_dir or args.root_dir
    # [B34] --checkpoint was documented as "overrides default" but was parsed and never
    # read, so an explicit path silently fell back to the derived name. That made it
    # impossible to score a checkpoint someone else trained -- which is exactly the test
    # that eventually found [B29].
    checkpoint_path = (args.checkpoint if args.checkpoint
                       else os.path.join(checkpoint_base_dir, checkpoint_name))

    params.loss_matched_ce_weight = 0.5
    params.loss_unmatched_ce_weight = 0.1
    params.loss_dice_weight = 1
    params.loss_focal_weight = 30
    params.num_embedder_layers = 0
    params.return_reg_test = True

    # Ensure output directory exists
    log_dir = args.root_dir
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, params.log_file_name)

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
        trainer.eval_combine_weights = w       # so inference() collapses the stack the same way
        print(f'[eval] layer-combined head: {[round(x,4) for x in w]}')
    trainer.launch()
    trainer.inference(
        checkpoint_path=checkpoint_path,
        pretrain=args.usepretrain,                # evaluation uses downstream checkpoint
        logfile=logfile,
        # [FIX B23] inference() has always accepted these, but the CLI never set them, so
        # the per-point export was unreachable dead code and the efficiency/purity
        # analysis had no input.
        save_csv=args.save_csv,
        csv_output_path=args.csv_output_path,
    )
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
