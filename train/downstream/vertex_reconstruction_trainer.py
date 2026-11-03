"""
vertex_reconstruction_trainer.py: VertexHead after an already-trained track-finding head.

    frozen backbone -> frozen track finder (train_track_finding.py checkpoint) -> VertexHead

DownstreamTrainer is standalone: its setup is a verbatim copy of track_finding_trainer's, and
train() and the epoch loops are phase-2 hooks. The entry point is inference(), as in the other
trainers (driven by eval_vertex_reconstruction.py). VertexHead finds secondary vertices by default
and reports the PV refitted without their tracks where valid (vertexhead.py); run_sv_vertexing()
also scores the original PV and summarizes the SVs. Truth is used only for scoring.

Per-point CSV: track_finding_trainer.inference()'s, row for row, except vtx_x/y/z hold the
reconstructed vertex; then the TESTING ONLY test_* truth columns (never reconstruction output),
and SV_COLUMNS from run_sv_vertexing.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import os, sys, time, shutil, random
import argparse
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gc, torch, torch.distributed as dist
from torch.amp import autocast
from contextlib import nullcontext

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm 
from ruamel.yaml import YAML
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts # pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

sys.path.append('../..')

from fm4npp.utils import *
from fm4npp.datasets.dataset import *
from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT
from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.models.mamba2 import Mamba2
from fm4npp.models.longformer_gpt import LongformerGPT
from fm4npp.models.linformer_gpt import LinformerGPT


from trackinghead import *
from loss import *
from cached_dataset import get_cached_data_loader
import csv

from vertexhead import VertexHead
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics, assign_points_to_masks

PAD_VALUE = -100

# track_finding_trainer.inference()'s per-point CSV columns, in its order
PER_POINT_COLUMNS = ("batch_idx", "point_idx", "E_norm", "eta_norm", "phi_norm", "r_norm",
                     "px", "py", "pz", "vtx_x", "vtx_y", "vtx_z", "charge", "energy",
                     "seg_target", "pred_assignment", "confidence")

# TESTING ONLY, appended after PER_POINT_COLUMNS: the event's truth vertex (get_vertex_label, the
# one the metrics score against) and the reconstructed-minus-truth residual, all cm; NaN where
# either vertex is undefined
TEST_COLUMNS = ("test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z",
                "test_residual_x", "test_residual_y", "test_residual_z", "test_residual_3d")

# SV columns appended by run_sv_vertexing when save_csv=True.
# is_sv_track: the track that owns this hit was assigned to at least one SV.
# sv_x, sv_y, sv_z: position of the best (lowest chi²/ndf) SV the track belongs to; NaN otherwise.
SV_COLUMNS = ("is_sv_track", "sv_x", "sv_y", "sv_z")

CSV_COLUMNS        = PER_POINT_COLUMNS + TEST_COLUMNS
CSV_COLUMNS_SV     = PER_POINT_COLUMNS + TEST_COLUMNS + SV_COLUMNS


def write_metrics_log(metrics, logfile):
    """
    Write metrics to logfile as track_finding_trainer.inference() does: a tab-separated header line,
    then the values.
    """
    with open(logfile, "w") as f:
        f.write("\t".join(metrics) + "\n")
        f.write("\t".join(str(v) for v in metrics.values()) + "\n")


def write_per_point_csv(columns, path):
    """
    Write {column: 1-D array} as pandas' to_csv(index=False) would (round-trip floats, NaN as
    empty), without pandas.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator=os.linesep)
        writer.writerow(columns)
        for row in zip(*(c.tolist() for c in columns.values())):
            writer.writerow(["" if v != v else v for v in row])    # v != v only for NaN


def per_point_columns(step, grouped, label, reg, mask, class_probs, mask_probs, vertex, truth_vertex):
    """
    One batch of the per-point CSV with the reconstructed vertex.

    Args:
        step: loader step, written as batch_idx.
        grouped: (B, N, 4) normalized hits; label: (B, N) truth track ID; mask: (B, N) real hits.
        reg: (B, N, 8) truth (px, py, pz, vx, vy, vz, q, e).
        class_probs, mask_probs: raw (bf16) track-head outputs, so assignments match inference().
        vertex: (B, 3) reconstructed vertex.
        truth_vertex: (B, 3) truth vertex, NaN where undefined; testing only.

    Returns:
        dict column -> 1-D numpy array in CSV_COLUMNS order, one entry per real hit.
    """
    assign = assign_points_to_masks({"pred_probs": class_probs, "pred_masks": mask_probs.permute(0, 2, 1)},
                                    option=2)["assignments"]
    b, n = mask.nonzero(as_tuple=True)

    def col(t):
        # float64 (exact for bf16 and float32) so the CSV prints each value as inference()'s .item() does
        t = t[b, n] if t.dim() == 2 else t[b]
        return t.double().cpu().numpy()

    columns = {
        "batch_idx": np.full(len(b), step, dtype=np.int64),
        "point_idx": n.cpu().numpy(),
        **{name: col(grouped[..., k]) for k, name in enumerate(("E_norm", "eta_norm", "phi_norm", "r_norm"))},
        **{name: col(reg[..., k]) for k, name in enumerate(("px", "py", "pz"))},
        "vtx_x": col(vertex[:, 0]), "vtx_y": col(vertex[:, 1]), "vtx_z": col(vertex[:, 2]),
        "charge": col(reg[..., 6]), "energy": col(reg[..., 7]),
        "seg_target": label.to(mask.device)[b, n].cpu().numpy(),
        "pred_assignment": assign[b, n].cpu().numpy(),
        "confidence": col(mask_probs.max(dim=-1).values),
    }
    # TESTING ONLY
    residual = vertex.double() - truth_vertex.double()
    for k, axis in enumerate("xyz"):
        columns[f"test_truth_vtx_{axis}"] = col(truth_vertex[:, k])
        columns[f"test_residual_{axis}"] = col(residual[:, k])
    columns["test_residual_3d"] = col(residual.norm(dim=-1))
    return {k: columns[k] for k in CSV_COLUMNS}


def per_point_columns_sv(
    step, grouped, label, reg, mask, class_probs, mask_probs,
    vertex, truth_vertex, sv_result,
):
    """
    per_point_columns plus SV_COLUMNS: whether each hit's track is in an SV, and the position of its
    lowest chi2/ndf SV.

    Args:
        sv_result: VertexHead output; the other arguments as per_point_columns.

    Returns:
        dict column -> 1-D numpy array in CSV_COLUMNS_SV order.
    """
    base = per_point_columns(step, grouped, label, reg, mask, class_probs, mask_probs,
                              vertex, truth_vertex)

    assign = assign_points_to_masks(
        {"pred_probs": class_probs, "pred_masks": mask_probs.permute(0, 2, 1)},
        option=2,
    )["assignments"]                           # (B, N)  long – track index per hit
    b, n = mask.nonzero(as_tuple=True)

    # sv_track_mask: (B, K, T) bool; sv_chi2_per_dof: (B, K)
    sv_mask  = sv_result["sv_track_mask"]      # (B, K, T)
    sv_chi2  = sv_result["sv_chi2_per_dof"]    # (B, K)  NaN where invalid
    sv_pos   = sv_result["sv_estimate"]        # (B, K, 3)

    # For each track, find which SV it belongs to (the one with lowest chi²/ndf)
    # sv_track_any: (B, T) – True if track i belongs to any SV
    sv_track_any = sv_mask.any(dim=1)          # (B, T)

    # Best SV index per track: among the K slots that include this track,
    # pick the one with lowest chi²/ndf.  Slots not including the track get +inf.
    # sv_q_per_track: (B, T, K) – chi²/ndf or +inf
    K = sv_mask.shape[1]
    sv_q_per_track = sv_chi2.unsqueeze(1).expand(-1, sv_mask.shape[2], -1)   # (B, T, K)
    sv_q_per_track = sv_q_per_track.nan_to_num(nan=float("inf"))
    sv_q_per_track = sv_q_per_track.masked_fill(~sv_mask.permute(0, 2, 1), float("inf"))
    best_sv_idx = sv_q_per_track.argmin(dim=-1)    # (B, T)  – 0 where not in any SV

    # Look up the best SV position for each track
    ev_idx  = torch.arange(sv_pos.shape[0], device=sv_pos.device).unsqueeze(-1).expand_as(best_sv_idx)
    best_sv_pos = sv_pos[ev_idx, best_sv_idx]      # (B, T, 3)
    # Zero out tracks not in any SV
    best_sv_pos = best_sv_pos.masked_fill(~sv_track_any.unsqueeze(-1), float("nan"))

    # Look up per-hit values: the track each hit belongs to
    track_idx  = assign[b, n]                       # (n_real_hits,) long
    is_sv      = sv_track_any[b, track_idx]         # (n_real_hits,) bool
    hit_sv_pos = best_sv_pos[b, track_idx]          # (n_real_hits, 3)

    columns = dict(base)
    columns["is_sv_track"] = is_sv.cpu().numpy().astype(np.float32)
    for k, axis in enumerate("xyz"):
        columns[f"sv_{axis}"] = hit_sv_pos[:, k].double().cpu().numpy()

    return {k: columns[k] for k in CSV_COLUMNS_SV}


class DownstreamTrainer():

    # ------------------------------------------------------------------------------------------
    # Infrastructure: verbatim copy of track_finding_trainer.DownstreamTrainer (everything except
    # inference, train and the epoch loops). Bug fixes there must be repeated here.
    # ------------------------------------------------------------------------------------------

    def _find_available_gpu(self, max_memory_threshold=1000):
        '''Find the first GPU with memory usage below a threshold (in MB).'''
        for gpu_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # Convert to MB
            if allocated < max_memory_threshold:
                return gpu_id
        return None 


    """ trainer class """
    def __init__(self, params, args):
        
        ''' init vars for distributed training (ddp) and logging'''
        self.root_dir = args.root_dir
        self.global_log_dir = os.path.join(args.root_dir, args.global_log_dir)
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        
        if self.world_size > 1: # multigpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True

        self.log_to_screen = (self.world_rank==0)
        if torch.cuda.is_available():
            available_gpu = self._find_available_gpu()
            if available_gpu is not None:
                
                torch.cuda.set_device(available_gpu)
                print(f"Using GPU {available_gpu} with memory below threshold.")
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        
        self.params = params
        self.use_lora = False
        self.use_amp = bool(getattr(self.params, 'use_amp', True))
        print("running on rank {} with world size {}".format(self.world_rank, self.world_size))



        
    def init_exp_dir(self, exp_dir):
                   
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')

        if self.params.continue_from_best:
            self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt_best.tar')

        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
        idx = 0
        logfile = os.path.join(exp_dir, 'performance{}.log'.format(idx))
        
        if self.world_rank==0:    
            while os.path.exists(logfile):
                idx += 1
                logfile = os.path.join(exp_dir, 'performance{}.log'.format(idx))
                
        if dist.is_initialized():
            dist.barrier()
        
        self.logfile = logfile

        if self.world_rank==0:            
            with open(self.logfile, 'w') as f:
                f.write('Initialized at: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    
            # Preparing global log directory
            if not os.path.isdir(self.global_log_dir):
                os.makedirs(self.global_log_dir)   
                
            self.globalfile = os.path.join(self.global_log_dir, 
                                           'config_{}_run_{}_{}.csv'.format(
                                               self.config,
                                               self.run_num, 
                                               self.parse_exp_details(self.params.params, 
                                                                      partial = ['data_version', 
                                                                                 'limit_size',
                                                                                 'model_version'],
                                                                      globalfile=True)
                                           ))
            print(self.globalfile)

        if dist.is_initialized():
            dist.barrier()
        
        if self.world_rank == 0 and not os.path.exists(self.globalfile):
            with open(self.globalfile, 'w') as f:
                pass
        if dist.is_initialized():
            dist.barrier()  
        
    def log_infile(self, log):
        with open(self.logfile, "a") as f:
            f.write("{}\n".format(log))

    def log_globalfile(self, split, step, loss, lr):
        with open(self.globalfile, "a") as f:
            f.write("{},{},{},{}\n".format(split, step, loss, lr))

    def finish_training(self):
        with open(self.finisher, 'w') as f:
            f.write(' ')
        raise FinishedTrainingError
    
    def parse_exp_details(self, D, partial=None, globalfile = False):
        """
        D: a dictionary listing parameters
        partial: a list of columns of interest
        """
        
        if globalfile:
            if partial is None:
                out = ','.join(['{}:{}'.format(a, b) for a,b in D.items()])
            else:
                out = ','.join(['{}:{}'.format(a, b) for a,b in D.items() if a in partial])
        else:
            out = 'Important Details:\n' + ''.join(['{}: {}\n'.format(a, b) for a,b in D.items()])
        return out

    def get_bin_index(self, seq_length):
        """Find the appropriate bin for a given sequence length."""
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= seq_length < self.bins[i + 1]:
                return i
        return len(self.bins) - 2  # Assign to last bin if out of range

    def update_moving_average(self, bin_idx, loss_value):
        """Update exponential moving average of loss per bin."""
        self.loss_moving_avg[bin_idx] = (
            self.smoothing_factor * self.loss_moving_avg[bin_idx] +
            (1 - self.smoothing_factor) * loss_value
        )

    def compute_inverse_loss_weights(self):
        """Compute inverse loss weights for each bin."""
        weights = {i: 1 / (self.loss_moving_avg[i] + self.epsilon) for i in self.loss_moving_avg}
        total_weight = sum(weights.values())
        return {i: weights[i] / total_weight for i in weights}  # Normalize weights

    def launch(self):
        print(self.root_dir, self.config, self.run_num)
        exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])
        self.init_exp_dir(exp_dir)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        print('batch size: ', self.params['global_batch_size'])
        print('local batch size: ', self.params['local_batch_size'])

        self.log_infile(self.parse_exp_details(self.params.params))       

        # get the pretrained model
        self.klen = self.params.klen
        if self.params.mambaversion == 'mamba1':
            self.model = Mamba1GPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                                d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                                embed_method=self.params.embed_method, pe_method=self.params.pe_method)
        elif self.params.mambaversion == 'longformer':
            self.model = LongformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                window_size=getattr(self.params, 'window_size', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        elif self.params.mambaversion == 'linformer':
            self.model = LinformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                seq_len=getattr(self.params, 'seq_len', 512),
                proj_dim=getattr(self.params, 'proj_dim', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        else:
            self.model = MambaGPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                    d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                    embed_method=self.params.embed_method, pe_method=self.params.pe_method)
        



        def initialize_mamba2(model, d_state, embed_dim):
            """ Properly initializes Mamba v2 to ensure stable learning. """
            
            with torch.no_grad():
                for name, param in model.named_parameters():

                    if "lin_B" in name:
                        param.normal_(mean=0.0, std=(d_state / embed_dim)**0.5)

                    elif "lin_C" in name:
                        param.normal_(mean=0.0, std=(1.0 / (embed_dim*d_state))**0.5)

                    elif "norm.weight" in name:
                        init.ones_(param)

                    # Bias Terms
                    elif "bias" in name:
                        init.zeros_(param)

                print(f"✅ Mamba v2 Model Initialized")
                
        Nu = self.params.embed_dim
        Nx = getattr(self.params, 'd_state', 16)

        # Only initialize Mamba models with Mamba-specific initialization
        if self.params.mambaversion in ['mamba1', 'mamba2']:
            initialize_mamba2(self.model, Nx, Nu)
        else:
            if self.world_rank == 0:
                print(f"✅ {self.params.mambaversion.capitalize()} Model Initialized")

        self.model = self.model.to(self.device)
        if self.world_rank == 0:
            print('Nparams: ', count_parameters(self.model))

        # distributed wrapper for data parallel
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank],
                                                find_unused_parameters=True)

            

        # set an optimizer and learning rate scheduler   
        params_a   = []
        params_b   = []
        params_c   = []
        params_else= []

        for name, p in self.model.named_parameters():
            if "A_log" in name:
                params_a.append(p)   # might do LR ~ Nu
            elif "lin_B" in name:
                params_b.append(p)   # might do LR ~ Nx / sqrt(Nu)
            elif "lin_C" in name:
                params_c.append(p)   # might do LR ~ sqrt(Nu) / Nx
            else:
                params_else.append(p)
                
        self.optimizer = torch.optim.AdamW([
            {"params": params_a,   "lr": self.params.min_lr * Nu},                   # e.g. for A
            {"params": params_b,   "lr": self.params.min_lr * Nx / (Nu**0.5)},       # e.g. for B
            {"params": params_c,   "lr": self.params.min_lr * (Nu**0.5) / Nx},       # e.g. for C
            {"params": params_else,"lr": self.params.min_lr},
        ], weight_decay=0.1, betas=(0.9, 0.95))

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=getattr(self.params, 'total_steps', 200),
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=self.params.warmup_steps)

        
        # get the dataloaders
        # [B27] With a feature cache the frozen backbone forward and the CPU preprocessing
        # are both skipped. Measured on one GB10 at batch 32: backbone 786 ms/batch (~45% of
        # step time) and dataloading 379 ms (~22%). For the 175M model this is the difference
        # between ~48 min/epoch and ~8 min/epoch.
        self.feature_cache = getattr(self.params, 'feature_cache', None)
        self.feature_cache_val = getattr(self.params, 'feature_cache_val', None)
        if self.feature_cache:
            self.train_data_loader, ds_tr = get_cached_data_loader(
                self.feature_cache, batch_size=self.params.local_batch_size, shuffle=True,
                num_workers=self.params.num_data_workers, limit_size=self.params.limit_size)
            self.train_sampler = None
            if ds_tr.embed_dim != self.params.embed_dim:
                raise ValueError(
                    f'feature cache width {ds_tr.embed_dim} != config embed_dim '
                    f'{self.params.embed_dim} -- wrong cache for this config')
            # [B28] A combined cache stores the softmax-weighted sum over layers rather than
            # the stack, which is 12x smaller (3,072 vs 36,864 bytes/point at width 1536 --
            # the difference between 185 GB and 2.17 TB for the 70k split). The head must be
            # told, so it builds with num_feature_layers=1 instead of 12.
            self.cache_combined = ds_tr.combine_layers
            if self.cache_combined:
                self.cache_layer_weights = ds_tr.layer_weights
                print(f'[cache] layer-combined cache: head layer mixing frozen to '
                      f'{[round(w, 4) for w in (self.cache_layer_weights or [])]}')
            if self.feature_cache_val:
                self.val_data_loader, _ = get_cached_data_loader(
                    self.feature_cache_val, batch_size=self.params.local_valid_batch_size,
                    shuffle=False, num_workers=self.params.num_data_workers)
            else:
                _, _, self.val_data_loader, _ = get_data_loader(self.params, dist.is_initialized())
        else:
            self.train_data_loader, self.train_sampler, self.val_data_loader, _ = get_data_loader(self.params,
                                                                                                  dist.is_initialized())

        # set loss functions
        self.loss_func = nn.MSELoss(reduction='none')
        self.centroid_loss_func = nn.MSELoss(reduction='none')
        self.loss_func_eval = nn.MSELoss(reduction='none')

        # checkpointing
        self.iters = 0
        self.startEpoch = 0
        self.resumed = False

        ##### Pretraining checkpoint
        print("Loading checkpoint %s"%self.params.pretrained_ckpt)
        self.restore_checkpoint(self.params.pretrained_ckpt)
        self.resumed = True

        # LoRA: apply low-rank adapters to frozen backbone
        if getattr(self.params, 'use_lora', False):
            from fm4npp.models.lora import apply_lora_to_model, get_lora_params
            target_modules = getattr(self.params, 'lora_target_modules', 'in_proj,out_proj,lin_B,lin_C').split(',')
            lora_rank = getattr(self.params, 'lora_rank', 8)
            lora_alpha = getattr(self.params, 'lora_alpha', 16)
            n_lora = apply_lora_to_model(self.model, target_modules, lora_rank, lora_alpha)
            self.use_lora = True
            self.lora_optimizer = torch.optim.AdamW(
                get_lora_params(self.model), lr=self.params.max_lr, weight_decay=0.01
            )
            total_backbone = sum(p.numel() for p in self.model.parameters())
            trainable_backbone = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"LoRA applied: {n_lora} adapter params ({trainable_backbone}/{total_backbone} trainable in backbone)")

        self.startEpoch = 0
        self.epoch = self.startEpoch
        self.logs = {}

        # 
        #  training
        #self.train()


    def cleanup(self):
        # 1) remove hooks
        for hook_list in ("fwd_hooks", "bwd_hooks"):
            for h in getattr(self, hook_list, []):
                h.remove()

        # 2) break references to big objects
        for obj in ("model", "down_model",
                    "optimizer", "down_optimizer",
                    "scheduler", "down_scheduler",
                    "train_data_loader", "val_data_loader"):
            if hasattr(self, obj):
                delattr(self, obj)

        # 3) empty CUDA cache and run GC
        torch.cuda.empty_cache()
        gc.collect()

        # 4) tear down DDP if we set it up
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        print("Cleanup complete. All resources released.")

    def _save_checkpoint(self, filename, epoch, is_best, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.down_model.state_dict(),
            'optimizer_state_dict': self.down_optimizer.state_dict(),
            'scheduler_state_dict': self.down_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'current_loss': loss,
            'params': vars(self.params)  # Save all hyperparameters
        }

        # Handle DistributedDataParallel wrapper
        if isinstance(self.down_model, torch.nn.parallel.DistributedDataParallel):
            checkpoint['model_state_dict'] = self.down_model.module.state_dict()

        if self.use_lora:
            lora_state = {k: v for k, v in self.model.state_dict().items()
                          if 'lora_A' in k or 'lora_B' in k}
            checkpoint['lora_state_dict'] = lora_state
            checkpoint['lora_optimizer_state_dict'] = self.lora_optimizer.state_dict()

        torch.save(checkpoint, os.path.join(self.params.checkpoint_dir, filename))

        msg = f"Saved {'best ' if is_best else ''}checkpoint at epoch {epoch} with loss {loss:.4f}"
        #print(msg) if self.log_to_screen else None

    def load_checkpoint(self, checkpoint_path, inference=False):
        """Load checkpoint with proper device mapping and DDP handling.
           If inference=True, only loads the model weights."""

        # 1. Get proper device string
        if isinstance(self.device, int):
            device_str = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        else:
            device_str = str(self.device)

        # 2. Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device_str, weights_only=False)

        # 3. Handle DDP keys
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 4. Load model weights
        if isinstance(self.down_model, torch.nn.parallel.DistributedDataParallel):
            self.down_model.module.load_state_dict(new_state_dict, strict=False)
        else:
            self.down_model.load_state_dict(new_state_dict, strict=False)

        # Load LoRA weights if present
        if self.use_lora and 'lora_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['lora_state_dict'], strict=False)

        # 5. If not inference mode, load optimizer/scheduler states
        if not inference:
            self.down_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.down_scheduler is not None:
                self.down_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.use_lora and 'lora_optimizer_state_dict' in checkpoint:
                self.lora_optimizer.load_state_dict(checkpoint['lora_optimizer_state_dict'])

            self.startEpoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))

        # 6. Log info
        if self.log_to_screen:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")



    


    def report_loss(self, loss_, dist):
        step_loss = torch.zeros((1), dtype=torch.float32, device=self.device)
        step_loss += loss_.detach()

        if dist.is_initialized():
            dist.all_reduce(step_loss)
            loss_log = float(step_loss.item()/dist.get_world_size())
        else:
            loss_log = step_loss.item()
        return loss_log

    def set_portion_condition(self, tmask, portion = 0.2):
        """tmask: a mask showing effective (i.e., non-padding area) region as 1"""
        total = tmask.sum(-1)
        condidx = torch.ceil(total * portion).long()        
        index_tensor = torch.arange(tmask.size(1)).expand(tmask.size(0), -1).to(tmask.device)  # Shape (B, N)
        newmask = (index_tensor < condidx.unsqueeze(1)).float()
        return newmask.bool()


    
            
    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.device), weights_only=False)
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()}
        #try:
            #self.model.load_state_dict(checkpoint['model_state'])
        self.model.load_state_dict(new_state_dict)
        #except:
        #    new_state_dict = OrderedDict()
        #    for key, val in checkpoint['model_state'].items():
        #        name = key[7:]
        #        new_state_dict[name] = val 
        #    self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']+1 if self.iters % len(self.train_data_loader) == 0 else checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # ------------------------------------------------------------------------------------------
    # Machine-learning hooks (phase 2). Same names and signatures as the existing trainers'
    # task-specific methods; VertexHead has no trainable parameters yet, so they are not implemented.
    # launch() above already builds the train loader, optimizer, scheduler and LoRA, and
    # _save_checkpoint / load_checkpoint save and load self.down_model, as the existing trainers do.
    # ------------------------------------------------------------------------------------------

    def train(self, pretrain=True, train_from_checkpoint=False, checkpoint_path=None):
        raise NotImplementedError("VertexHead has no trainable parameters yet (phase 2)")

    def downstream_end_to_end_one_epoch(self, pretrain=False):
        raise NotImplementedError("VertexHead has no trainable parameters yet (phase 2)")

    def validate_end_to_end_one_epoch(self, pretrain=False):
        raise NotImplementedError("VertexHead has no trainable parameters yet (phase 2)")

    # ------------------------------------------------------------------------------------------
    # VertexHead evaluation
    # ------------------------------------------------------------------------------------------

    def setup_vertexing(self, track_ckpt, momentum_ckpt=None):
        """
        Build and strictly load the track head, then build VertexHead. Call after launch().

        VertexHead settings from params (defaults; cm unless noted): vertex_bz (1.4 T),
        sv_min_track_pv_dca (0.05), sv_min_track_beam_dca (0.05), sv_max_pair_doca (1.0),
        sv_max_track_sv_doca (0.5), sv_seed_merge_distance (0.1), sv_min_sv_tracks (2),
        sv_min_sv_displacement (0.02), sv_max_sv_chi2_per_dof (10), sv_max_sv_per_event (6).

        Args:
            track_ckpt: train_track_finding.py checkpoint.
            momentum_ckpt: optional MomentumHead checkpoint to use instead of vertex_bz (obsolete with
                the momentum task, which moves to ACTS).
        """
        n_feat = 1 if getattr(self, "cache_combined", False) else self.params.num_layers_backbone
        self.down_model = MambaAttentionHead(
            input_dim=self.params.embed_dim, num_layers=0,
            num_embedder_layers=self.params.num_embedder_layers,
            d_state=64, d_conv=4, expand=2, num_feature_layers=n_feat,
            num_prototypes=self.params.max_gt_classes,
            embed_method=getattr(self.params, "embed_method", "add")).to(self.device)

        sd = torch.load(track_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

        # strict, unlike load_checkpoint(): a mismatched checkpoint raises instead of half-loading
        self.down_model.load_state_dict(sd)
        self.down_model.eval()

        bz = float(getattr(self.params, "vertex_bz", 1.4))

        momentum_head = None
        if momentum_ckpt:            # obsolete momentum task: imported only when its checkpoint is used
            from momentumhead import MomentumHead
            momentum_head = MomentumHead.from_checkpoint(momentum_ckpt)

        # PV + SV head (secondary vertexing is VertexHead's default); SV cuts can be tuned via params
        def _p(name, default):
            return type(default)(getattr(self.params, name, default))

        self.vertex_head = VertexHead(
            Bz=bz,
            momentum_head=momentum_head,
            min_track_pv_dca    = _p("sv_min_track_pv_dca",    0.05),
            min_track_beam_dca  = _p("sv_min_track_beam_dca",  0.05),
            max_pair_doca       = _p("sv_max_pair_doca",        1.0),
            max_track_sv_doca   = _p("sv_max_track_sv_doca",   0.5),
            seed_merge_distance = _p("sv_seed_merge_distance",  0.1),
            min_sv_tracks       = _p("sv_min_sv_tracks",        2),
            min_sv_displacement = _p("sv_min_sv_displacement",  0.02),
            max_sv_chi2_per_dof = _p("sv_max_sv_chi2_per_dof", 10.0),
            max_sv_per_event    = _p("sv_max_sv_per_event",     6),
        ).to(self.device).eval()

    def inference(self, checkpoint_path, pretrain=True, logfile=None, save_csv=False, csv_output_path=None,
                  momentum_ckpt=None):
        """
        Entry point, as in track_finding_trainer: setup_vertexing, then run_vertexing over at most
        self.max_eval_events (default 20000); writes the metrics log and, with save_csv, the CSV.

        Args:
            checkpoint_path: train_track_finding.py checkpoint.
            pretrain: True = frozen-backbone features (as trained); False = head on raw points.
            logfile: metrics log path; None = not written.
            save_csv, csv_output_path: per-point CSV switch and path (default derived from logfile).
            momentum_ckpt: optional MomentumHead checkpoint.

        Returns:
            run_vertexing's dict, plus csv_output_path when the CSV was written.
        """
        self.setup_vertexing(checkpoint_path, momentum_ckpt=momentum_ckpt)
        result = self.run_vertexing(pretrain=pretrain, max_events=getattr(self, "max_eval_events", 20000),
                                    save_csv=save_csv)
        if self.world_rank == 0:
            if logfile is not None:
                write_metrics_log(result["metrics"], logfile)
            if save_csv:
                result["csv_output_path"] = csv_output_path or (
                    logfile.replace(".log", "_per_point_data.csv") if logfile else "vertex_per_point_data.csv")
                write_per_point_csv(result["per_point"], result["csv_output_path"])
                print(f"[vertex] wrote {result['csv_output_path']} ({len(result['per_point']['batch_idx'])} points)")
        return result

    def _layer_features(self, pre_embed):
        """
        torch.stack(pre_embed), collapsed with eval_combine_weights when set (a head trained on a
        --combine_layers cache).
        """
        feature = torch.stack(pre_embed)
        weights = getattr(self, "eval_combine_weights", None)
        if weights is None:
            return feature
        return torch.einsum("lbnd,l->bnd", feature, feature.new_tensor(weights)).unsqueeze(0)

    @torch.no_grad()
    def run_vertexing(self, pretrain=True, max_events=None, save_csv=False):
        """
        Backbone -> track head -> VertexHead over the validation loader.

        Args:
            pretrain: True = frozen-backbone features (as trained); False = head on raw points.
            max_events: stop after this many events; None = whole loader.
            save_csv: also build the per-point CSV columns.

        Returns:
            dict: metrics (compute_vertex_metrics plus n_events, fit_valid_rate); vertex_estimate,
            vertex_target (n_events, 3); fit_is_valid (n_events,) fit valid and truth defined;
            per_point (CSV columns, with save_csv).
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp
        est, tgt, ok, per_point = [], [], [], []

        for step, batch in enumerate(tqdm(self.val_data_loader)):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            reg = reg.to(self.device)
            mask = grouped[..., 0] != PAD_VALUE

            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    out = self.down_model(grouped, self._layer_features(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)

            result = self.vertex_head(
                class_probs=out["class_probs"].float(), mask_probs=out["mask_probs"].float(),
                points=grouped, padding_mask=mask)

            # padded rows are zeros and would pass get_vertex_label's 1 cm cut; exclude them
            truth = get_vertex_label(reg, valid_tracks=(torch.hypot(reg[..., 3], reg[..., 4]) < 1.0) & mask)
            if save_csv:
                truth_vertex = truth["vertex_target"].masked_fill(~truth["vertex_valid"][:, None], float("nan"))
                per_point.append(per_point_columns(step, grouped, label, reg, mask, out["class_probs"],
                                                   out["mask_probs"], result["vertex_estimate"], truth_vertex))
            est.append(result["vertex_estimate"])
            tgt.append(truth["vertex_target"])
            ok.append(result["fit_is_valid"] & truth["vertex_valid"])
            if max_events and sum(len(x) for x in est) >= max_events:
                break

        est, tgt, ok = torch.cat(est), torch.cat(tgt), torch.cat(ok)
        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item())
        if self.log_to_screen:
            print("[vertex] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items()))
        output = {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                  "fit_is_valid": ok.cpu()}
        if save_csv:
            output["per_point"] = {k: np.concatenate([p[k] for p in per_point]) for k in CSV_COLUMNS}
        return output

    # ------------------------------------------------------------------------------------------
    # Secondary-vertex evaluation (PV + SVs, then the PV refitted without SV tracks)
    # ------------------------------------------------------------------------------------------

    @torch.no_grad()
    def run_sv_vertexing(self, pretrain=True, max_events=None, save_csv=False, refit_pv=True):
        """
        run_vertexing that also scores the original PV and summarizes the secondary vertices.

        Args:
            pretrain: True = frozen-backbone features (as trained); False = head on raw points.
            max_events: stop after this many events; None = whole loader.
            save_csv: also build the per-point CSV columns. SV columns (SV_COLUMNS) are added.
            refit_pv: report VertexHead's PV (the refit where used); False = the original fit.

        Returns:
            dict: metrics, vertex_estimate (reported PV); pv_original_metrics, pv_original
            (original fit); sv_summary (n_sv_mean, n_sv_max, sv_valid_rate, sv_displacement_mean,
            sv_n_tracks_mean); vertex_target; fit_is_valid; per_point (with save_csv).
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp

        est, est_orig, tgt, ok, ok_orig = [], [], [], [], []
        sv_valid_counts  = []   # (B,) number of valid SVs per event
        sv_displace_list = []   # flat list of valid SV displacements
        sv_ntracks_list  = []   # flat list of valid SV n_tracks
        per_point_rows   = []

        for step, batch in enumerate(tqdm(self.val_data_loader)):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            reg     = reg.to(self.device)
            mask    = grouped[..., 0] != PAD_VALUE

            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    out = self.down_model(grouped, self._layer_features(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)

            class_probs = out["class_probs"].float()
            mask_probs  = out["mask_probs"].float()

            # ── Step 1: PV, SVs and the PV refitted without SV tracks ────────
            sv_result = self.vertex_head(
                class_probs=class_probs,
                mask_probs=mask_probs,
                points=grouped,
                padding_mask=mask,
            )
            pv_estimate  = sv_result["pv_original"]["vertex_estimate"]    # (B, 3)
            pv_valid     = sv_result["pv_original"]["fit_is_valid"]       # (B,)

            # ── Step 2: refined PV (SV tracks removed where that refit is used) ──
            if refit_pv:
                refined_estimate = sv_result["vertex_estimate"]  # (B, 3)
                refined_valid    = sv_result["fit_is_valid"]     # (B,)
            else:
                refined_estimate = pv_estimate
                refined_valid    = pv_valid

            # ── Step 3: truth + metrics accumulation ─────────────────────────
            truth = get_vertex_label(
                reg,
                valid_tracks=(torch.hypot(reg[..., 3], reg[..., 4]) < 1.0) & mask,
            )
            truth_valid  = truth["vertex_valid"]
            truth_vertex = truth["vertex_target"]

            ok_r   = refined_valid   & truth_valid
            ok_pvo = pv_valid        & truth_valid

            est.append(refined_estimate)
            est_orig.append(pv_estimate)
            tgt.append(truth_vertex)
            ok.append(ok_r)
            ok_orig.append(ok_pvo)

            # ── Step 4: SV summary stats ──────────────────────────────────────
            sv_is_valid  = sv_result["sv_is_valid"]      # (B, K)
            sv_displace  = sv_result["sv_displacement"]  # (B, K)
            sv_n_tracks  = sv_result["sv_n_tracks"]      # (B, K) long

            sv_valid_counts.append(sv_is_valid.sum(dim=-1).float().cpu())   # (B,)
            if sv_is_valid.any():
                sv_displace_list.append(sv_displace[sv_is_valid].cpu())
                sv_ntracks_list.append(sv_n_tracks[sv_is_valid].float().cpu())

            # ── Step 5: optional per-point CSV ───────────────────────────────
            if save_csv:
                tv_nan = truth_vertex.masked_fill(~truth_valid[:, None], float("nan"))
                per_point_rows.append(
                    per_point_columns_sv(
                        step, grouped, label, reg, mask,
                        class_probs, mask_probs,
                        refined_estimate, tv_nan,
                        sv_result,
                    )
                )

            if max_events and sum(len(x) for x in est) >= max_events:
                break

        # ── Aggregate over all batches ────────────────────────────────────────
        est      = torch.cat(est)
        est_orig = torch.cat(est_orig)
        tgt      = torch.cat(tgt)
        ok       = torch.cat(ok)
        ok_orig  = torch.cat(ok_orig)

        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item())

        pv_orig_metrics = compute_vertex_metrics(est_orig, tgt, ok_orig)
        pv_orig_metrics.update(n_events=len(ok_orig), fit_valid_rate=ok_orig.float().mean().item())

        sv_counts  = torch.cat(sv_valid_counts)
        sv_summary = {
            "n_sv_mean":          sv_counts.mean().item(),
            "n_sv_max":           int(sv_counts.max().item()),
            "sv_valid_rate":      (sv_counts > 0).float().mean().item(),
        }
        if sv_displace_list:
            all_disp   = torch.cat(sv_displace_list)
            all_ntracks = torch.cat(sv_ntracks_list)
            sv_summary["sv_displacement_mean"] = all_disp.mean().item()
            sv_summary["sv_n_tracks_mean"]     = all_ntracks.mean().item()
        else:
            sv_summary["sv_displacement_mean"] = float("nan")
            sv_summary["sv_n_tracks_mean"]     = float("nan")

        if self.log_to_screen:
            print("[vertex-refined] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()))
            print("[vertex-original] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in pv_orig_metrics.items()))
            print("[sv] " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sv_summary.items()))

        output = {
            "metrics":              metrics,
            "pv_original_metrics":  pv_orig_metrics,
            "sv_summary":           sv_summary,
            "vertex_estimate":      est.cpu(),          # refined PV
            "pv_original":          est_orig.cpu(),     # PV before SV removal
            "vertex_target":        tgt.cpu(),
            "fit_is_valid":         ok.cpu(),
        }
        if save_csv:
            output["per_point"] = {
                k: np.concatenate([p[k] for p in per_point_rows])
                for k in CSV_COLUMNS_SV
            }
        return output
