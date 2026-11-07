"""
vertex_reconstruction_trainer.py: VertexHead after a trained track-finding head.

    frozen backbone -> frozen track finder -> VertexHead (PV + SVs)

Entry point: inference() (eval_vertex_reconstruction.py); train() is a phase-2 hook. Truth is used only for
scoring. Per-point CSV: track finding's columns (vtx_*: truth production vertex), reco_vtx_* (reconstructed PV)
and test_* truth columns.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import os, sys, time, shutil, random
import argparse
import inspect
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

import vertexhead
from vertexhead import VertexHead
from downstream_util import get_vertex_label
from loss import compute_vertex_metrics, assign_points_to_masks

PAD_VALUE = -100

# track finding's per-point CSV columns (vtx_*: the hit's particle's truth production vertex, cm)
PER_POINT_COLUMNS = ("batch_idx", "point_idx", "E_norm", "eta_norm", "phi_norm", "r_norm",
                     "px", "py", "pz", "vtx_x", "vtx_y", "vtx_z", "charge", "energy",
                     "seg_target", "pred_assignment", "confidence")

# the reconstructed PV, cm
RECO_COLUMNS = ("reco_vtx_x", "reco_vtx_y", "reco_vtx_z")

# TESTING ONLY: the truth PV (get_vertex_label) and reconstructed - truth, cm, NaN where undefined; the rank of
# the hit's particle's primary vertex (0: the PV, 1, 2...: pileup; -1: none)
TEST_COLUMNS = ("test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z",
                "test_residual_x", "test_residual_y", "test_residual_z", "test_residual_3d", "test_truth_pv_index")

CSV_COLUMNS = PER_POINT_COLUMNS + RECO_COLUMNS + TEST_COLUMNS


def write_metrics_log(metrics, logfile):
    """
    Append metrics to logfile as track_finding_trainer.inference() writes them: a tab-separated header line,
    then the values.
    """
    with open(logfile, "a") as f:
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


def per_point_columns(step, grouped, label, reg, mask, class_probs, mask_probs, vertex, truth_vertex, hit_vertex):
    """
    One batch of the per-point CSV.

    Args:
        step: written as batch_idx; grouped: (B, N, 4); label: (B, N) truth track ID; reg: (B, N, 8) truth;
        mask: (B, N) real hits; class_probs, mask_probs: raw track-head outputs; vertex: (B, 3) reconstructed PV;
        truth_vertex: (B, 3), NaN where undefined; hit_vertex: (B, N) the hit's primary-vertex rank (get_vertex_label).

    Returns:
        dict column -> 1-D array (CSV_COLUMNS), one entry per real hit.
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
        **{name: col(reg[..., k]) for k, name in zip((3, 4, 5), ("vtx_x", "vtx_y", "vtx_z"))},
        **{name: col(vertex[:, k]) for k, name in enumerate(RECO_COLUMNS)},
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
    columns["test_truth_pv_index"] = hit_vertex.to(mask.device)[b, n].cpu().numpy()
    return {k: columns[k] for k in CSV_COLUMNS}


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
        """
        Run folder exp_dir with a new logs/vertex_eval_<N>.log (parameters, then metrics); its per-point CSV goes
        to data/ (csv_path), its metrics JSON to metrics/ (metrics_path). No files with no_output.
        """
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        if getattr(self.params, "no_output", False):
            self.logfile = self.csv_path = self.metrics_path = os.devnull
            return
        logs, idx = os.path.join(exp_dir, "logs"), 0
        if self.world_rank == 0:
            os.makedirs(logs, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
            while os.path.exists(os.path.join(logs, f'vertex_eval_{idx}.log')):
                idx += 1
        self.logfile = os.path.join(logs, f'vertex_eval_{idx}.log')
        self.csv_path = os.path.join(exp_dir, "data", f'vertex_eval_{idx}_per_point_data.csv')
        self.metrics_path = os.path.join(exp_dir, "metrics", f'vertex_eval_{idx}.json')
        if self.world_rank == 0:
            with open(self.logfile, 'w') as f:
                f.write('Initialized at: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        if dist.is_initialized():
            dist.barrier()

    def log_infile(self, log):
        with open(self.logfile, "a") as f:
            f.write("{}\n".format(log))

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
        # the run folder: params.run_folder (the eval: <backbone>/<track head>/e<events>), else <config>/<run_num>
        exp_dir = os.path.join(self.root_dir, getattr(self.params, "run_folder", None) or os.path.join(self.config, self.run_num))
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

    def setup_vertexing(self, track_ckpt):
        """
        Load the track head and build VertexHead (after launch()). From params: vertex_detector (detector file),
        vertex_bz, vertex_device, and any VertexHead argument as vertex_<argument>, each only where set.
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

        # VertexHead arguments set in params as vertex_<argument>; vertex_bz is the field (Bz)
        names = set(inspect.signature(VertexHead.__init__).parameters) - {"self"}
        settings = {name: getattr(self.params, "vertex_" + name) for name in names
                    if getattr(self.params, "vertex_" + name, None) is not None}
        if getattr(self.params, "vertex_bz", None) is not None:
            settings["Bz"] = float(self.params.vertex_bz)
        # VertexHead's device (vertex_device, e.g. "cpu": faster than the GPU for its many small operations)
        self.vertex_device = torch.device(getattr(self.params, "vertex_device", None) or self.device)
        detector = getattr(self.params, "vertex_detector", None)
        head = VertexHead.from_detector(detector, **settings) if detector else VertexHead(**settings)
        self.vertex_head = head.to(self.vertex_device).eval()
        self.vertex_head.start_acts()       # boots while the checkpoints and data load

    def _vertex_head(self, class_probs, mask_probs, points, padding_mask):
        """VertexHead on its device (vertex_device); its tensor outputs back on the trainer's device."""
        d = self.vertex_device
        result = self.vertex_head(class_probs=class_probs.to(d), mask_probs=mask_probs.to(d), points=points.to(d),
                                  padding_mask=padding_mask.to(d))
        move = lambda v: v.to(self.device) if torch.is_tensor(v) else (  # noqa: E731
            {k: move(x) for k, x in v.items()} if isinstance(v, dict) else v)
        return {k: move(v) for k, v in result.items()}

    def _vertex_head_chunk(self, records):
        """
        VertexHead once over several loader batches, padded to a common size; returns each batch's
        (vertex_estimate, fit_is_valid). Each event keeps only the track slots VertexHead can fit, those at least
        min_hits real hits are assigned to (VertexHead's argmax), and the hits of the others stay unassigned: the
        same result, without the track head's mostly empty max_gt_classes slots.
        """
        events = []
        for r in records:
            for e in range(r["mask"].shape[0]):
                probs, real = r["mask_probs"][e].float(), r["mask"][e]
                assigned = real & (probs.max(dim=-1).values > 0)
                best = probs.argmax(dim=-1)
                n_hits = torch.zeros(probs.shape[-1], dtype=torch.long, device=probs.device).index_add_(
                    0, best[assigned], torch.ones_like(best[assigned]))
                slots = (n_hits >= self.vertex_head.min_hits).nonzero(as_tuple=True)[0]
                probs = probs * (n_hits >= self.vertex_head.min_hits)[best].unsqueeze(-1)    # other hits: unassigned
                events.append((r["class_probs"][e, slots].float(), probs[:, slots], r["grouped"][e], real))
        n_hits = max(len(real) for *_, real in events)
        n_tracks = max(1, max(len(c) for c, *_ in events))
        pad = lambda t, dims, value=0.0: torch.nn.functional.pad(t, dims, value=value)  # noqa: E731
        result = self._vertex_head(
            class_probs=torch.stack([pad(c, (0, 0, 0, n_tracks - len(c))) for c, *_ in events]),
            mask_probs=torch.stack([pad(m, (0, n_tracks - m.shape[1], 0, n_hits - m.shape[0])) for _, m, *_ in events]),
            points=torch.stack([pad(p, (0, 0, 0, n_hits - len(p))) for *_, p, _ in events]),
            padding_mask=torch.stack([pad(real, (0, n_hits - len(real)), value=False) for *_, real in events]))
        sizes = [r["mask"].shape[0] for r in records]
        return list(zip(result["vertex_estimate"].split(sizes), result["fit_is_valid"].split(sizes)))

    def inference(self, checkpoint_path, pretrain=True, save_csv=False, csv_output_path=None):
        """
        setup_vertexing, then run_vertexing over at most max_eval_events; appends the metrics to the run log and
        writes the CSV (default: next to the run log).

        Args:
            checkpoint_path: track-head checkpoint; pretrain: backbone features (True) or raw points.
            save_csv, csv_output_path: the per-point CSV.
        """
        self.setup_vertexing(checkpoint_path)
        result = self.run_vertexing(pretrain=pretrain, max_events=getattr(self, "max_eval_events", 20000),
                                    save_csv=save_csv)
        if self.world_rank == 0:
            write_metrics_log(result["metrics"], self.logfile)
            if save_csv:
                result["csv_output_path"] = csv_output_path or self.csv_path
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
        Backbone and track head per loader batch; VertexHead per vertex_chunk_events events (default 32).

        Args:
            pretrain: backbone features (True) or raw points; max_events: limit (None: all); save_csv: build the CSV.

        Returns:
            dict: metrics (incl. time_<stage>_ms_per_event), vertex_estimate, vertex_target (n, 3), fit_is_valid (n,),
            timing (seconds per stage), per_point (with save_csv).
        """
        self.model.eval()
        amp = torch.cuda.is_available() and self.use_amp
        est, tgt, ok, per_point = [], [], [], []
        chunk_events = int(getattr(self.params, "vertex_chunk_events", None) or 32)
        pending = []        # loader batches waiting for VertexHead

        def flush():
            """VertexHead over the pending batches, then their metrics inputs and CSV rows."""
            if not pending:
                return
            fitted = self._vertex_head_chunk(pending)
            lap("vertex_head")
            for r, (vertex, valid) in zip(pending, fitted):
                if save_csv:
                    truth_vertex = r["truth"]["vertex_target"].masked_fill(~r["truth"]["vertex_valid"][:, None],
                                                                           float("nan"))
                    per_point.append(per_point_columns(r["step"], r["grouped"], r["label"], r["reg"], r["mask"],
                                                       r["class_probs"], r["mask_probs"], vertex, truth_vertex,
                                                       r["truth"]["hit_vertex"]))
                est.append(vertex)
                tgt.append(r["truth"]["vertex_target"])
                ok.append(valid & r["truth"]["vertex_valid"])
            pending.clear()
            lap("truth_and_csv")

        timing = dict.fromkeys(("data", "backbone", "track_head", "vertex_head", "truth_and_csv"), 0.0)
        for key in vertexhead.ACTS_TIMING:
            vertexhead.ACTS_TIMING[key] = type(vertexhead.ACTS_TIMING[key])(0)
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        clock = [time.perf_counter()]

        def lap(stage):
            sync()
            now = time.perf_counter()
            timing[stage] += now - clock[0]
            clock[0] = now

        for step, batch in enumerate(tqdm(self.val_data_loader)):
            if len(batch) != 4:
                raise RuntimeError("loader must yield (grouped, label, knearest, reg); "
                                   "set params.return_reg_test = True before launch()")
            grouped, label, _, reg = batch
            b, c = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(b, -1, c).to(self.device)
            reg = reg.to(self.device)
            mask = grouped[..., 0] != PAD_VALUE
            lap("data")

            with autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
                if pretrain:
                    _, pre_embed, _ = self.model(grouped, return_z=True)
                    lap("backbone")
                    out = self.down_model(grouped, self._layer_features(pre_embed), pretrain=True, padding_mask=mask)
                else:
                    out = self.down_model(grouped, feature=None)
                lap("track_head")

            # padded rows are zeros and would pass get_vertex_label's 1 cm cut; exclude them. The truth PV: the
            # origin shared by the most particles
            truth = get_vertex_label(reg, valid_tracks=(torch.hypot(reg[..., 3], reg[..., 4]) < 1.0) & mask,
                                     particle_id=label)
            pending.append({"step": step, "grouped": grouped, "label": label, "reg": reg, "mask": mask, "truth": truth,
                            "class_probs": out["class_probs"], "mask_probs": out["mask_probs"]})
            lap("truth_and_csv")
            n_seen = sum(len(x) for x in est) + sum(r["mask"].shape[0] for r in pending)
            last = bool(max_events) and n_seen >= max_events
            if last or sum(r["mask"].shape[0] for r in pending) >= chunk_events:
                flush()
            if last:
                break
        flush()              # the loader ended with a partial chunk

        est, tgt, ok = torch.cat(est), torch.cat(tgt), torch.cat(ok)
        metrics = compute_vertex_metrics(est, tgt, ok)
        metrics.update(n_events=len(ok), fit_valid_rate=ok.float().mean().item())
        timing["acts_start"] = vertexhead.ACTS_TIMING["start_s"]
        timing["acts_propagate"] = vertexhead.ACTS_TIMING["propagate_s"]
        n_events = max(len(ok), 1)
        metrics.update({f"time_{k}_ms_per_event": v / n_events * 1e3 for k, v in timing.items()})
        if self.log_to_screen:
            print("[vertex] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items() if not k.startswith("time_")))
            total = sum(timing[k] for k in ("data", "backbone", "track_head", "vertex_head", "truth_and_csv"))
            print(f"[vertex] time over {len(ok)} events: {total:.1f} s ({total / n_events * 1e3:.1f} ms/event)")
            for k in ("data", "backbone", "track_head", "vertex_head", "truth_and_csv"):
                print(f"[vertex]   {k:14s} {timing[k]:8.2f} s  {timing[k] / n_events * 1e3:8.2f} ms/event  "
                      f"{timing[k] / max(total, 1e-12):6.1%}")
            if timing["acts_start"] or timing["acts_propagate"]:
                print(f"[vertex]     of which ACTS: worker start {timing['acts_start']:.2f} s, propagation "
                      f"{timing['acts_propagate']:.2f} s ({vertexhead.ACTS_TIMING['states']} track states)")
        output = {"metrics": metrics, "vertex_estimate": est.cpu(), "vertex_target": tgt.cpu(),
                  "fit_is_valid": ok.cpu(), "timing": timing}
        if save_csv:
            output["per_point"] = {k: np.concatenate([p[k] for p in per_point]) for k in CSV_COLUMNS}
        return output
