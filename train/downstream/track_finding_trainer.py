import numpy as np
from sklearn.metrics import adjusted_rand_score
import os, sys, time, shutil, random
import argparse
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gc, torch, torch.distributed as dist

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
from fm4npp.models.longformer_gpt import LongformerGPT
from fm4npp.models.linformer_gpt import LinformerGPT
from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.models.mamba2 import Mamba2


from trackinghead import *
from loss import *


class DownstreamTrainer():
    
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

            if self.world_rank == 0:
                print(f"✅ Mamba Model Initialized")
                
        # Set model dimension variables for optimizer (used by all models)
        Nu = self.params.embed_dim
        Nx = getattr(self.params, 'd_state', 16)  # Default to 16 for non-Mamba models

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

        self.scaler = torch.amp.GradScaler('cuda') 
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=self.params.total_steps,
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=self.params.warmup_steps)

        
        # get the dataloaders
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
        self.restore_checkpoint(self.params.pretrained_ckpt, load_optimizer_state=False)
        self.resumed = False  # Not resuming, starting fresh with pretrained weights

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

    def inference(self, checkpoint_path, pretrain=True, logfile=None):
        """Initialize model and load weights for inference"""
        # 1. Initialize model architecture
        #self.down_model = MambaAttentionHead(
        #    input_dim=self.params.embed_dim,
        #    num_layers=3,
        #    d_state=64,
        #    d_conv=4,
        #    expand=2,
        #    num_feature_layers=self.params.num_layers_backbone,
        #    num_prototypes=self.params.max_gt_classes
        #).to(self.device)

        self.down_model = MambaAttentionHead(input_dim=self.params.embed_dim, num_layers=0,
                                  num_embedder_layers= self.params.num_embedder_layers, 
                                  d_state=64, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone, num_prototypes = self.params.max_gt_classes).to(self.device)
    
        total_params = sum(p.numel() for p in self.down_model.parameters())
        print(f"Total parameters in down_model: {total_params}")
        self.down_optimizer = optim.AdamW(self.down_model.parameters(), 
                                         lr=self.params.max_lr, # Mamba: Linear-Time Sequence Modeling with Selective State Spaces
                                         weight_decay=0.0001) 
        
        torch.nn.utils.clip_grad_norm_(self.down_model.parameters(), max_norm=1.0)


        self.down_scheduler = CosineAnnealingWarmupRestarts(self.down_optimizer,
                                          first_cycle_steps=200,
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=20)

        self.matcher = PointHungarianMatcher(
            cost_class=1,
            cost_dice=1,
            cost_focal=20
        )

        self.loss_matched_ce_weight = self.params.loss_matched_ce_weight
        self.loss_unmatched_ce_weight = self.params.loss_unmatched_ce_weight
        self.loss_dice_weight = self.params.loss_dice_weight
        self.loss_focal_weight = self.params.loss_focal_weight

        # Add safe global class
        from ruamel.yaml.scalarfloat import ScalarFloat
        torch.serialization.add_safe_globals([ScalarFloat])
    
        try:
            self.load_checkpoint(checkpoint_path, inference=True)
        except Exception as e:
            print(f"❌ Checkpoint loading failed: {str(e)}")
            return None
        
        self.down_model.eval()
        self.model.eval()
        print(f"✅ Model loaded from {checkpoint_path}")

        seg_target = []
        segmentation_result = []
        point_feature = []
        reg_target = []
        avg_loss = 0
        avg_ARI = 0
        
        with torch.no_grad():  # Disable gradient calculation
            for i, (grouped, label, knearest, reg) in enumerate(tqdm(self.val_data_loader)):
            #for i, (grouped, label, knearest) in enumerate(tqdm(self.train_data_loader)):
                #reg = 0
                #validate for 500 samples
                if i > 20000:
                    break
                b, c = grouped.size(0), grouped.size(-1)
                labels = label.to(self.device)
                grouped = grouped.reshape(b, -1, c).to(self.device)  # B X N X C
                mask = grouped[..., 0] != -100  # B X N
                # One-hot encode the labels using the inverse mapping
                #one_hot_labels = F.one_hot(inverse_indices, num_classes=n_gt_classes).float().to(self.device) # B X N X C_gt
                targets = []
                for batch_idx in range(b):
                    # Get labels for this sample
                    sample_labels = labels[batch_idx]
                    
                    # Find unique labels for this specific sample
                    unique_labels, inverse_indices = torch.unique(
                        sample_labels, 
                        sorted=True, 
                        return_inverse=True
                    )
                    n_gt_classes = unique_labels.numel()
                    
                    #if n_gt_classes > self.params.max_gt_classes:
                    #    continue
                        
                    # Create one-hot encoding for this sample
                    sample_one_hot = F.one_hot(inverse_indices, num_classes=n_gt_classes).float()
                    
                    # Convert to dictionary format
                    targets.append({
                        "masks": sample_one_hot.permute(1, 0).to(self.device),  # (n_gt_classes, N)
                        "labels": torch.ones(n_gt_classes, dtype=torch.long).to(self.device)  # (n_gt_classes,)
                    })

                if pretrain:
                    #print(grouped.size())
                    with torch.no_grad():
                        _, pre_embed, _ = self.model(grouped, return_z = True)
                    #feature = torch.stack(pre_embed).mean(0)
                    feature = torch.stack(pre_embed)
                    pred_dict = self.down_model(grouped, feature, pretrain=pretrain, padding_mask=mask) 
                    #pred_logit = self.down_model(grouped, feature, pretrain=pretrain) #B X N X C_classes
                    
                else:
                    pred_dict = self.down_model(grouped, feature=None)
                    #pred_logit = self.down_model(grouped, feature=None) #B X N X C_classes
                #softmax it to the prob
                #pred_probs = F.softmax(pred_logit, dim=-1) # B X N X C_classes
                pred_probs = pred_dict['mask_probs'] # (B, N, N_pred)
                class_probs = pred_dict['class_probs'] # (B, N_pred, 2)
        
                outputs = {
                    "pred_probs": class_probs,  
                    "pred_masks": pred_probs.permute(0, 2, 1) 
                }
                inference_result = assign_points_to_masks(outputs, option=2)
                segmentation_result.append(inference_result["assignments"])
                point_feature.append(grouped)
                reg_target.append(reg)
                seg_target.append(label)
                losses = compute_point_loss(
                    outputs=outputs,
                    targets=targets,
                    mask=mask,
                    matcher=self.matcher,
                    no_object_class=0
                )

                avg_ARI += adjusted_rand_score(inverse_indices.squeeze().cpu().numpy(), inference_result["assignments"].squeeze().cpu().numpy())
                # Compute loss and get matching indices
                loss = losses["loss_matched_ce"] * self.loss_matched_ce_weight + losses["loss_unmatched_ce"] * self.loss_unmatched_ce_weight + losses["loss_dice"] * self.loss_dice_weight + losses["loss_focal"] * self.loss_focal_weight
                avg_loss += loss
        
        print(avg_loss/len(segmentation_result))
        print(avg_ARI/len(segmentation_result))
        if logfile is not None:
            with open(logfile, "w") as f:
                f.write("Avg_Loss\tAvg_ARI\n")
                f.write(f"{avg_loss/len(segmentation_result)}\t{avg_ARI/len(segmentation_result)}\n")
        return segmentation_result, seg_target, point_feature, reg_target


    def train(self, pretrain = True, train_from_checkpoint = False, checkpoint_path = None):
        ###%%%%%%%
        # Debugging
        self.fwd_hooks = register_fine_grained_forward_hooks(self.model)
        self.bwd_hooks = register_param_backward_nan_hooks(self.model)
        ###%%%%%%%%

        def initialize_mamba2(model, num_layers, num_residuals=1):
            """ Properly initializes Mamba v2 to ensure stable learning. """
            for name, param in model.named_parameters():
            
                # Stable State-Space Matrix (A_t)
                if "A" in name:  
                    init.uniform_(param, -0.1 / num_layers, 0.1 / num_layers)
        
                # State Decay D (Ensure nonzero values)
                elif "D" in name:
                    init.normal_(param, mean=0.1, std=0.02)
        
                # Convolution Weights
                elif "conv1d.weight" in name:
                    init.kaiming_uniform_(param, mode="fan_in", nonlinearity="linear")
        
                # Projection Layers (Mapping Activations)
                elif "out_proj.weight" in name or "in_proj.weight" in name:
                    init.xavier_uniform_(param, gain=1.0 / (num_layers ** 0.5))
        
                # Normalization Layers (LayerNorm, RMSNorm)
                elif "norm.weight" in name:
                    init.ones_(param)
        
                # Bias Terms
                elif "bias" in name:
                    init.zeros_(param)

            if self.world_rank == 0:
                print(f"✅ Downstream Attention Head Initialized (Safe Scaling for {num_layers} Layers)")
                
    
        if self.world_rank == 0:
            print(f"Creating downstream head for {self.params.mambaversion} backbone...")

        self.down_model = MambaAttentionHead(input_dim=self.params.embed_dim, num_layers=0,
                                  num_embedder_layers= self.params.num_embedder_layers,
                                  d_state=64, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone, num_prototypes = self.params.max_gt_classes).to(self.device)

        initialize_mamba2(self.down_model, 3, num_residuals=1)

        total_params = sum(p.numel() for p in self.down_model.parameters())
        if self.world_rank == 0:
            print(f"Total parameters in downstream head: {total_params}")

        self.down_optimizer = optim.AdamW(self.down_model.parameters(), 
                                         lr=self.params.max_lr, # Mamba: Linear-Time Sequence Modeling with Selective State Spaces
                                         weight_decay=0.0001) 
        
        torch.nn.utils.clip_grad_norm_(self.down_model.parameters(), max_norm=1.0)

        # Ensure warmup_steps < first_cycle_steps for scheduler
        effective_warmup = min(self.params.warmup_steps, self.params.max_epochs - 1)
        self.down_scheduler = CosineAnnealingWarmupRestarts(self.down_optimizer,
                                          first_cycle_steps=self.params.max_epochs,
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=effective_warmup)
        # Initialize matcher ------------------------------------------------------
        self.matcher = PointHungarianMatcher(
            cost_class=self.params.loss_matched_ce_weight,
            cost_dice=self.params.loss_dice_weight,
            cost_focal=self.params.loss_focal_weight
        )

        #{'loss_matched_ce': tensor(1.2029), 'loss_unmatched_ce': tensor(7.6153), 'loss_dice': tensor(2.5837), 'loss_focal': tensor(0.5504)}

        self.loss_matched_ce_weight = self.params.loss_matched_ce_weight
        self.loss_unmatched_ce_weight = self.params.loss_unmatched_ce_weight
        self.loss_dice_weight = self.params.loss_dice_weight
        self.loss_focal_weight = self.params.loss_focal_weight


        # Add safe global class
        from ruamel.yaml.scalarfloat import ScalarFloat
        torch.serialization.add_safe_globals([ScalarFloat])
    
        if train_from_checkpoint:
            try:
                self.load_checkpoint(checkpoint_path, inference=False)
            except Exception as e:
                print(f"❌ Checkpoint loading failed: {str(e)}")
                return None
            
            self.down_model.eval()
            print(f"✅ Model loaded from {checkpoint_path}")

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.params.checkpoint_dir, exist_ok=True)

        log_file_path = os.path.join(self.params.checkpoint_dir, self.params.log_file_name)

        checkpoint_file_name = self.params.log_file_name.split('.')[0] + '_checkpoint.pth'

        if self.log_to_screen:
            print("Starting training loop...")

        # Always create log file and write header
        with open(log_file_path, "w") as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\tARI\tARI_2\tmatched_CE\tUnmatched_CE\tDice\tFocal\tTime\n")

        self.best_loss = np.inf
        self.best_ARI = 0
        self.down_results = {'epoch': 0, 'train': [], 'val': [], 'ARI': [], 'ARI_2': [],  'loss_matched_ce': [], 'loss_unmatched_ce': [], 'loss_dice': [], 'loss_focal': []}
        
        #early stopping
        self.patience = 15
        self.min_delta = 1e-4
        self.stagnation_counter = 0
        self.warmup_steps = 60

        self.loss_bin = pickle_load('{}/loss_bin_pp.pkl'.format(self.params.stat_dir))
        self.loss_weight = pickle_load('{}/loss_weight_pp.pkl'.format(self.params.stat_dir))
        
        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.down_results['epoch'] = epoch
            self.down_results['train'] = []
            self.down_results['val'] = []
            self.down_results['ARI'] = []
            self.down_results['ARI_2'] = []
            self.down_results['loss_matched_ce'] = []
            self.down_results['loss_unmatched_ce'] = []
            self.down_results['loss_dice'] = []
            self.down_results['loss_focal'] = []
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                
            self.resumed = False
                
            self.starttime = time.time()
            self.downstream_end_to_end_one_epoch(pretrain = pretrain)
            train_epoch_loss = np.mean(self.down_results['train'])
            val_epoch_loss = 0

            if epoch % 1 == 0:
                val_epoch_loss = self.validate_end_to_end_one_epoch(pretrain=pretrain)
            epoch_time = time.time() - self.starttime
            avg_matched_ce = np.mean(self.down_results['loss_matched_ce'])
            avg_unmatched_ce = np.mean(self.down_results['loss_unmatched_ce'])
            avg_dice = np.mean(self.down_results['loss_dice'])
            avg_focal = np.mean(self.down_results['loss_focal'])
            avg_ari = np.mean(self.down_results['ARI'])
            avg_ari_2 = np.mean(self.down_results['ARI_2'])

            # Log to file
            with open(log_file_path, "a") as f:  # Append mode
                f.write(f"{epoch}\t{train_epoch_loss:.8f}\t{val_epoch_loss:.8f}\t{avg_ari:.8f}\t{avg_ari_2:.8f}\t{avg_matched_ce:.8f}\t{avg_unmatched_ce:.8f}\t{avg_dice:.8f}\t{avg_focal:.8f}\t{epoch_time:.2f}\n")

            epoch_loss = val_epoch_loss

            # Print detailed metrics to screen
            if self.log_to_screen:
                print(f"Epoch {epoch}/{self.params.max_epochs-1} | Time: {epoch_time:.2f}s")
                print(f"  Train Loss: {train_epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")
                print(f"  ARI: {avg_ari:.6f} | ARI_2: {avg_ari_2:.6f}")
                print(f"  Matched CE: {avg_matched_ce:.6f} | Unmatched CE: {avg_unmatched_ce:.6f}")
                print(f"  Dice: {avg_dice:.6f} | Focal: {avg_focal:.6f}")
                print(f"  Best Loss: {self.best_loss:.6f} | Best ARI: {self.best_ARI:.6f}")
                print("-" * 80)
            #if (epoch_loss < self.best_loss) or (avg_ari_2 > self.best_ARI):
            if (epoch_loss < (self.best_loss - self.min_delta)):
                self.best_loss = epoch_loss
                self._save_checkpoint(
                    filename=checkpoint_file_name,
                    epoch=epoch,
                    is_best=True,
                    loss=epoch_loss
                )
                self.stagnation_counter = 0
            elif (avg_ari_2 > self.best_ARI):
                self.best_ARI = avg_ari_2
                self._save_checkpoint(
                    filename=checkpoint_file_name,
                    epoch=epoch,
                    is_best=True,
                    loss=epoch_loss
                )
            elif epoch>= self.warmup_steps:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch} due to no improvement in validation loss for {self.patience} epochs.")
                    print(f"Best validation loss: {self.best_loss:.4f}, current loss: {epoch_loss:.4f}")
                    break
            self.down_scheduler.step()

            

    def downstream_end_to_end_one_epoch(self, pretrain = False):
        tr_time = 0
        self.model.eval()
        self.down_model.train()
        # Buffers for logs
        tr_start = time.time()
        start_idx = 0
        for i, (grouped, label, knearest) in enumerate(tqdm(self.train_data_loader)):
            if i> 10000:
                break
            #only work for b ==1 now
            self.iters += 1
            b, c = grouped.size(0), grouped.size(-1)
            labels = label.to(self.device)# B X N
            grouped = grouped.reshape(b, -1, c).to(self.device) # B X N X C
            mask = grouped[..., 0] != -100 # B X N

            targets = []
            for batch_idx in range(b):
                # Get labels for this sample
                sample_labels = labels[batch_idx]
                
                # Find unique labels for this specific sample
                unique_labels, inverse_indices = torch.unique(
                    sample_labels, 
                    sorted=True, 
                    return_inverse=True
                )
                n_gt_classes = unique_labels.numel()
                
                #if n_gt_classes > self.params.max_gt_classes:
                #    continue
                    
                # Create one-hot encoding for this sample
                sample_one_hot = F.one_hot(inverse_indices, num_classes=n_gt_classes).float()
                
                # Convert to dictionary format
                targets.append({
                    "masks": sample_one_hot.permute(1, 0).to(self.device),  # (n_gt_classes, N)
                    "labels": torch.ones(n_gt_classes, dtype=torch.long).to(self.device)  # (n_gt_classes,)
                })
            

            self.down_optimizer.zero_grad()
            if pretrain:

                with torch.no_grad():
                    _, pre_embed, _ = self.model(grouped, return_z = True)
                #feature = torch.stack(pre_embed).mean(0)
                feature = torch.stack(pre_embed)
                #print('feature: ', feature.size())
                pred_dict = self.down_model(grouped, feature, pretrain=pretrain, padding_mask=mask)
                
            else:
                pred_dict = self.down_model(grouped, feature=None)
                #pred_logit = self.down_model(grouped, feature=None) #B X N X C_classes
            #softmax it to the prob
            #pred_probs = F.softmax(pred_logit, dim=-1) # B X N X C_classes
            pred_probs = pred_dict['mask_probs'] # (B, N, N_pred)
            class_probs = pred_dict['class_probs'] # (B, N_pred, 2)
     
            #outputs["pred_masks"] = pred_probs.permute(0, 2, 1)
            #outputs["pred_probs"] = class_probs

            outputs = {
                "pred_probs": class_probs,  
                "pred_masks": pred_probs.permute(0, 2, 1) 
            }
            #print('pred_probs: ', pred_probs.size(), 'class_probs: ', class_probs.size())
            losses = compute_point_loss(
                outputs=outputs,
                targets=targets,
                mask=mask,
                matcher=self.matcher,
                no_object_class=0
            )

            
            # Compute loss and get matching indices
            #loss = sum(losses.values())

            loss = losses["loss_matched_ce"] * self.loss_matched_ce_weight + losses["loss_unmatched_ce"] * self.loss_unmatched_ce_weight + losses["loss_dice"] * self.loss_dice_weight + losses["loss_focal"] * self.loss_focal_weight
            
            aux_list = pred_dict['aux_list']
            for aux_dict in aux_list:
                outputs = {
                    "pred_probs": aux_dict['class_probs'],  
                    "pred_masks": aux_dict['mask_probs'].permute(0, 2, 1) 
                }
                losses_aux = compute_point_loss(
                    outputs=outputs,
                    targets=targets,
                    mask=mask,
                    matcher=self.matcher,
                    no_object_class=0
                )
                loss_aux = losses_aux["loss_matched_ce"] * self.loss_matched_ce_weight + losses_aux["loss_unmatched_ce"] * self.loss_unmatched_ce_weight + losses_aux["loss_dice"] * self.loss_dice_weight + losses_aux["loss_focal"] * self.loss_focal_weight
                loss += loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.down_model.parameters(),  # Or specific parameters
                max_norm=1.0,  
                norm_type=2.0   
            )
            
            self.down_optimizer.step()
                
            self.down_results['train'].append(loss.item())

    def validate_end_to_end_one_epoch(self, pretrain=False):
        self.model.eval()  # Set backbone model to eval mode
        self.down_model.eval()  # Set downstream head to eval mode
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():  # Disable gradient calculation
            for i, (grouped, label, knearest) in enumerate(tqdm(self.val_data_loader)):
                #validate for 500 samples
                if i > 1000:
                    break
                b, c = grouped.size(0), grouped.size(-1)
                labels = label.to(self.device)
                grouped = grouped.reshape(b, -1, c).to(self.device)  # B X N X C
                mask = grouped[..., 0] != -100  # B X N
                # One-hot encode the labels using the inverse mapping
                targets = []
                inverse_indices_list = []
                for batch_idx in range(b):
                    # Get labels for this sample
                    sample_labels = labels[batch_idx]
                    
                    # Find unique labels for this specific sample
                    unique_labels, inverse_indices = torch.unique(
                        sample_labels, 
                        sorted=True, 
                        return_inverse=True
                    )
                    inverse_indices_list.append(inverse_indices)
                    n_gt_classes = unique_labels.numel()
                    
                    #if n_gt_classes > self.params.max_gt_classes:
                    #    continue
                        
                    # Create one-hot encoding for this sample
                    sample_one_hot = F.one_hot(inverse_indices, num_classes=n_gt_classes).float()
                    
                    # Convert to dictionary format
                    targets.append({
                        "masks": sample_one_hot.permute(1, 0).to(self.device),  # (n_gt_classes, N)
                        "labels": torch.ones(n_gt_classes, dtype=torch.long).to(self.device)  # (n_gt_classes,)
                    })

                if pretrain:
                    #print(grouped.size())
                    with torch.no_grad():
                        _, pre_embed, _ = self.model(grouped, return_z = True)
                    #feature = torch.stack(pre_embed).mean(0)
                    feature = torch.stack(pre_embed)
                    pred_dict = self.down_model(grouped, feature, pretrain=pretrain, padding_mask=mask)  
                    #pred_logit = self.down_model(grouped, feature, pretrain=pretrain) #B X N X C_classes
                    
                else:
                    pred_dict = self.down_model(grouped, feature=None)
                    #pred_logit = self.down_model(grouped, feature=None) #B X N X C_classes
                #softmax it to the prob
                #pred_probs = F.softmax(pred_logit, dim=-1) # B X N X C_classes
                pred_probs = pred_dict['mask_probs'] # (B, N, N_pred)
                class_probs = pred_dict['class_probs'] # (B, N_pred, 2)
        
                outputs = {
                    "pred_probs": class_probs,  
                    "pred_masks": pred_probs.permute(0, 2, 1) 
                }
    
                losses = compute_point_loss(
                    outputs=outputs,
                    targets=targets,
                    mask=mask,
                    matcher=self.matcher,
                    no_object_class=0
                )
                inference_result = assign_points_to_masks(outputs, option=1)
                segmentation_result = inference_result["assignments"]
                
                infrence_result_opt2 = assign_points_to_masks(outputs, option=2)
                segmentation_result_opt2 = infrence_result_opt2["assignments"]
                #calculate adjust rand score between segmentation result and label
                
                for j in range(b):
                    adjusted_rand_index = adjusted_rand_score(inverse_indices_list[j].cpu().numpy(), segmentation_result[j].cpu().numpy())
                    adjusted_rand_index_opt2 = adjusted_rand_score(inverse_indices_list[j].cpu().numpy(), segmentation_result_opt2[j].cpu().numpy())
                    

                adjusted_rand_index /= b
                adjusted_rand_index_opt2 /= b
                #print(adjusted_rand_index)
                # Compute loss and get matching indices
                loss = losses["loss_matched_ce"] * self.loss_matched_ce_weight + losses["loss_unmatched_ce"] * self.loss_unmatched_ce_weight + losses["loss_dice"] * self.loss_dice_weight + losses["loss_focal"] * self.loss_focal_weight

                #loss = (loss * mask.unsqueeze(-1)).sum(-1).sum(-1) / mask.sum(-1).sum(-1)
                #print(loss)
                #loss = loss.mean()
                self.down_results['val'].append(loss.item())
                self.down_results['ARI'].append(adjusted_rand_index)
                self.down_results['ARI_2'].append(adjusted_rand_index_opt2)
                self.down_results['loss_matched_ce'].append(losses["loss_matched_ce"].item())
                self.down_results['loss_unmatched_ce'].append(losses["loss_unmatched_ce"].item())
                self.down_results['loss_dice'].append(losses["loss_dice"].item())
                self.down_results['loss_focal'].append(losses["loss_focal"].item())



        # Final validation metrics
        avg_loss = np.mean(self.down_results['val'])

        # Print validation results
        if self.log_to_screen:
            print(f"\nValidation Loss: {avg_loss:.4f}")

        return avg_loss

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
    
        # 5. If not inference mode, load optimizer/scheduler states
        if not inference:
            self.down_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.down_scheduler is not None:
                self.down_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
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


    
            
    def restore_checkpoint(self, checkpoint_path, load_optimizer_state=True):
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer_state: If True, load optimizer/scheduler state (for resuming training).
                                 If False, only load model weights (for pretrained initialization).
        """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.device), weights_only=False)
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()}
        self.model.load_state_dict(new_state_dict)

        if load_optimizer_state:
            # Load optimizer and scheduler state for resuming training
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']+1 if self.iters % len(self.train_data_loader) == 0 else checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Only loaded model weights for pretrained initialization
            self.iters = 0
            self.startEpoch = 0
            if self.world_rank == 0:
                print(f"✅ Loaded pretrained weights only (optimizer state not loaded)")




