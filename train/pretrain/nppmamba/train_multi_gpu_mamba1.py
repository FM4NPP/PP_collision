"""
Mamba1 Training Script - Simplified Mamba (no μ-transfer, FP32)
Uses Mamba1GPT for O(n) complexity with state space model

Key features:
1. Uses Mamba1GPT with state space blocks
2. Standard AdamW optimizer
3. Target: 2M-5M parameters
4. FP32 training
5. Batch size adjusted for 40GB GPU
"""

import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from fm4npp.models.mambagpt import Mamba1GPT
from fm4npp.datasets.dataset_pretrain import get_data_loader
from fm4npp.utils import *

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_parameters(model):
    """Check for NaN or Inf in model parameters."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return f"NaN/Inf detected in {name}"
    return ""


def apply_bin_weights_torch(bin_list, weight_list, target):
    """Apply bin-based weighting for loss."""
    device = target.device
    weight_list = weight_list.to(device)
    bin_list = bin_list.to(device)

    target = target.unsqueeze(-1)
    mask_larger_than = (target >= bin_list[:-1].unsqueeze(0)).float()
    mask_smaller_than = (target < bin_list[1:].unsqueeze(0)).float()
    mask_in_bin = mask_larger_than * mask_smaller_than

    weights = (mask_in_bin * weight_list.unsqueeze(0)).sum(-1)
    return weights


class Trainer():
    def __init__(self, params, args):
        self.params = params
        self.args = args
        self.root_dir = args.root_dir
        self.config = args.config
        self.run_num = args.run_num

        # Initialize distributed training
        self.world_rank = 0
        self.local_rank = 0
        self.world_size = 1

        if dist.is_initialized():
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()

        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.local_rank)

        # Setup directories
        exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])
        self.exp_dir = exp_dir
        self.checkpoint_path = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')

        if self.world_rank == 0:
            if not os.path.isdir(self.exp_dir):
                os.makedirs(self.exp_dir, exist_ok=True)
                os.makedirs(os.path.join(self.exp_dir, 'training_checkpoints'), exist_ok=True)

        # Setup logging
        self.iters = 0
        self.startEpoch = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.logs = {}

        # Setup log files
        self.log_to_screen = True
        self.logfile = os.path.join(exp_dir, 'train.log')
        self.globalfile = os.path.join(exp_dir,
            'config_{}_run_{}.csv'.format(self.config, self.run_num))
        self.finisher = os.path.join(exp_dir, 'finished.txt')

        if self.world_rank == 0:
            if not os.path.exists(self.globalfile):
                with open(self.globalfile, 'w') as f:
                    f.write("split,step,loss,lr\n")

        # Load loss bin weights
        self.loss_bin = pickle_load('{}/loss_bin_pp.pkl'.format(self.params.stat_dir))
        self.loss_weight = pickle_load('{}/loss_weight_pp.pkl'.format(self.params.stat_dir))

        # Get data loaders
        self.train_data_loader, self.train_sampler, self.valid_data_loader, _ = \
            get_data_loader(self.params, dist.is_initialized())

        # Create model - Using Mamba1GPT  
        self.klen = self.params.klen
        d_state = getattr(self.params, 'd_state', 16)

        self.model = Mamba1GPT(
            embed_dim=self.params.embed_dim,
            num_layers=self.params.num_layers_backbone,
            d_state=d_state,
            klen=self.klen,
            dropout=self.params.dropout,
            embed_method=self.params.embed_method,
            pe_method=self.params.pe_method
        )

        # Standard initialization
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "norm.weight" in name:
                    init.ones_(param)
                elif "bias" in name:
                    init.zeros_(param)

        if self.world_rank == 0:
            print(f"✅ Mamba1GPT Model Initialized")
            print(f"   D state: {d_state}")

        self.model = self.model.to(self.device)

        if self.world_rank == 0:
            print(f'Nparams: {count_parameters(self.model):,}')

        # Distributed wrapper
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=[self.local_rank],
                find_unused_parameters=True
            )

        # Standard optimizer (no μ-transfer)
        # Simple AdamW with single learning rate for all parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.min_lr,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )

        if self.world_rank == 0:
            print(f"✅ Using standard AdamW optimizer (no μ-transfer scaling)")
            print(f"   Learning rate: {self.params.min_lr}")

        # No mixed precision for Mamba1 (testing simplification alone)

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.params.total_steps,
            cycle_mult=1.0,
            max_lr=self.params.max_lr,
            min_lr=self.params.min_lr,
            warmup_steps=self.params.warmup_steps,
            gamma=1.0
        )

        # Loss function
        self.loss_func = nn.MSELoss(reduction='none')
        self.loss_func_eval = nn.MSELoss(reduction='none')

        # Load checkpoint if exists
        self.restore_checkpoint()

    def log_infile(self, log):
        """Write log to file"""
        if self.world_rank == 0:
            with open(self.logfile, "a") as f:
                f.write("{}\n".format(log))

    def log_globalfile(self, split, step, loss, lr):
        """Write to global CSV log"""
        if self.world_rank == 0:
            with open(self.globalfile, "a") as f:
                f.write("{},{},{},{}\n".format(split, step, loss, lr))

    def init_exp_dir(self, exp_dir):
        """Initialize experiment directory"""
        if self.world_rank == 0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir, exist_ok=True)
                os.makedirs(os.path.join(exp_dir, 'training_checkpoints'), exist_ok=True)

    def restore_checkpoint(self):
        """Restore from checkpoint if exists"""
        if os.path.isfile(self.checkpoint_path):
            if self.world_rank == 0:
                print(f"Loading checkpoint from {self.checkpoint_path}")

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            try:
                self.model.load_state_dict(checkpoint['model_state'])
            except:
                if dist.is_initialized():
                    self.model.module.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint['model_state'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch'] + 1

            if self.world_rank == 0:
                print(f"Resuming from epoch {self.startEpoch}, iteration {self.iters}")

    def save_checkpoint(self, checkpoint_path, is_best=False):
        """Save checkpoint"""
        if self.world_rank != 0:
            return

        try:
            model_state = self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict()
        except:
            model_state = self.model.state_dict()

        torch.save({
            'iters': self.iters,
            'epoch': self.epoch,
            'model_state': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        if is_best:
            best_path = checkpoint_path.replace('.tar', '_best.tar')
            torch.save({
                'iters': self.iters,
                'epoch': self.epoch,
                'model_state': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, best_path)
            if self.world_rank == 0:
                print(f"Saved best checkpoint to {best_path}")
        else:
            if self.world_rank == 0:
                print(f"Saved checkpoint to {checkpoint_path}")

    def report_loss(self, loss, dist_state):
        """Report loss across distributed processes"""
        if dist_state:
            loss_tensor = loss.clone().detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            return loss_tensor.item()
        else:
            return loss.item()

    def train_one_epoch(self):
        """Train for one epoch"""
        tr_time = 0
        self.model.train()

        tr_start = time.time()

        for i, (grouped, _, knearest) in enumerate(self.train_data_loader):
            self.iters += 1
            b, c = grouped.size(0), grouped.size(-1)

            # Prepare targets
            targets = grouped.reshape(b, -1, 4)[:, :, 1:].to(self.device)
            klabel = knearest.reshape(b, -1, self.klen * 3).to(self.device)
            grouped = grouped.reshape(b, -1, c).to(self.device)

            self.model.zero_grad()

            # Forward pass (FP32 - no mixed precision)
            point_pred = self.model(grouped)

            # Handle rep_aaai logic
            if self.params.rep_aaai:
                if not self.params.nexttoken:
                    point_pred = point_pred[:, :-1]
                    targets = targets[:, 1:]
                    klabel = klabel[:, 1:]
                else:
                    point_pred = point_pred[:, :-2]
                    targets = targets[:, 2:]
                    klabel = klabel[:, 2:]

            kpred = point_pred
            kmask = klabel != -100
            tmask = targets[..., 0] != -100

            loss = self.loss_func(kpred, klabel)

            # Loss weighting
            if self.params.loss_reweight:
                loss = (loss * kmask).sum(-1).sum(-1) / kmask.sum(-1).sum(-1)
                loss_weight_ = apply_bin_weights_torch(
                    torch.Tensor(self.loss_bin).to(self.device),
                    torch.Tensor(self.loss_weight).to(self.device),
                    tmask.sum(-1)
                )
                loss = loss * loss_weight_
                loss = loss.mean()
            else:
                loss = (loss * kmask).sum() / kmask.sum()

            if self.params.ablate_loss_scale:
                loss = loss * self.params.ablate_loss_scale_rate

            # Backward pass (FP32)
            loss.backward()

            # Gradient clipping
            grad_norm = torch.zeros(1)
            clip_value = self.params.grad_clip_value
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            loss_log = '{},{},{},{},{}'.format(
                self.report_loss(loss, dist.is_initialized()),
                grad_norm.item(),
                tmask.sum(-1).float().mean().item(),
                tmask.sum(-1).float().std().item(),
                check_model_parameters(self.model) if self.iters % 100 == 0 or torch.isnan(loss) else ''
            )

            if self.world_rank == 0:
                if self.iters % 100 == 0:
                    print(f'Iter {self.iters}: loss={self.report_loss(loss, dist.is_initialized()):.4f}')
                self.log_globalfile('train', self.iters, self.report_loss(loss, dist.is_initialized()), self.scheduler.get_lr()[0])

            # Validation every n_eval_steps
            if self.iters % self.params.n_eval_steps == 0:
                tr_time += time.time() - tr_start
                self.val_one_epoch(tr_time)
                tr_start = time.time()

            # Stop at total steps
            if self.iters >= self.params.total_steps:
                break

        return 0

    def val_one_epoch(self, tr_time):
        """Validate for one epoch"""
        self.model.eval()
        val_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = logs_buff[0].view(-1)

        with torch.no_grad():
            for i, (grouped, _, knearest) in enumerate(self.valid_data_loader):
                b, c = grouped.size(0), grouped.size(-1)
                targets = grouped.reshape(b, -1, 4)[:, :, 1:].to(self.device)
                klabel = knearest.reshape(b, -1, self.klen * 3).to(self.device)
                grouped = grouped.reshape(b, -1, c).to(self.device)

                point_pred = self.model(grouped)

                if self.params.rep_aaai:
                    if not self.params.nexttoken:
                        point_pred = point_pred[:, :-1]
                        targets = targets[:, 1:]
                        klabel = klabel[:, 1:]
                    else:
                        point_pred = point_pred[:, :-2]
                        targets = targets[:, 2:]
                        klabel = klabel[:, 2:]

                kpred = point_pred
                kmask = klabel != -100
                tmask = targets[..., 0] != -100

                loss_kpred = self.loss_func_eval(kpred[kmask], klabel[kmask]).mean()
                loss = loss_kpred

                if self.params.ablate_loss_scale:
                    loss = loss * self.params.ablate_loss_scale_rate

                self.logs['val_loss'] += loss.detach()

        self.logs['val_loss'] /= len(self.valid_data_loader)

        if dist.is_initialized():
            dist.all_reduce(self.logs['val_loss'].detach())
            self.logs['val_loss'] = self.logs['val_loss'] / dist.get_world_size()

        val_time = time.time() - val_start

        # Track best model
        is_best_loss = False
        if self.logs['val_loss'] <= self.best_loss:
            is_best_loss = True
            self.best_loss = self.logs['val_loss']

        # Save checkpoint
        if self.params.save_checkpoint:
            self.save_checkpoint(self.checkpoint_path, is_best=is_best_loss)

        # Print and log results
        tolog = 'Time taken {:.2f} sec; with {:.2f} / {:.2f} in tr/val\n'.format(
            time.time() - self.starttime if hasattr(self, 'starttime') else val_time,
            tr_time, val_time)
        tolog += 'Step = {}, Val loss = {}'.format(self.iters, float(self.logs['val_loss']))

        if self.world_rank == 0 and self.log_to_screen:
            print(tolog)

        if self.world_rank == 0:
            self.log_infile(tolog)
            self.log_globalfile('val', self.iters, float(self.logs['val_loss']), self.scheduler.get_lr()[0])

        self.model.train()
        return 0

    def launch(self):
        """Launch training"""
        if self.world_rank == 0:
            print("="*80)
            print("STARTING LONGFORMER TRAINING")
            print("="*80)
            print(f"Model: Mamba1GPT")
            print(f"Embed dim: {self.params.embed_dim}")
            print(f"Num layers: {self.params.num_layers_backbone}")
            print(f"Num heads: {getattr(self.params, 'num_heads_backbone', 4)}")
            print(f"Window size: {getattr(self.params, 'window_size', 256)}")
            print(f"Learning rate: {self.params.min_lr}")
            print(f"Total steps: {self.params.total_steps}")
            print("="*80)

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if self.iters >= self.params.total_steps:
                break

            self.epoch = epoch
            self.starttime = time.time()

            if dist.is_initialized() and self.train_sampler:
                self.train_sampler.set_epoch(epoch)

            tr_time = self.train_one_epoch()

            if self.world_rank == 0:
                print(f"Epoch {epoch}: train_time={tr_time:.2f}s")

        if self.world_rank == 0:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field_train', type=str)
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--root_dir", default='/pscratch/sd/d/dpark1/NPFN/PRETRAIN_MAMBA', type=str)
    parser.add_argument("--global_log_dir", default='globallogs', type=str)

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    trainer = Trainer(params, args)
    trainer.launch()
